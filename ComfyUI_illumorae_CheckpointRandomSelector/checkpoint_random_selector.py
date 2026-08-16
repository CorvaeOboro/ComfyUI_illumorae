"""
illumorae Checkpoint Random Selector - a ComfyUI custom_node

Randomly selects a Diffusion checkpoint from a specified category (SDXL, PONY, SD15),
using a deterministic randomization at a time interval, such as every hour.

- choose a category (SDXL, PONY, SD15) and specify the sub folder for each.
- Randomly selects a checkpoint file (.safetensors or .sft) from the chosen category's folder.
- The selection is stable for the duration of the interval (e.g., 1 hour), so it only changes at interval boundaries.
- Outputs the folder path, full file path, and filename of the selected checkpoint.
- Useful for workflows that want to periodically rotate checkpoints

Inputs:
    base_folder: Root folder containing the per-category subfolders.
    category: Which checkpoint category to use (SDXL, PONY, SD15)
    interval_minutes: How often to randomize the selection (default: 60 minutes; 1-1440)
    sdxl_folder_name, pony_folder_name, sd15_folder_name: Subfolder name for each category
    safe_mode (optional): When True (default), only .safetensors/.sft are considered and
        file_extensions is ignored. When False, file_extensions is used instead.
    file_extensions (optional): Comma-separated extension list used only when
        safe_mode is False (default: ".safetensors,.sft"). A leading dot is optional.

Outputs:
    folder_path: The local folder path used
    file_path: The full file path to the selected checkpoint
    filename: The filename of the checkpoint

TITLE::Checkpoint Random Selector
DESCRIPTIONSHORT::Randomly selects a checkpoint from a category, changing each interval.
VERSION::20260815
GROUP::Checkpoint
GROUPORDER::2
LISTORDER::2
STATUS::working
IMAGE::comfyui_illumorae_checkpoint_random_selector.png
"""
#region IMPORTS
import hashlib
import os
import random
from datetime import datetime
#endregion


class illumoraeCheckpointRandomSelectorNode:

    #region CMETA
    # ComfyUI-facing metadata: input schema, return types, display fields,
    # and the canonical category tuple shared by validation and selection.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_folder": ("STRING", {"default": "D:/CODE/STABLEDIFFUSION_AUTO/models/Stable-diffusion"}),
                "category": ("STRING", {"default": "SDXL", "choices": ["SDXL", "PONY", "SD15"]}),
                "interval_minutes": ("INT", {"default": 60, "min": 1, "max": 1440}),
                "sdxl_folder_name": ("STRING", {"default": "SDXL 10"}),
                "pony_folder_name": ("STRING", {"default": "Pony"}),
                "sd15_folder_name": ("STRING", {"default": "SD15"}),
            },
            "optional": {
                "safe_mode": ("BOOLEAN", {"default": True}),
                "file_extensions": ("STRING", {"default": ".safetensors,.sft"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("folder_path", "file_path", "filename")
    FUNCTION = "select_checkpoint"
    CATEGORY = "illumorae"
    DESCRIPTION = "Randomly selects a checkpoint from a category, changing each interval."

    _CATEGORIES = ("SDXL", "PONY", "SD15")
    #endregion

    #region CBUCKET
    # Time-bucket helper: floors the current timestamp to the interval length
    # so every integer interval_minutes in [1, 1440] yields a consistent,
    # epoch-aligned bucket. Shared by IS_CHANGED and select_checkpoint.
    @staticmethod
    def _bucket_start(now, interval_minutes):
        epoch = int(now.timestamp())
        bucket_epoch = epoch - (epoch % (interval_minutes * 60))
        return datetime.fromtimestamp(bucket_epoch)
    #endregion

    #region CHOOKS
    # ComfyUI pre-execution hooks. VALIDATE_INPUTS runs before execution to
    # reject bad inputs; IS_CHANGED controls cache invalidation so the node
    # re-rolls exactly at interval boundaries and stays stable in between.
    @classmethod
    def IS_CHANGED(cls, category, interval_minutes, **kwargs):
        if category not in cls._CATEGORIES:
            return float("nan")
        bucket = int(datetime.now().timestamp()) // (interval_minutes * 60)
        return (category, interval_minutes, bucket)

    @classmethod
    def VALIDATE_INPUTS(cls, category, **kwargs):
        if category not in cls._CATEGORIES:
            return f"Unknown category: {category}. Expected one of {cls._CATEGORIES}."
        return True
    #endregion

    #region CSELECT
    # Main execution: resolve the category folder, collect and sort candidate
    # checkpoint files, derive a deterministic seed from the current interval
    # bucket, and pick one file via an isolated RNG instance.
    def select_checkpoint(self, base_folder, category, interval_minutes, sdxl_folder_name, pony_folder_name, sd15_folder_name, safe_mode=True, file_extensions=".safetensors,.sft"):
        folder_name_map = {
            "SDXL": sdxl_folder_name,
            "PONY": pony_folder_name,
            "SD15": sd15_folder_name,
        }
        if category not in folder_name_map:
            raise ValueError(f"Unknown category: {category}. Expected one of {self._CATEGORIES}.")
        folder_name = folder_name_map[category]
        folder = os.path.abspath(os.path.join(base_folder, folder_name))
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Checkpoint folder does not exist: {folder}")

        # safe_mode forces a known-good extension set and ignores file_extensions;
        # see the module docstring for the full interaction.
        if safe_mode:
            exts_lower = (".safetensors", ".sft")
        else:
            exts_lower = tuple(
                (ext.strip() if ext.strip().startswith(".") else "." + ext.strip()).lower()
                for ext in file_extensions.split(",")
                if ext.strip()
            )
        files = sorted(
            f
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and any(f.lower().endswith(ext) for ext in exts_lower)
        )
        if not files:
            raise FileNotFoundError(f"No checkpoint files found in {folder}")
        # Stable seed for the current interval bucket: deterministic across
        # processes (hashlib, not hash()) and isolated from the global RNG
        # (local random.Random, not random.seed()).
        interval_start = self._bucket_start(datetime.now(), interval_minutes)
        seed_material = f"{category}|{interval_start:%Y-%m-%d %H:%M}"
        seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest(), "big")
        rng = random.Random(seed)
        filename = rng.choice(files)
        file_path = os.path.join(folder, filename)
        return folder, file_path, filename
    #endregion


#region REGISTRY
# ComfyUI node registration mappings.
NODE_CLASS_MAPPINGS = {
    'illumoraeCheckpointRandomSelectorNode': illumoraeCheckpointRandomSelectorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeCheckpointRandomSelectorNode': 'Checkpoint Random Selector',
}
#endregion
