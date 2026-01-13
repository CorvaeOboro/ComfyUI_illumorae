"""
illumorae Checkpoint Random Selector - a ComfyUI custom_node

Randomly selects a Diffusion checkpoint from a specified category (SDXL, PONY, SD15),
using a deterministic randomization at a time interval , such as every hour .

- choose a category (SDXL, PONY, SD15) and specify the sub folder for each.
- Randomly selects a checkpoint file ( .safetensors or .sft ) from the chosen category's folder.
- The selection is stable for the duration of the interval (e.g., 1 hour), so it only changes at interval boundaries.
- Outputs the folder path, full file path, and filename of the selected checkpoint.
- Useful for workflows that want to periodically rotate checkpoints

Inputs:
    category: Which checkpoint category to use (SDXL, PONY, SD15)
    interval_minutes: How often to randomize the selection (default: 60 minutes)
    sdxl_folder, pony_folder, sd15_folder: Paths to the checkpoint folders for each category

Outputs:
    folder_path: The local folder path used
    file_path: The full file path to the selected checkpoint
    filename: The filename of the checkpoint

TITLE::Checkpoint Random Selector
DESCRIPTIONSHORT::Randomly selects a checkpoint from a category, changing each interval.
VERSION::20260113
GROUP::Checkpoint
"""
import os
import random
from datetime import datetime, timedelta

class illumoraeCheckpointRandomSelector:
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
    DESCRIPTION = "Randomly selects a checkpoint from a category, changing only every interval. Outputs: folder, full path, filename."

    def select_checkpoint(self, base_folder, category, interval_minutes, sdxl_folder_name, pony_folder_name, sd15_folder_name, safe_mode=True, file_extensions=".safetensors,.sft"):
        # Map category to folder name
        folder_name_map = {
            "SDXL": sdxl_folder_name,
            "PONY": pony_folder_name,
            "SD15": sd15_folder_name,
        }
        folder_name = folder_name_map.get(category, sdxl_folder_name)
        folder = os.path.abspath(os.path.join(base_folder, folder_name))
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Checkpoint folder does not exist: {folder}")

        if safe_mode:
            exts = (".safetensors", ".sft")
        else:
            exts = tuple(
                ext.strip() if ext.strip().startswith(".") else "." + ext.strip()
                for ext in file_extensions.split(",")
                if ext.strip()
            )

        exts_lower = tuple(ext.lower() for ext in exts)
        files = [
            f
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and any(f.lower().endswith(ext) for ext in exts_lower)
        ]
        if not files:
            raise FileNotFoundError(f"No checkpoint files found in {folder}")
        # Use datetime and interval to create a stable seed
        now = datetime.now()
        interval_start = now - timedelta(minutes=now.minute % interval_minutes, seconds=now.second, microseconds=now.microsecond)
        seed = hash((category, interval_start.strftime('%Y-%m-%d %H:%M')))
        random.seed(seed)
        filename = random.choice(files)
        file_path = os.path.join(folder, filename)
        return folder, file_path, filename

NODE_CLASS_MAPPINGS = {
    'illumoraeCheckpointRandomSelector': illumoraeCheckpointRandomSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeCheckpointRandomSelector': 'Checkpoint Random Selector',
}
