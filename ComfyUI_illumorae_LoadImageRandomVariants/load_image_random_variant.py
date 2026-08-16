r"""
illumorae Image Load Random Variant - a ComfyUI custom_node

Load image from a folderpath, with options to load one of its variants.
Variants may be in a subfolder or have a suffix, possibly many variants.
If no variants found, just load the input image.

Now, you only need to pass in the folder path,
the filename (without extension), and specify the extension (defaults to .png).

Example
D:\items\potionA.png
D:\items\potionA\potionA_CAM_ORTHO_PROJ_1.png
D:\items\potionA\potionA_CAM_ORTHO_PROJ_2.png

the base image is potionA.png
the variants are in a subfolder based on the image name, with suffix input
there may be an unknown amount of variants with suffix number increasing from 1
there should be a boolean whether to look for variants or not


TITLE::Load Image Random Variant
DESCRIPTIONSHORT::Loads an image from a folder with optional variant selection (subfolder + suffix pattern), with seed and override controls.
VERSION::20260127
IMAGE::comfyui_illumorae_load_image_random_variant.png
GROUP::Load
GROUPORDER::3
LISTORDER::3
STATUS::working
"""

# region imports
import os
import random
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths  # noqa: F401  # kept for ComfyUI runtime availability

# endregion imports


# region helpers
def _natural_sort_key(path: Path):
    """Sort key that orders numeric substrings by their integer value.

    e.g. potionA_CAM_ORTHO_PROJ_2.png < potionA_CAM_ORTHO_PROJ_10.png
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


# endregion helpers


# region nodecls
class illumoraeLoadImageRandomVariantNode:
    # region core
    def load_image(
        self,
        folder,
        base_filename,
        extension,
        variant_suffixes,
        search_variants,
        seed,
        variant_index_override,
        debug_mode,
    ):
        # region C_valid
        # Input validation to avoid NoneType errors
        if folder is None or not isinstance(folder, str):
            raise ValueError("'folder' must be a non-empty string")
        if base_filename is None or not isinstance(base_filename, str):
            raise ValueError("'base_filename' must be a non-empty string")
        if extension is None or not isinstance(extension, str):
            raise ValueError("'extension' must be a non-empty string")
        if variant_suffixes is None or not isinstance(variant_suffixes, str):
            raise ValueError("'variant_suffixes' must be a non-empty string")
        # Random seed logic: a negative seed draws a fresh random seed so each run
        # picks a different variant; a non-negative seed is reproducible. Use a local
        # Random instance so the process-global RNG is not disturbed.
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
        rng = random.Random(seed)
        self.debug_print(debug_mode, "[load_image] Called with:")
        self.debug_print(debug_mode, "  folder:", folder)
        self.debug_print(debug_mode, "  base_filename (user input):", base_filename)
        self.debug_print(debug_mode, "  extension (user input):", extension)
        self.debug_print(debug_mode, "  variant_suffixes (raw):", variant_suffixes)
        self.debug_print(debug_mode, "  search_variants:", search_variants)
        # endregion C_valid

        # region C_paths
        # Normalize extension: ensure it starts with a dot.
        if not extension.startswith("."):
            self.debug_print(
                debug_mode, f"  WARNING: extension '{extension}' does not start with a dot. Prepending dot."
            )
            extension = f".{extension}"
        ext = extension

        # Check for accidental extension in base_filename
        stem, ext_in_name = os.path.splitext(base_filename)
        if ext_in_name:
            self.debug_print(
                debug_mode,
                f"  WARNING: base_filename '{base_filename}' includes extension '{ext_in_name}'. This will be ignored and '{ext}' will be used instead.",
            )
            base_filename = stem  # strip extension
        else:
            stem = base_filename
        self.debug_print(debug_mode, f"  Using stem: '{stem}', extension: '{ext}'")

        # Enhanced logging for parsing
        if "," in variant_suffixes:
            self.debug_print(debug_mode, "  Detected ',' in variant_suffixes (splitting on commas)")
        if "=" in variant_suffixes:
            self.debug_print(debug_mode, "  Detected '=' in variant_suffixes (possible assignment or error?)")

        folder = Path(folder)
        base = folder / f"{stem}{ext}"
        self.debug_print(debug_mode, "  base path:", base)
        self.debug_print(debug_mode, "  stem:", stem, "ext:", ext)

        suffixes = [s.strip() for s in variant_suffixes.split(",") if s.strip()]
        self.debug_print(debug_mode, "  Parsed suffixes:", suffixes)
        for idx, sfx in enumerate(suffixes):
            if "=" in sfx:
                self.debug_print(debug_mode, f"    Suffix {idx} contains '=': {sfx}")
        # endregion C_paths

        # region C_discover
        # Variant discovery: glob the variant subfolder for each suffix pattern,
        # natural-sort matches so numbering gaps do not drop later variants.
        variants = []
        if search_variants:
            variant_folder = folder / stem
            self.debug_print(debug_mode, "  Looking for variants in folder:", variant_folder)
            if variant_folder.exists() and variant_folder.is_dir():
                for suffix in suffixes:
                    # Glob all files matching {stem}{suffix}*{ext} and natural-sort them
                    # so numbering gaps (e.g. _1, _2, _5) do not drop later variants.
                    pattern = f"{stem}{suffix}*{ext}"
                    matches = [p for p in variant_folder.glob(pattern) if p.is_file()]
                    matches.sort(key=_natural_sort_key)
                    for m in matches:
                        self.debug_print(debug_mode, f"    Found variant: {m}")
                    variants.extend(matches)
                    if not matches:
                        self.debug_print(debug_mode, f"    No match for pattern: {pattern}")
            else:
                self.debug_print(debug_mode, "    Variant folder does not exist or is not a directory.")
        # Always include the base image as a possible variant if it exists.
        # base lives in `folder`, every entry in `variants` lives in `folder / stem`,
        # so base can never already be in variants.
        if base.exists():
            variants.append(base)
            self.debug_print(debug_mode, f"  Added base image to variants: {base}")
        self.debug_print(debug_mode, "  Variants found:", [str(v) for v in variants])
        # endregion C_discover

        # region C_select
        # Variant selection: explicit 1-based override wins; otherwise pick randomly.
        chosen = None

        self.debug_print(
            debug_mode,
            f"[load_image] Variant selection: variants count={len(variants)}, override={variant_index_override}",
        )

        # 1. If override is set and valid, use it. Indexing is 1-based for UI friendliness:
        #    -1 (or any value < 1) means "no override, pick randomly". 0 is not a valid
        #    1-based index, so it is treated as "no override" with a warning.
        if variant_index_override is not None and variant_index_override > 0:
            idx = variant_index_override - 1
            self.debug_print(
                debug_mode,
                f"[load_image] Override mode: index={variant_index_override}, array_index={idx}, variants_len={len(variants)}",
            )
            if 0 <= idx < len(variants):
                chosen = variants[idx]
                self.debug_print(
                    debug_mode, f"  Override: Using variant index {variant_index_override} (file: {chosen})"
                )
            else:
                self.debug_print(
                    debug_mode,
                    f"  WARNING: variant_index_override={variant_index_override} is out of range (1-{len(variants)}). Falling back to random selection.",
                )
        elif variant_index_override == 0:
            self.debug_print(
                debug_mode,
                "  WARNING: variant_index_override=0 is not a valid 1-based index (use >=1, or -1 for random). Treating as random.",
            )
        # 2. Else pick randomly, always seeded for reproducibility
        if chosen is None and variants:
            self.debug_print(debug_mode, f"  Using seed: {seed} for random selection.")
            chosen = rng.choice(variants)
            self.debug_print(debug_mode, f"  Randomly chose variant: {chosen}")

        self.debug_print(debug_mode, f"[load_image] After selection: chosen={chosen}, type={type(chosen)}")

        if chosen is None:
            # No valid image found: raise clear error
            msg = f"No valid base image or variant found for base: '{base}'. Please check your folder, filename, extension, and variant_suffixes settings."
            self.debug_print(debug_mode, f"  ERROR: {msg}")
            raise FileNotFoundError(msg)

        # Verify chosen file exists before trying to open it
        if not chosen.exists():
            msg = f"Selected image file does not exist: '{chosen}'. File was in variants list but cannot be accessed."
            self.debug_print(debug_mode, f"  ERROR: {msg}")
            raise FileNotFoundError(msg)
        # endregion C_select

        # region C_load
        # Image loading: open, EXIF-orient, convert to tensor, derive mask.
        self.debug_print(debug_mode, f"  Opening image file: {chosen}")
        i = Image.open(chosen)
        self.debug_print(debug_mode, f"  Opened image: {chosen}")
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if "A" in i.getbands():
            self.debug_print(debug_mode, "  Image has alpha channel. Generating mask from alpha.")
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            self.debug_print(debug_mode, "  No alpha channel. Using default mask.")
            mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")
        folder_path = str(chosen.parent)
        file_name = chosen.stem
        self.debug_print(debug_mode, f"  Output file_name: {file_name}, folder_path: {folder_path}")
        return (image, mask, file_name, folder_path)
        # endregion C_load

    # endregion core

    # region util
    def debug_print(self, debug_mode, *args, **kwargs):
        """Print only if debug_mode is True."""
        if debug_mode:
            print(*args, **kwargs)

    # endregion util

    # region ui
    # ComfyUI input declarations and node identity metadata.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": r"C:/input"}),
                "base_filename": ("STRING", {"default": "image"}),  # no extension
                "extension": ("STRING", {"default": ".png"}),
                "variant_suffixes": ("STRING", {"default": "_CAM_ORTHO_PROJ_"}),
                "search_variants": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1}),  # -1 means random
                "variant_index_override": ("INT", {"default": -1, "min": -1}),  # -1 = random; >=1 = 1-based pick
                "debug_mode": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "illumorae"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "file name", "folder path")
    FUNCTION = "load_image"
    DESCRIPTION = "Loads an image from a folder with optional variant selection (subfolder + suffix pattern), with seed/override controls."
    # endregion ui

    # region lifecycle
    # ComfyUI validation and caching hooks.
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        folder = kwargs.get("folder")
        base_filename = kwargs.get("base_filename")
        if not folder or not isinstance(folder, str):
            return "'folder' must be a non-empty string"
        if not base_filename or not isinstance(base_filename, str):
            return "'base_filename' must be a non-empty string"
        if not Path(folder).exists():
            return f"Invalid folder path: {folder}"
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This is a random-variant loader: re-execute on every run so a new
        # variant can be picked. With an explicit 1-based override the result
        # is stable, so caching is safe.
        override = kwargs.get("variant_index_override", -1)
        if override is not None and override > 0:
            return False
        return float("nan")

    # endregion lifecycle


# endregion nodecls


# region register
NODE_CLASS_MAPPINGS = {
    "illumoraeLoadImageRandomVariantNode": illumoraeLoadImageRandomVariantNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeLoadImageRandomVariantNode": "Load Image Random Variant",
}
# endregion register
