"""
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

TODO:
variants are typically a render from a camera of a texture (on the 3d mesh) and therefore we would want to disable tiling
random seed
overide 
"""

import os
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths


class illumoraeLoadImageRandomVariant:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": ("STRING", {"default": r"C:/input"}),
            "base_filename": ("STRING", {"default": "image"}),  # no extension
            "extension": ("STRING", {"default": ".png"}),
            "variant_suffixes": ("STRING", {"default": "_CAM_ORTHO_PROJ_"}),
            "search_variants": ("BOOLEAN", {"default": True}),
            "seed": ("INT", {"default": -1, "min": -1}),  # -1 means random
            "variant_index_override": ("INT", {"default": -1, "min": -1}),  # -1 means random
            
            "debug_mode": ("BOOLEAN", {"default": False})
        }}

    def debug_print(self, debug_mode, *args, **kwargs):
        """Print only if debug_mode is True."""
        if debug_mode:
            print(*args, **kwargs)


    CATEGORY = "illumorae"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "MASK", "file name", "folder path")
    FUNCTION = "load_image"
    DESCRIPTION = "Loads an image or its variants from a specified file path string and outputs the image, mask, file name, and folder path."

    def load_image(self, folder, base_filename, extension, variant_suffixes, search_variants, seed, variant_index_override, debug_mode):
        # Input validation to avoid NoneType errors
        if folder is None or not isinstance(folder, str):
            raise ValueError("'folder' must be a non-empty string")
        if base_filename is None or not isinstance(base_filename, str):
            raise ValueError("'base_filename' must be a non-empty string")
        if extension is None or not isinstance(extension, str):
            raise ValueError("'extension' must be a non-empty string")
        if variant_suffixes is None or not isinstance(variant_suffixes, str):
            raise ValueError("'variant_suffixes' must be a non-empty string")
        # Random seed logic: always seed with seed for reproducibility (ComfyUI pattern)
        random.seed(seed)
        # Remove randomize_seed_after logic for strict reproducibility.
        self.debug_print(debug_mode, "[load_image] Called with:")
        self.debug_print(debug_mode, "  folder:", folder)
        self.debug_print(debug_mode, "  base_filename (user input):", base_filename)
        self.debug_print(debug_mode, "  extension (user input):", extension)
        self.debug_print(debug_mode, "  variant_suffixes (raw):", variant_suffixes)
        self.debug_print(debug_mode, "  search_variants:", search_variants)

        # Input validation and warnings
        if not isinstance(extension, str):
            self.debug_print(debug_mode, f"  WARNING: extension should be a string, got {type(extension)}. Forcing to '.png'.")
            extension = ".png"
        if not extension.startswith('.'):
            self.debug_print(debug_mode, f"  WARNING: extension '{extension}' does not start with a dot. Prepending dot.")
            extension = f'.{extension}'
        if not isinstance(variant_suffixes, str):
            self.debug_print(debug_mode, f"  WARNING: variant_suffixes should be a string, got {type(variant_suffixes)}. Forcing to '_CAM_ORTHO_PROJ_'.")
            variant_suffixes = "_CAM_ORTHO_PROJ_"

        # Check for accidental extension in base_filename
        stem, ext_in_name = os.path.splitext(base_filename)
        ext = extension if extension.startswith('.') else f'.{extension}'
        if ext_in_name:
            self.debug_print(debug_mode, f"  WARNING: base_filename '{base_filename}' includes extension '{ext_in_name}'. This will be ignored and '{ext}' will be used instead.")
            base_filename = stem  # strip extension
        else:
            stem = base_filename
        self.debug_print(debug_mode, f"  Using stem: '{stem}', extension: '{ext}'")

        # Enhanced logging for parsing
        if ',' in variant_suffixes:
            self.debug_print(debug_mode, "  Detected ',' in variant_suffixes (splitting on commas)")
        if '=' in variant_suffixes:
            self.debug_print(debug_mode, "  Detected '=' in variant_suffixes (possible assignment or error?)")

        folder = Path(folder)
        base = folder / f"{stem}{ext}"
        self.debug_print(debug_mode, "  base path:", base)
        self.debug_print(debug_mode, "  stem:", stem, "ext:", ext)

        suffixes = [s.strip() for s in variant_suffixes.split(",") if s.strip()]
        self.debug_print(debug_mode, "  Parsed suffixes:", suffixes)
        for idx, sfx in enumerate(suffixes):
            if '=' in sfx:
                self.debug_print(debug_mode, f"    Suffix {idx} contains '=': {sfx}")

        variants = []
        if search_variants:
            variant_folder = folder / stem
            self.debug_print(debug_mode, "  Looking for variants in folder:", variant_folder)
            if variant_folder.exists() and variant_folder.is_dir():
                for suffix in suffixes:
                    i = 1
                    while True:
                        candidate = variant_folder / f"{stem}{suffix}{i}{ext}"
                        if candidate.exists():
                            self.debug_print(debug_mode, f"    Found variant: {candidate}")
                            variants.append(candidate)
                            i += 1
                        else:
                            self.debug_print(debug_mode, f"    No match for: {candidate}")
                            break
            else:
                self.debug_print(debug_mode, "    Variant folder does not exist or is not a directory.")
        # Always include the base image as a possible variant if it exists and not already present
        if base.exists() and base not in variants:
            variants.append(base)
            self.debug_print(debug_mode, f"  Added base image to variants: {base}")
        self.debug_print(debug_mode, "  Variants found:", [str(v) for v in variants])

        # Variant selection logic
        chosen = None
        variant_index_used = None
        
        print(f"[load_image] Variant selection: variants count={len(variants)}, override={variant_index_override}")
        
        # 1. If override is set and valid, use it (1-based index for UI friendliness)
        if variant_index_override is not None and variant_index_override > 0:
            idx = variant_index_override - 1
            print(f"[load_image] Override mode: index={variant_index_override}, array_index={idx}, variants_len={len(variants)}")
            if 0 <= idx < len(variants):
                chosen = variants[idx]
                variant_index_used = idx + 1
                print(f"[load_image] Override: Selected variant at index {variant_index_override}: {chosen}")
                self.debug_print(debug_mode, f"  Override: Using variant index {variant_index_override} (file: {chosen})")
            else:
                print(f"[load_image] Override index {variant_index_override} out of range (0-{len(variants)-1})")
                self.debug_print(debug_mode, f"  WARNING: variant_index_override={variant_index_override} is out of range. Falling back to random selection.")
        # 2. Else pick randomly, always seeded for reproducibility
        if chosen is None and variants:
            print(f"[load_image] Random selection mode (chosen is None, variants exist)")
            self.debug_print(debug_mode, f"  Using seed: {seed} for random selection.")
            chosen = random.choice(variants)
            variant_index_used = variants.index(chosen) + 1
            print(f"[load_image] Randomly selected: {chosen}")
            self.debug_print(debug_mode, f"  Randomly chose variant: {chosen} (index {variant_index_used})")
        elif chosen is None and not variants and base.exists():
            # This branch should never be hit now, but keep for safety
            print(f"[load_image] Fallback: No variants, using base image")
            chosen = base
            variant_index_used = 0
            self.debug_print(debug_mode, f"  No variants found. Using base image: {chosen}")
        
        print(f"[load_image] After selection: chosen={chosen}, type={type(chosen)}")
        
        if chosen is None:
            # No valid image found: raise clear error
            msg = f"No valid base image or variant found for base: '{base}'. Please check your folder, filename, extension, and variant_suffixes settings."
            print(f"[load_image] ERROR: {msg}")
            self.debug_print(debug_mode, f"  ERROR: {msg}")
            raise FileNotFoundError(msg)
        
        # Verify chosen file exists before trying to open it
        if not chosen.exists():
            msg = f"Selected image file does not exist: '{chosen}'. File was in variants list but cannot be accessed."
            self.debug_print(debug_mode, f"  ERROR: {msg}")
            raise FileNotFoundError(msg)
        
        self.debug_print(debug_mode, f"  Opening image file: {chosen}")
        i = Image.open(chosen)
        self.debug_print(debug_mode, f"  Opened image: {chosen}")
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            self.debug_print(debug_mode, "  Image has alpha channel. Generating mask from alpha.")
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            self.debug_print(debug_mode, "  No alpha channel. Using default mask.")
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        dirname, basename = os.path.split(str(chosen))
        file_name = self.get_file_name_without_extension(str(chosen))
        folder_path = dirname
        self.debug_print(debug_mode, f"  Output file_name: {file_name}, folder_path: {folder_path}")
        return (image, mask, file_name, folder_path)


    @staticmethod
    def get_file_name_without_extension(file_path):
        file_name_with_extension = os.path.basename(file_path)
        file_name, _ = os.path.splitext(file_name_with_extension)
        return file_name

    @staticmethod
    def _resolve_path(image) -> Path:
        image_path = Path(folder_paths.get_annotated_filepath(image))
        return image_path

    @classmethod
    def IS_CHANGED(cls, image=None, **kwargs):
        import hashlib
        if image is None:
            # Return a default value or None if image is not provided
            print("WARNING: illumoraeLoadImageRandomVariant.IS_CHANGED() called without 'image' argument.")
            return None
        image_path = cls._resolve_path(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image=None, **kwargs):
        # If image is an output of another node, it will be None during validation
        if image is None:
            return True

        image_path = cls._resolve_path(image)
        if not image_path.exists():
            return f"Invalid image path: {image_path}"

        return True


NODE_CLASS_MAPPINGS = {
    'illumoraeLoadImageRandomVariant': illumoraeLoadImageRandomVariant,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeLoadImageRandomVariant': 'Load Image Random Variant',
}
