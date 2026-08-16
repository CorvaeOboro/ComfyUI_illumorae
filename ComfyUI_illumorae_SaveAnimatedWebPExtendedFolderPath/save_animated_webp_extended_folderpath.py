"""
Save Animated WEBP FolderPath - a custom node for ComfyUI
webp is ideal video format , loops like gif with lossless quality , contains the comfy workflow in metadata
this version is a minor modification of the core node with added inputs for saving to an external project structure

TITLE::Save Animated WEBP FolderPath
DESCRIPTIONSHORT::Saves an animated WebP to an external folder path
VERSION::20260201
IMAGE::comfyui_illumorae_save_animated_webp_folderpath.png
GROUP::Save
GROUPORDER::7
LISTORDER::1
STATUS::working
WORKFLOWNOTE::Saves to an external folder path; the ComfyUI preview panel will not populate because the file lives outside the managed output directory. Use the returned filepath output or the console log line to locate the saved WebP.
"""
#region imports
import os
import json
import string
from datetime import datetime
from PIL import Image
import numpy as np
import folder_paths
from comfy.cli_args import args
#endregion

class illumoraeSaveAnimatedWEBPFolderPathNode:
    #region classmeta
    """Saves an animated WebP to an external folder path."""
    TITLE = "Save Animated WEBP FolderPath"
    methods = {"default": 4, "fastest": 0, "slowest": 6}

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
    #endregion

    #region core helpers
    # Sanitization, folder creation, filename formatting, counter scanning.
    # These are the building blocks used by save_images below.

    @staticmethod
    def sanitize_name(name):
        """Replace illegal Windows path characters with underscore, strip whitespace, drop control chars."""
        invalid_chars = '<>:"/\\|?*'
        sanitized = ''.join('_' if c in invalid_chars else c for c in name)
        sanitized = sanitized.strip()
        sanitized = ''.join(c for c in sanitized if c in string.printable and ord(c) >= 32)
        return sanitized

    def create_folder_structure(self, base_path, folder_name):
        """Create folder structure and return full path"""
        # Ensure base path exists and is absolute
        base_path = os.path.abspath(base_path)
        if not os.path.exists(base_path):
            print(f"[SaveAnimatedWEBPExtendedFolderPath] Creating base directory: {base_path}")
            os.makedirs(base_path, exist_ok=True)

        # Create target folder
        full_path = os.path.join(base_path, folder_name)
        os.makedirs(full_path, exist_ok=True)
        print(f"[SaveAnimatedWEBPExtendedFolderPath] Using output directory: {full_path}")

        return full_path

    def format_counter_filename(self, base_filename, counter, counter_digits, counter_position="last"):
        """Format a filename from a base name and counter. Does not check uniqueness."""
        webp_ext = ".webp"
        counter_str = str(counter).zfill(counter_digits)

        if counter_position == "first":
            filename = f"{counter_str}_{base_filename}{webp_ext}"
        else:  # last
            filename = f"{base_filename}_{counter_str}{webp_ext}"

        return filename

    def get_latest_counter(self, folder_path, filename_prefix, counter_digits, counter_position='last'):
        """Return the next counter value by scanning existing .webp files once."""
        counter = 1
        if not os.path.exists(folder_path):
            return counter

        try:
            files = [f for f in os.listdir(folder_path) if f.endswith('.webp')]
            if files:
                if counter_position == 'first':
                    counters = [
                        int(f[:counter_digits]) if f[:counter_digits].isdigit() else 0
                        for f in files
                        if f[counter_digits + 1:].startswith(filename_prefix)
                    ]
                else:  # last
                    counters = [
                        int(f[-(5 + counter_digits):-5]) if f[-(5 + counter_digits):-5].isdigit() else 0
                        for f in files
                        if f.startswith(filename_prefix)
                    ]

                if counters:
                    counter = max(counters) + 1
        except Exception as e:
            print(f"[SaveAnimatedWEBPExtendedFolderPath] Error scanning existing counters: {e}")

        return counter
    #endregion

    #region main save
    def save_images(self, images, filename_prefix, folderpath_input, foldername_prefix, fps,
                   lossless, quality, method, save_metadata="enabled", counter_digits=3,
                   counter_position="last", prompt=None, extra_pnginfo=None):
        try:
            #region resolve
            # Resolve method enum, sanitize names, fall back on empty folderpath.
            method = self.methods.get(method, self.methods["default"])

            filename_prefix = self.sanitize_name(filename_prefix)
            foldername_prefix = self.sanitize_name(foldername_prefix)

            folderpath_input = folderpath_input.strip()
            if folderpath_input == '':
                folderpath_input = self.output_dir
            #endregion

            #region counter
            # Create the target folder and pick the next free counter.
            full_output_folder = self.create_folder_structure(folderpath_input, foldername_prefix)
            counter = self.get_latest_counter(full_output_folder, filename_prefix, counter_digits, counter_position)
            filename = self.format_counter_filename(filename_prefix, counter, counter_digits, counter_position)
            #endregion

            #region convert
            # Convert ComfyUI image tensors to PIL images.
            results = list()
            pil_images = []

            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                pil_images.append(img)

            if not pil_images:
                return {"ui": {"images": results, "animated": (False,)}, "result": ("",)}
            #endregion

            #region metadata
            # Build EXIF metadata from prompt / extra_pnginfo when enabled.
            metadata = None
            if save_metadata == "enabled" and not args.disable_metadata:
                metadata = pil_images[0].getexif()
                if prompt is not None:
                    metadata[0x0110] = f"prompt:{json.dumps(prompt)}"
                if extra_pnginfo is not None:
                    initial_exif = 0x010f
                    for x in extra_pnginfo:
                        metadata[initial_exif] = f"{x}:{json.dumps(extra_pnginfo[x])}"
                        initial_exif -= 1
            #endregion

            #region write
            # Save the animated WebP, passing exif only when metadata is present.
            file_path = os.path.join(full_output_folder, filename)
            save_kwargs = dict(
                save_all=True,
                duration=int(round(1000.0/fps)),
                append_images=pil_images[1:],
                lossless=lossless,
                quality=quality,
                method=method,
            )
            if metadata is not None:
                save_kwargs["exif"] = metadata
            pil_images[0].save(file_path, **save_kwargs)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[SaveAnimatedWEBPExtendedFolderPath] [{timestamp}] Saved animated WebP to: {file_path}")
            #endregion

            #region result
            # Build the UI result and the filepath return value.
            results.append({
                "filename": filename,
                "subfolder": os.path.basename(full_output_folder),
                "type": self.type,
                "folder": full_output_folder
            })

            return {"ui": {"images": results, "animated": (True,)}, "result": (file_path,)}
            #endregion

        except Exception as e:
            print(f"[SaveAnimatedWEBPExtendedFolderPath] Error saving animated WebP: {str(e)}")
            raise
    #endregion

    #region ui
    # ComfyUI-facing declarations: input schema, return types, node metadata.
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "folderpath_input": ("STRING", {"default": "c:/"}),
                "foldername_prefix": ("STRING", {"default": "gen"}),
                "filename_prefix": ("STRING", {"default": "output"}),
                "fps": ("FLOAT", {"default": 20.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "lossless": ("BOOLEAN", {"default": True}),
                "quality": ("INT", {"default": 100, "min": 0, "max": 100}),
                "method": (list(s.methods.keys()),),
                "save_metadata": (["disabled", "enabled"], {"default": "enabled"}),
                "counter_digits": ([2, 3, 4, 5, 6], {"default": 3}),
                "counter_position": (["first", "last"], {"default": "last"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "illumorae"
    DESCRIPTION = "Saves an animated WebP to an external folder path"
    #endregion

#region register
NODE_CLASS_MAPPINGS = {
    "illumoraeSaveAnimatedWEBPFolderPathNode": illumoraeSaveAnimatedWEBPFolderPathNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeSaveAnimatedWEBPFolderPathNode": "Save Animated WEBP FolderPath",
}
#endregion
