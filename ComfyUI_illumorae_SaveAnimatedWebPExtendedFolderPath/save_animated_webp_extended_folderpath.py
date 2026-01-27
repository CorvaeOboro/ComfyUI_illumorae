"""
Save Animated WEBP FolderPath - a custom node for ComfyUI
webp is ideal video format , loops like gif with lossless quality , contains the comfy workflow in metadata
this version is a minor modification of the core node with added inputs for saving to an external project structure

TITLE::Save Animated WEBP FolderPath
DESCRIPTIONSHORT::Saves an animated WebP to an external folder path
VERSION::20260127
IMAGE::comfyui_illumorae_save_animated_webp_folderpath.png
GROUP::Save
"""
import os
import json
from datetime import datetime
from PIL import Image
import numpy as np
import folder_paths
from comfy.cli_args import args

class illumoraeSaveAnimatedWEBPFolderPathNode:
    TITLE = "Save Animated WEBP FolderPath"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    methods = {"default": 4, "fastest": 0, "slowest": 6}
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "folderpath_input": ("STRING", {"default": "c:/"}),
                "foldername_prefix": ("STRING", {"default": "gen"}),
                "filename_prefix": ("STRING", {"default": ""}),
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

    def get_unique_filename(self, folder_path, base_filename, counter, counter_digits, counter_position="last"):
        """Generate a unique filename with counter position handling"""
        webp_ext = ".webp"
        counter_str = str(counter).zfill(counter_digits)
        
        if counter_position == "first":
            filename = f"{counter_str}_{base_filename}{webp_ext}"
        else:  # last
            filename = f"{base_filename}_{counter_str}{webp_ext}"
            
        return filename

    def save_images(self, images, filename_prefix, folderpath_input, foldername_prefix, fps, 
                   lossless, quality, method, save_metadata="enabled", counter_digits=3,
                   counter_position="last", prompt=None, extra_pnginfo=None):
        try:
            method = self.methods.get(method)
            
            # Create folder structure
            full_output_folder = self.create_folder_structure(folderpath_input, foldername_prefix)
            
            # Initialize counter
            counter = 1
            while True:
                filename = self.get_unique_filename(full_output_folder, filename_prefix, 
                                                 counter, counter_digits, counter_position)
                if not os.path.exists(os.path.join(full_output_folder, filename)):
                    break
                counter += 1

            results = list()
            pil_images = []
            
            # Convert tensor images to PIL
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                pil_images.append(img)

            # Handle metadata
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

            # Save the animated WebP
            file_path = os.path.join(full_output_folder, filename)
            pil_images[0].save(
                file_path,
                save_all=True,
                duration=int(1000.0/fps),
                append_images=pil_images[1:],
                exif=metadata,
                lossless=lossless,
                quality=quality,
                method=method
            )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[SaveAnimatedWEBPExtendedFolderPath] [{timestamp}] Saved animated WebP to: {file_path}")

            results.append({
                "filename": filename,
                "subfolder": os.path.basename(full_output_folder),
                "type": self.type,
                "folder": full_output_folder
            })

            return {"ui": {"images": results, "animated": (True,)}, "result": (file_path,)}
        except Exception as e:
            print(f"[SaveAnimatedWEBPExtendedFolderPath] Error saving animated WebP: {str(e)}")
            raise

NODE_CLASS_MAPPINGS = {
    "illumoraeSaveAnimatedWEBPFolderPathNode": illumoraeSaveAnimatedWEBPFolderPathNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeSaveAnimatedWEBPFolderPathNode": "Save Animated WEBP FolderPath",
}
