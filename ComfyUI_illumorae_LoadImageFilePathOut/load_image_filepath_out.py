"""
illumoraeLoadImageFilePathOut - a ComfyUI Custom Node
------------------------------
Load image from a specified file path string and outputs the filepath 

Inputs:
    image: (str) Path to the image file to load

Outputs:
    image: Loaded image as a torch tensor
    mask: Alpha mask or default mask
    file name: Name of the loaded file (no extension)
    folder path: Directory containing the image file

useful for workflows where you need to pass along the image's file path or name for downstream processing or logging.

TITLE::Load Image w FilePath Out
DESCRIPTIONSHORT::Loads an image from a file path string and outputs image, mask, file name, and folder path.
VERSION::20260113
GROUP::Load
"""
import os
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths

class illumoraeLoadImageFilePathOut:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("STRING", {"default": r"C:/a/image.png [output]"})},
                }

    CATEGORY = "illumorae"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image","MASK","file name","folder path")
    FUNCTION = "load_image"
    DESCRIPTION = "Loads an image from a specified file path string and outputs the image, mask, file name, and folder path."
    
    def load_image(self, image):
        print(f"[LoadImageFilePathOut] Input image string: '{image}'")
        image_path = illumoraeLoadImageFilePathOut._resolve_path(image)
        print(f"[LoadImageFilePathOut] Resolved path: '{image_path}'")
        print(f"[LoadImageFilePathOut] Path exists: {image_path.exists()}")

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            
        dirname, basename = os.path.split(image_path)
        file_name = self.get_file_name_without_extension(image_path)
        folder_path = dirname
        return (image, mask, file_name, folder_path)

    @staticmethod
    def _resolve_path(image) -> Path:
        print(f"[LoadImageFilePathOut._resolve_path] Input: '{image}' (type: {type(image)})")
        
        # If input is already a valid path, use it directly
        if isinstance(image, (str, Path)):
            direct_path = Path(image)
            if direct_path.exists() and direct_path.is_file():
                print(f"[LoadImageFilePathOut._resolve_path] Input is valid file path, using directly: '{direct_path}'")
                return direct_path
        
        # Otherwise use ComfyUI's annotation system
        try:
            annotated = folder_paths.get_annotated_filepath(image)
            print(f"[LoadImageFilePathOut._resolve_path] After get_annotated_filepath: '{annotated}'")
            image_path = Path(annotated)
            print(f"[LoadImageFilePathOut._resolve_path] Final Path object: '{image_path}'")
            
            # Verify the path exists
            if not image_path.exists():
                print(f"[LoadImageFilePathOut._resolve_path] WARNING: Resolved path does not exist!")
                print(f"[LoadImageFilePathOut._resolve_path] Trying to use input directly as fallback...")
                fallback_path = Path(image)
                if fallback_path.exists():
                    print(f"[LoadImageFilePathOut._resolve_path] Fallback successful: '{fallback_path}'")
                    return fallback_path
            
            return image_path
        except Exception as e:
            print(f"[LoadImageFilePathOut._resolve_path] Error with get_annotated_filepath: {e}")
            print(f"[LoadImageFilePathOut._resolve_path] Using input directly as Path")
            return Path(image)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = illumoraeLoadImageFilePathOut._resolve_path(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        # If image is an output of another node, it will be None during validation
        if image is None:
            return True

        image_path = illumoraeLoadImageFilePathOut._resolve_path(image)
        if not image_path.exists():
            return "Invalid image path: {}".format(image_path)

        return True

    @staticmethod
    def get_file_name_without_extension(file_path):
        file_name_with_extension = os.path.basename(file_path)
        file_name, _ = os.path.splitext(file_name_with_extension)
        return file_name

NODE_CLASS_MAPPINGS = {
    'illumoraeLoadImageFilePathOut': illumoraeLoadImageFilePathOut,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeLoadImageFilePathOut': 'Load Image w FilePath Out',
}
