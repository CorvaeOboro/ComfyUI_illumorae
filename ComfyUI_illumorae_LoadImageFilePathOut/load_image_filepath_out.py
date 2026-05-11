"""
illumoraeLoadImageFilePathOut - a ComfyUI Custom Node
------------------------------
Load image from a specified file path string and outputs the filepath 

Inputs:
    image: (str) Path to the image file to load
    debug_mode: (bool) When True, prints diagnostic log messages

Outputs:
    image: Loaded image as a torch tensor
    mask: Alpha mask or default mask
    file name: Name of the loaded file (no extension)
    folder path: Directory containing the image file
    width: (int) Image width in pixels
    height: (int) Image height in pixels

useful for workflows where you need to pass along the image's file path or name for downstream processing or logging.

TITLE::Load Image w FilePath Out
DESCRIPTIONSHORT::Loads an image from a file path string and outputs image, mask, file name, and folder path.
VERSION::20260127
IMAGE::comfyui_illumorae_load_image_filepath_out.png
GROUP::Load
"""
import os
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths

class illumoraeLoadImageWFilePathOutNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "image": ("STRING", {"default": r"C:/a/image.png [output]"}),
                        "debug_mode": ("BOOLEAN", {"default": False}),
                    },
                }

    CATEGORY = "illumorae"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "MASK", "file name", "folder path", "width", "height")
    FUNCTION = "load_image"
    DESCRIPTION = "Loads an image from a file path string and outputs image, mask, file name, folder path, width, and height."

    def load_image(self, image, debug_mode=False):
        def _log(msg):
            if debug_mode:
                print(msg)

        _log(f"[LoadImageFilePathOut] Input image string: '{image}'")
        image_path = illumoraeLoadImageWFilePathOutNode._resolve_path(image, debug_mode=debug_mode)
        _log(f"[LoadImageFilePathOut] Resolved path: '{image_path}'")
        _log(f"[LoadImageFilePathOut] Path exists: {image_path.exists()}")

        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        rgb = i.convert("RGB")
        width, height = rgb.size
        image = np.array(rgb).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

        dirname, basename = os.path.split(image_path)
        file_name = self.get_file_name_without_extension(image_path)
        folder_path = dirname
        return (image, mask, file_name, folder_path, width, height)

    @staticmethod
    def _resolve_path(image, debug_mode=False) -> Path:
        def _log(msg):
            if debug_mode:
                print(msg)

        _log(f"[LoadImageFilePathOut._resolve_path] Input: '{image}' (type: {type(image)})")

        # Handle None input (can happen during IS_CHANGED before validation)
        if image is None:
            return None

        # If input is already a valid path, use it directly
        if isinstance(image, (str, Path)):
            direct_path = Path(image)
            if direct_path.exists() and direct_path.is_file():
                _log(f"[LoadImageFilePathOut._resolve_path] Input is valid file path, using directly: '{direct_path}'")
                return direct_path

        # Otherwise use ComfyUI's annotation system
        try:
            annotated = folder_paths.get_annotated_filepath(image)
            _log(f"[LoadImageFilePathOut._resolve_path] After get_annotated_filepath: '{annotated}'")
            image_path = Path(annotated)
            _log(f"[LoadImageFilePathOut._resolve_path] Final Path object: '{image_path}'")

            # Verify the path exists
            if not image_path.exists():
                _log(f"[LoadImageFilePathOut._resolve_path] WARNING: Resolved path does not exist!")
                _log(f"[LoadImageFilePathOut._resolve_path] Trying to use input directly as fallback...")
                fallback_path = Path(image)
                if fallback_path.exists():
                    _log(f"[LoadImageFilePathOut._resolve_path] Fallback successful: '{fallback_path}'")
                    return fallback_path

            return image_path
        except Exception as e:
            _log(f"[LoadImageFilePathOut._resolve_path] Error with get_annotated_filepath: {e}")
            _log(f"[LoadImageFilePathOut._resolve_path] Using input directly as Path")
            return Path(image)

    @classmethod
    def IS_CHANGED(s, image, debug_mode=False):
        image_path = illumoraeLoadImageWFilePathOutNode._resolve_path(image, debug_mode=debug_mode)
        if image_path is None:
            return ""
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, debug_mode=False):
        # If image is an output of another node, it will be None during validation
        if image is None:
            return True

        image_path = illumoraeLoadImageWFilePathOutNode._resolve_path(image, debug_mode=debug_mode)
        if not image_path.exists():
            return "Invalid image path: {}".format(image_path)

        return True

    @staticmethod
    def get_file_name_without_extension(file_path):
        file_name_with_extension = os.path.basename(file_path)
        file_name, _ = os.path.splitext(file_name_with_extension)
        return file_name

NODE_CLASS_MAPPINGS = {
    'illumoraeLoadImageWFilePathOutNode': illumoraeLoadImageWFilePathOutNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeLoadImageWFilePathOutNode': 'Load Image w FilePath Out',
}
