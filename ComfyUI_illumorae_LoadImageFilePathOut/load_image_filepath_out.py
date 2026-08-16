"""
illumoraeLoadImageFilePathOut - a ComfyUI Custom Node
------------------------------
Load image from a specified file path string and outputs the filepath 

Inputs:
    image: (str) Path to the image file to load
    debug_mode: (bool) When True, prints diagnostic log messages

Outputs:
    image: Loaded image as a torch tensor
    mask: Alpha mask (inverted so mask=1 means transparent). When the image has
          no alpha channel, a 64x64 zero tensor is returned for parity with
          ComfyUI core LoadImage; this does NOT match the image dimensions.
    file name: Name of the loaded file (no extension)
    folder path: Directory containing the image file
    width: (int) Image width in pixels
    height: (int) Image height in pixels

useful for workflows where you need to pass along the image's file path or name for downstream processing or logging.

TITLE::Load Image w FilePath Out
DESCRIPTIONSHORT::Loads an image from a file path string and outputs image, mask, file name, and folder path.
VERSION::20260426
IMAGE::comfyui_illumorae_load_image_filepath_out.png
GROUP::Load
GROUPORDER::3
LISTORDER::2
STATUS::working
"""
#region IMPORT - stdlib, third-party, and ComfyUI runtime imports
import hashlib
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
import folder_paths
#endregion


class illumoraeLoadImageWFilePathOutNode:
    #region CORE - load_image: main FUNCTION entry point, decodes image to tensor
    def load_image(self, image, debug_mode=False):
        def _log(msg):
            if debug_mode:
                print(msg)

        #region GUARD - reject None / unresolvable paths before decode
        # Reject None input explicitly. IS_CHANGED and VALIDATE_INPUTS tolerate
        # None (it can arrive during validation when the input is wired from
        # another node's output), but the actual run requires a real path.
        if image is None:
            raise ValueError("image path is required (received None)")

        _log(f"[LoadImageFilePathOut] Input image string: '{image}'")
        image_path = illumoraeLoadImageWFilePathOutNode._resolve_path(image, debug_mode=debug_mode)
        if image_path is None or not image_path.is_file():
            raise ValueError(f"image path did not resolve to a file: {image!r}")
        _log(f"[LoadImageFilePathOut] Resolved path: '{image_path}'")
        _log(f"[LoadImageFilePathOut] Path exists: {image_path.exists()}")
        #endregion

        #region DECODE - PIL open, EXIF orient, palette alpha, RGB tensor + mask
        try:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            # Palette-mode images store transparency as an index into the palette
            # (i.info['transparency']), not as an 'A' band. Convert through RGBA
            # first so the transparency becomes a real alpha channel, then read
            # the alpha from the converted image.
            if i.mode == "P" and "transparency" in i.info:
                i = i.convert("RGBA")
            rgb = i.convert("RGB")
            width, height = rgb.size
            image_tensor = np.array(rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_tensor)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        except (OSError, UnidentifiedImageError, ValueError) as e:
            _log(f"[LoadImageFilePathOut] Failed to decode image '{image_path}': {e}")
            raise RuntimeError(f"Failed to load image '{image_path}': {e}") from e
        #endregion

        #region META - extract file name stem and parent folder path
        file_name = image_path.stem
        folder_path = str(image_path.parent)
        #endregion

        return (image_tensor, mask, file_name, folder_path, width, height)
    #endregion

    #region RESOLVE - _resolve_path: direct path or ComfyUI annotation lookup
    @staticmethod
    def _resolve_path(image, debug_mode=False) -> Optional[Path]:
        def _log(msg):
            if debug_mode:
                print(msg)

        _log(f"[LoadImageFilePathOut._resolve_path] Input: '{image}' (type: {type(image)})")

        #region NULLCHK - None short-circuits to None (used by IS_CHANGED / VALIDATE)
        # Handle None input (can happen during IS_CHANGED before validation)
        if image is None:
            return None
        #endregion

        #region DIRECT - if input is already a valid file path, use it as-is
        # If input is already a valid path, use it directly
        if isinstance(image, (str, Path)):
            direct_path = Path(image)
            if direct_path.exists() and direct_path.is_file():
                _log(f"[LoadImageFilePathOut._resolve_path] Input is valid file path, using directly: '{direct_path}'")
                return direct_path
        #endregion

        #region ANNOTATE - fall back to ComfyUI annotation, then raw Path
        # Otherwise use ComfyUI's annotation system
        try:
            annotated = folder_paths.get_annotated_filepath(image)
            _log(f"[LoadImageFilePathOut._resolve_path] After get_annotated_filepath: '{annotated}'")
            image_path = Path(annotated)
            _log(f"[LoadImageFilePathOut._resolve_path] Final Path object: '{image_path}'")

            # Verify the path exists
            if not image_path.exists():
                _log("[LoadImageFilePathOut._resolve_path] WARNING: Resolved path does not exist!")
                _log("[LoadImageFilePathOut._resolve_path] Trying to use input directly as fallback...")
                fallback_path = Path(image)
                if fallback_path.exists():
                    _log(f"[LoadImageFilePathOut._resolve_path] Fallback successful: '{fallback_path}'")
                    return fallback_path

            return image_path
        except Exception as e:
            _log(f"[LoadImageFilePathOut._resolve_path] Error with get_annotated_filepath: {e}")
            _log("[LoadImageFilePathOut._resolve_path] Using input directly as Path")
            return Path(image)
        #endregion
    #endregion

    #region CACHE - IS_CHANGED: SHA-256 content hash for cache key
    @classmethod
    def IS_CHANGED(s, image, debug_mode=False):
        image_path = illumoraeLoadImageWFilePathOutNode._resolve_path(image, debug_mode=debug_mode)
        if image_path is None:
            return ""
        if not image_path.is_file():
            return float("nan")
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
    #endregion

    #region VALIDATE - VALIDATE_INPUTS: pre-run path existence check
    @classmethod
    def VALIDATE_INPUTS(s, image, debug_mode=False):
        # If image is an output of another node, it will be None during validation
        if image is None:
            return True

        image_path = illumoraeLoadImageWFilePathOutNode._resolve_path(image, debug_mode=debug_mode)
        if not image_path.exists():
            return "Invalid image path: {}".format(image_path)

        return True
    #endregion

    #region UI - ComfyUI interface declarations (inputs, outputs, metadata)
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "image": ("STRING", {"default": r"C:/path/to/your_image.png"}),
                        "debug_mode": ("BOOLEAN", {"default": False}),
                    },
                }

    CATEGORY = "illumorae"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "FILE NAME", "FOLDER PATH", "WIDTH", "HEIGHT")
    FUNCTION = "load_image"
    DESCRIPTION = "Loads an image from a file path string and outputs image, mask, file name, folder path, width, and height."
    #endregion


#region REGISTRY - node registration mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    'illumoraeLoadImageWFilePathOutNode': illumoraeLoadImageWFilePathOutNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeLoadImageWFilePathOutNode': 'Load Image w FilePath Out',
}
#endregion
