"""
illumoraeControlnetImagePreprocessCached - a ComfyUI Custom Node
----------------------------------------------------------------
Wraps aux controlnet preprocessors (Canny, LineArt, LeReS depth) with an
on-disk cache so the preprocessor only re-runs when needed. The cache PNG
is written to "<source_dir>/<source_stem>/controlnet/<process>.png" where
<process> is canny, depth_leres, or lineart.

LineArt and LeReS are implemented self-contained in companion modules
(lineart_model.py, leres_model.py) and do not depend on
comfyui_controlnet_aux. Model weights are downloaded from HuggingFace
(lllyasviel/Annotators) on first use.

Inputs:
    image: (IMAGE, optional) The source image tensor to preprocess when the
        cache is stale or missing. When not provided, the node loads the
        image from source_path directly.
    source_path: (str) Full filesystem path of the source image file, OR the
        folder containing it (when used with source_name). Used to derive
        the cache path, load the image (when image is not wired), and
        (by default) the dateModified for freshness comparison.
    source_name: (str, optional) File name stem of the source image (no
        extension). When provided, the cache is written to
        "<source_path>/<source_name>/controlnet/<process>.png". When
        empty, source_path is treated as the full file path and the stem
        is derived from it.
    preprocessor: (str) Which preprocessor to dispatch to. Registered names
        live in PREPROCESSOR_DISPATCH.
    update_cache_if_newer: (bool) When True (default), the node compares the
        source image's dateModified against the cache file's mtime and
        regenerates only when the source is newer. When False, the node
        always uses the cache if a cache file is found, otherwise it
        generates and writes the cache. This is the auto-refresh toggle.
    force_refresh: (bool) When True, ignores any cache and re-runs the
        preprocessor, overwriting the cached PNG. This is the manual
        override switch.
    source_date_modified: (float, optional) When > 0, this timestamp
        overrides the source file's mtime for the freshness comparison.
        When 0 or unset, the node reads the source file's mtime.
    low_threshold: (float) Canny low threshold (used only for "canny").
    high_threshold: (float) Canny high threshold (used only for "canny").
    coarse: (float) LineArt coarse parameter (used only for "lineart").
    resolution: (int) LineArt/LeReS resolution.
    rm_nearest: (float) LeReS remove-nearest threshold (used only for
        "leres").
    rm_background: (float) LeReS remove-background threshold (used only
        for "leres").
    boost: (str) LeReS boost mode "disable"/"enable" (used only for
        "leres").
    debug: (bool) When True, prints detailed logging to the ComfyUI
        console showing cache path, hit/miss, preprocessor execution,
        and write results.

Outputs:
    image: The preprocessed map (depth/edge/lineart) as an IMAGE tensor.
    cache_hit: (bool) True when the result came from the on-disk cache,
        False when the preprocessor ran and the cache was (re)written.

TITLE::Controlnet Image Preprocess Cached
DESCRIPTIONSHORT::Caches aux controlnet preprocessor outputs (Canny, LineArt, LeReS depth) to disk and only recomputes when the source image is newer than the cache.
VERSION::20260905
GROUP::Image
GROUPORDER::20
LISTORDER::99
STATUS::working
"""
#region IMPORT - stdlib, third-party, and ComfyUI runtime imports
import hashlib
import os
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
from PIL import Image

# Ensure sibling model modules (lineart_model, leres_model) are importable
# regardless of how ComfyUI loads this file (package import vs. file path import).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
#endregion


#region DISPATCH - preprocessor registry
# Each entry maps a preprocessor name (used in the COMBO widget) to a handler
# callable with signature (image: torch.Tensor, **widgets) -> torch.Tensor.
# Handlers import their backing implementation lazily so a missing optional
# package only errors when that preprocessor is actually selected.

def _canny(image: torch.Tensor, low_threshold: float = 0.15,
           high_threshold: float = 0.3, **_kw) -> torch.Tensor:
    """Run Canny edge detection.

    Prefers comfy-core's Canny node (comfy_extras.nodes_canny) for parity
    with the stock workflow node; falls back to a torch Sobel implementation
    when that module is unavailable (e.g. outside a ComfyUI process).
    """
    try:
        from comfy_extras.nodes_canny import Canny  # lazy import
        return Canny().detect_edge(image, low_threshold, high_threshold)[0]
    except Exception:
        return _canny_torch(image, low_threshold, high_threshold)


def _canny_torch(image: torch.Tensor, low_threshold: float,
                 high_threshold: float) -> torch.Tensor:
    """Torch Sobel-based Canny fallback. Input/output are (B, H, W, C) float32 in [0, 1]."""
    img = image.detach().clone()
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] == 1:
        img = img.repeat(1, 1, 1, 3)
    gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    gray = gray.unsqueeze(1)  # (B, 1, H, W)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(torch.nn.functional.pad(gray, (1, 1, 1, 1), mode='replicate'), sobel_x)
    gy = torch.nn.functional.conv2d(torch.nn.functional.pad(gray, (1, 1, 1, 1), mode='replicate'), sobel_y)
    mag = torch.sqrt(gx * gx + gy * gy)
    low = float(low_threshold)
    high = float(high_threshold)
    strong = mag >= high
    weak = (mag >= low) & (mag < high)
    out = strong.float()
    # simple weak-to-strong connectivity pass (8-neighbour)
    for _ in range(3):
        dilated = torch.nn.functional.max_pool2d(out, 3, stride=1, padding=1)
        out = torch.where(weak & (dilated > 0), torch.ones_like(out), out)
    out = out.clamp(0, 1)
    return out.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)  # (B, H, W, 3)


def _lineart(image: torch.Tensor, coarse: float = 1.0, resolution: int = 512,
             **_kw) -> torch.Tensor:
    """Run LineArt preprocessor using the self-contained lineart_model module.

    Downloads weights from HuggingFace on first use (lllyasviel/Annotators).
    The ``coarse`` widget selects the coarse model when > 0.5.
    """
    from lineart_model import LineartDetector
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = LineartDetector.from_pretrained()
    detector.to(device)

    # Convert (B, H, W, C) tensor to (H, W, 3) uint8 numpy for the detector.
    img_np = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
    elif img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)

    use_coarse = float(coarse) > 0.5
    detected = detector(img_np, coarse=use_coarse, detect_resolution=resolution,
                        image_resolution=resolution, output_type="np")
    # detected is (H, W, 3) uint8; convert back to (1, H, W, 3) float32 [0,1]
    result = torch.from_numpy(detected.astype(np.float32) / 255.0).unsqueeze(0)
    del detector
    return result


def _leres(image: torch.Tensor, rm_nearest: float = 0.0,
           rm_background: float = 0.0, boost: str = "disable",
           resolution: int = 512, **_kw) -> torch.Tensor:
    """Run LeReS depth map preprocessor using the self-contained leres_model module.

    Downloads weights from HuggingFace on first use (lllyasviel/Annotators).
    """
    from leres_model import LeresDetector
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = LeresDetector.from_pretrained(device=device)

    # Convert (B, H, W, C) tensor to (H, W, 3) uint8 numpy for the detector.
    img_np = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
    elif img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)

    detected = detector(img_np, thr_a=float(rm_nearest), thr_b=float(rm_background),
                        boost=(boost == "enable"), detect_resolution=resolution,
                        image_resolution=resolution)
    # detected is (H, W, 3) uint8; convert back to (1, H, W, 3) float32 [0,1]
    result = torch.from_numpy(detected.astype(np.float32) / 255.0).unsqueeze(0)
    del detector
    return result


PREPROCESSOR_DISPATCH: Dict[str, Callable[..., torch.Tensor]] = {
    "canny": _canny,
    "lineart": _lineart,
    "leres": _leres,
}
#endregion


#region HELPERS - cache path, mtime comparison, tensor/PNG conversion

# Maps preprocessor dispatch keys to the cache filename stem.
# "leres" is stored as "depth_leres" to match the controlnet_aux convention.
_CACHE_NAME_MAP: Dict[str, str] = {
    "canny": "canny",
    "lineart": "lineart",
    "leres": "depth_leres",
}


def _cache_path_for(source_path: str, preprocessor: str,
                    source_name: str = "") -> Path:
    """Derive the cache PNG path.

    When source_name is provided, the cache is placed under
    "<source_path>/<source_name>/controlnet/". Otherwise source_path is
    treated as the full file path and the cache is placed under
    "<source_parent>/<source_stem>/controlnet/".
    """
    name = _CACHE_NAME_MAP.get(preprocessor, preprocessor)
    if source_name:
        return Path(source_path) / source_name / "controlnet" / "{}.png".format(name)
    src = Path(source_path)
    return src.parent / src.stem / "controlnet" / "{}.png".format(name)


def _is_cache_fresh(source_path: str, cache_path: Path,
                    source_mtime_override: float = 0.0,
                    check_newer: bool = True) -> bool:
    """True when cache_path exists and (if check_newer) is at least as new
    as the source image.

    When ``source_mtime_override`` > 0, that timestamp is used instead of
    reading the source file's mtime. When ``check_newer`` is False, the
    mtime comparison is skipped and the cache is considered fresh as long
    as the cache file exists (the source file is still checked for
    existence when no override is given).
    """
    if not source_path and source_mtime_override <= 0:
        return False
    cache_exists = cache_path.is_file()
    if not cache_exists:
        return False
    if not check_newer:
        # Cache exists and mtime comparison is disabled -> always fresh.
        return True
    # Determine the source mtime.
    if source_mtime_override > 0:
        src_mtime = source_mtime_override
    else:
        src = Path(source_path)
        if not src.is_file():
            return False
        try:
            src_mtime = src.stat().st_mtime
        except OSError:
            return False
    try:
        return cache_path.stat().st_mtime >= src_mtime
    except OSError:
        return False


def _tensor_to_png(tensor: torch.Tensor, path: Path) -> None:
    """Save a (B, H, W, C) float32 [0,1] tensor as a PNG (first batch item)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (tensor[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        mode = "RGBA"
    elif arr.shape[-1] == 3:
        mode = "RGB"
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
        mode = "RGB"
    else:
        mode = "RGB"
    Image.fromarray(arr, mode=mode).save(str(path))


def _png_to_tensor(path: Path) -> torch.Tensor:
    """Load a PNG as a (1, H, W, 3) float32 [0,1] tensor."""
    pil = Image.open(str(path)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _load_image_from_path(file_path: Path) -> torch.Tensor:
    """Load an image file as a (1, H, W, 3) float32 [0,1] tensor."""
    pil = Image.open(str(file_path)).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _resolve_source(source_path: str, source_name: str,
                    source_date_modified: float) -> tuple:
    """Resolve the full source file path, stem, and mtime from inputs.

    Returns (file_path, stem, mtime, cache_path_parent) where:
      - file_path: Path to the actual source image file (or None)
      - stem: the source stem for cache path derivation
      - mtime: the source modification timestamp
      - cache_path_parent: the parent dir for the controlnet subfolder

    When source_name is provided, source_path is treated as the folder
    containing the file. Otherwise source_path is treated as the full
    file path.
    """
    if source_name:
        folder = Path(source_path)
        file_path = folder / source_name
        # Try common image extensions.
        if not file_path.is_file():
            for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                candidate = file_path.with_suffix(ext)
                if candidate.is_file():
                    file_path = candidate
                    break
        stem = source_name
        cache_parent = folder / stem
    else:
        file_path = Path(source_path)
        # When the path has no extension or doesn't exist as-is, try
        # common image extensions so a bare stem like "item_01" resolves
        # to "item_01.png" in the same directory.
        if file_path.name and not file_path.is_file() and file_path.suffix == "":
            for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                candidate = file_path.with_suffix(ext)
                if candidate.is_file():
                    file_path = candidate
                    break
        stem = file_path.stem
        cache_parent = file_path.parent / stem

    if file_path and file_path.is_file():
        if source_date_modified and float(source_date_modified) > 0:
            mtime = float(source_date_modified)
        else:
            try:
                mtime = float(file_path.stat().st_mtime)
            except OSError:
                mtime = 0.0
    else:
        mtime = float(source_date_modified) if source_date_modified else 0.0

    return file_path, stem, mtime, cache_parent


def _noop_log(fmt, *args):
    """No-op logging sink used when debug mode is off."""
    pass
#endregion


#region NODE - illumoraeControlnetImagePreprocessCachedNode
class illumoraeControlnetImagePreprocessCachedNode:
    """Caches aux controlnet preprocessor outputs to disk and only recomputes
    when the source image's dateModified is newer than the cached PNG.
    """

    #region CORE - execute: cache check, dispatch, save
    def execute(self, image=None, source_path="", preprocessor="canny",
                update_cache_if_newer=True, force_refresh=False,
                source_date_modified=0.0, source_name="",
                low_threshold=0.15, high_threshold=0.3, coarse=1.0,
                resolution=512, rm_nearest=0.0, rm_background=0.0,
                boost="disable", debug=False, **_kw):
        _log = self._log if debug else _noop_log
        handler = PREPROCESSOR_DISPATCH.get(preprocessor)
        if handler is None:
            raise ValueError(
                "Unknown preprocessor '{}'. Registered: {}".format(
                    preprocessor, ", ".join(sorted(PREPROCESSOR_DISPATCH)))
            )

        # Resolve the source file path, stem, and mtime from the inputs.
        file_path, stem, src_mtime, cache_parent = _resolve_source(
            source_path, source_name, source_date_modified)
        has_source = bool(source_path) and (
            bool(source_name) or (file_path and file_path.is_file()))
        cache_path = None
        if has_source:
            name = _CACHE_NAME_MAP.get(preprocessor, preprocessor)
            cache_path = cache_parent / "controlnet" / "{}.png".format(name)

        _log("preprocessor='{}' source_path='{}' source_name='{}'", preprocessor, source_path, source_name)
        _log("file_path='{}' stem='{}' cache_path='{}'", file_path, stem, cache_path)
        _log("force_refresh={} update_cache_if_newer={} src_mtime={}",
             force_refresh, update_cache_if_newer, src_mtime)

        # Cache hit: load the cached PNG and skip the preprocessor entirely.
        if cache_path is not None and not force_refresh:
            fresh = _is_cache_fresh(source_path, cache_path,
                                    source_mtime_override=src_mtime,
                                    check_newer=bool(update_cache_if_newer))
            _log("cache_fresh={}", fresh)
            if fresh:
                try:
                    result = _png_to_tensor(cache_path)
                    _log("CACHE HIT - loaded '{}'", cache_path)
                    return (result, True)
                except Exception as e:
                    _log("cache read failed, falling through to recompute: {}", e)

        # Cache miss or forced: need the source image tensor.
        # When image is not wired, load it from the resolved file path.
        if image is None:
            if file_path and file_path.is_file():
                _log("loading source image from '{}'", file_path)
                image = _load_image_from_path(file_path)
            else:
                raise ValueError(
                    "No image tensor provided and source_path does not "
                    "resolve to a file: '{}'. Either wire an IMAGE input "
                    "or provide a valid source_path.".format(source_path)
                )

        _log("CACHE MISS - running preprocessor '{}'", preprocessor)
        result = handler(image, low_threshold=low_threshold,
                         high_threshold=high_threshold, coarse=coarse,
                         resolution=resolution, rm_nearest=rm_nearest,
                         rm_background=rm_background, boost=boost)
        result = result if isinstance(result, torch.Tensor) else result[0]
        _log("preprocessor done, result shape={}", tuple(result.shape))

        if cache_path is not None:
            try:
                _tensor_to_png(result, cache_path)
                _log("cache written to '{}'", cache_path)
            except Exception as e:
                _log("cache write FAILED: {}", e)
        else:
            _log("cache skipped - no valid source_path provided")
        return (result, False)
    #endregion

    @staticmethod
    def _log(fmt, *args):
        """Debug logger: prints formatted message with node prefix."""
        try:
            msg = fmt.format(*args)
        except Exception:
            msg = fmt
        print("[ControlnetImagePreprocessCached] {}".format(msg))
    #endregion

    #region CACHE - IS_CHANGED: fold source + cache mtimes into the key
    @classmethod
    def IS_CHANGED(cls, image=None, source_path="", preprocessor="canny",
                   update_cache_if_newer=True, force_refresh=False,
                   source_date_modified=0.0, source_name="",
                   low_threshold=0.15, high_threshold=0.3, coarse=1.0,
                   resolution=512, rm_nearest=0.0, rm_background=0.0,
                   boost="disable", debug=False, **_kw):
        file_path, stem, src_mtime, cache_parent = _resolve_source(
            source_path, source_name, source_date_modified)
        cache_mtime = ""
        if source_path and (source_name or (file_path and file_path.is_file())):
            name = _CACHE_NAME_MAP.get(preprocessor, preprocessor)
            cache_path = cache_parent / "controlnet" / "{}.png".format(name)
            if cache_path.is_file():
                try:
                    cache_mtime = int(cache_path.stat().st_mtime)
                except OSError:
                    cache_mtime = "<unreadable>"
        key = "|".join([
            source_path, source_name, str(src_mtime), str(cache_mtime),
            preprocessor, str(bool(update_cache_if_newer)),
            str(bool(force_refresh)), str(float(source_date_modified)),
            str(low_threshold), str(high_threshold),
            str(coarse), str(resolution),
            str(rm_nearest), str(rm_background), str(boost),
        ])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
    #endregion

    #region UI - ComfyUI interface declarations
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": ""}),
                "preprocessor": (sorted(PREPROCESSOR_DISPATCH.keys()),),
                "update_cache_if_newer": ("BOOLEAN", {"default": True}),
                "force_refresh": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "source_name": ("STRING", {"default": ""}),
                "source_date_modified": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e12, "step": 1.0}),
                "debug": ("BOOLEAN", {"default": False}),
                "low_threshold": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "high_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "coarse": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "rm_nearest": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "rm_background": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "boost": (["disable", "enable"],),
            },
        }

    CATEGORY = "illumorae"
    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "cache_hit")
    FUNCTION = "execute"
    DESCRIPTION = (
        "Caches aux controlnet preprocessor outputs (Canny, LineArt, LeReS "
        "depth) to disk beside the source image. When update_cache_if_newer "
        "is True (default), recomputes only when the source is newer than "
        "the cache. When False, always uses the cache if found. Use "
        "force_refresh to bypass the cache entirely. source_date_modified "
        "overrides the file mtime when provided."
    )
    #endregion
#endregion


#region REGISTRY - node registration mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    'illumoraeControlnetImagePreprocessCachedNode': illumoraeControlnetImagePreprocessCachedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeControlnetImagePreprocessCachedNode': 'Controlnet Image Preprocess Cached (illumorae)',
}
#endregion
