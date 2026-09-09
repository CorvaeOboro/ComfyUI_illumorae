"""Self-contained LineArt detector for the Cached Preprocess Image node.

Implements the informative-drawings LineArt model (carolineec/informative-drawings)
without depending on comfyui_controlnet_aux. Weights are downloaded from the
HuggingFace hub (lllyasviel/Annotators) on first use and cached locally.

Model files:
    sk_model.pth  - fine lineart generator
    sk_model2.pth - coarse lineart generator

Architecture: Generator(3, 1, n_residual_blocks=3) with InstanceNorm,
reflection padding, and sigmoid output.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None  # type: ignore[assignment]

try:
    import comfy.model_management as model_management
    _get_device = model_management.get_torch_device
except ImportError:
    def _get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


#region MODEL - Generator network from informative-drawings
norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class Generator(nn.Module):
    """Lineart generator: 7x7 conv -> 2 downsample -> 3 residual -> 2 upsample -> 7x7 conv + sigmoid."""

    def __init__(self, input_nc: int, output_nc: int, n_residual_blocks: int = 9, sigmoid: bool = True):
        super().__init__()

        # Initial convolution block
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        layers1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            layers1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*layers1)

        # Residual blocks
        layers2 = [ResidualBlock(in_features) for _ in range(n_residual_blocks)]
        self.model2 = nn.Sequential(*layers2)

        # Upsampling
        layers3 = []
        out_features = in_features // 2
        for _ in range(2):
            layers3 += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*layers3)

        # Output layer
        layers4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            layers4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*layers4)

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        return out
#endregion


#region UTIL - HWC3 and resize helpers (from controlnet_aux.util)
def _hwc3(x: np.ndarray) -> np.ndarray:
    """Ensure a uint8 array has 3 channels."""
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    h, w, c = x.shape
    assert c == 1 or c == 3 or c == 4
    if c == 3:
        return x
    if c == 1:
        return np.concatenate([x, x, x], axis=2)
    # c == 4: blend alpha
    color = x[:, :, 0:3].astype(np.float32)
    alpha = x[:, :, 3:4].astype(np.float32) / 255.0
    y = color * alpha + 255.0 * (1.0 - alpha)
    return y.clip(0, 255).astype(np.uint8)


def _resize_np(img: np.ndarray, w: int, h: int, mode: str = "bilinear") -> np.ndarray:
    """Resize a numpy array (2D or 3D HWC) using torch.nn.functional.interpolate."""
    orig_dtype = img.dtype
    if img.ndim == 2:
        t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(h, w), mode=mode,
                          align_corners=False if mode in ("bilinear", "bicubic") else None)
        return t.squeeze().numpy().astype(orig_dtype)
    if img.ndim == 3:
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(h, w), mode=mode,
                          align_corners=False if mode in ("bilinear", "bicubic") else None)
        return t.squeeze(0).permute(1, 2, 0).numpy().astype(orig_dtype)
    return img


def _resize_image(input_image: np.ndarray, resolution: int) -> np.ndarray:
    """Resize so the shorter side ~= resolution, rounded to 64px."""
    h, w, _ = input_image.shape
    h = float(h)
    w = float(w)
    k = float(resolution) / min(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64
    mode = "bicubic" if k > 1 else "area"
    return _resize_np(input_image, w, h, mode=mode)
#endregion


#region DETECTOR - LineartDetector: loads weights, runs inference
class LineartDetector:
    """LineArt detector with fine and coarse models.

    Use ``coarse=True`` for the coarse model (sk_model2.pth), ``False`` for
    the fine model (sk_model.pth).
    """

    def __init__(self, model: Generator, coarse_model: Generator):
        self.model = model
        self.model_coarse = coarse_model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "lllyasviel/Annotators",
        filename: str = "sk_model.pth",
        coarse_filename: str = "sk_model2.pth",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "LineartDetector":
        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub is required to download LineArt weights. "
                "Install it with: pip install huggingface_hub"
            )
        model_path = hf_hub_download(
            repo_id, filename, cache_dir=cache_dir, local_files_only=local_files_only
        )
        coarse_path = hf_hub_download(
            repo_id, coarse_filename, cache_dir=cache_dir, local_files_only=local_files_only
        )

        model = Generator(3, 1, 3)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        coarse_model = Generator(3, 1, 3)
        coarse_model.load_state_dict(torch.load(coarse_path, map_location="cpu"))
        coarse_model.eval()

        return cls(model, coarse_model)

    def to(self, device: torch.device) -> "LineartDetector":
        self.model.to(device)
        self.model_coarse.to(device)
        return self

    def __call__(
        self,
        input_image: np.ndarray,
        coarse: bool = False,
        detect_resolution: int = 512,
        image_resolution: int = 512,
        output_type: str = "pil",
    ) -> np.ndarray | Image.Image:
        device = next(iter(self.model.parameters())).device
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)

        input_image = _hwc3(input_image)
        input_image = _resize_image(input_image, detect_resolution)

        model = self.model_coarse if coarse else self.model
        assert input_image.ndim == 3
        image = input_image
        with torch.no_grad():
            image = torch.from_numpy(image).float().to(device)
            image = image / 255.0
            image = image.permute(2, 0, 1).unsqueeze(0)  # HWC -> 1CHW
            line = model(image)[0][0]
            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = _hwc3(line)
        img = _resize_image(input_image, image_resolution)
        h, w, _ = img.shape
        detected_map = _resize_np(detected_map, w, h, mode="bilinear")
        detected_map = 255 - detected_map

        if output_type == "pil":
            return Image.fromarray(detected_map)
        return detected_map
#endregion
