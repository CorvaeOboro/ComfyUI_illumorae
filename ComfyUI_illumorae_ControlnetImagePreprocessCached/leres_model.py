"""Self-contained LeReS depth estimator for the Cached Preprocess Image node.

Implements the LeReS relative depth model and the pix2pix boost merge network
without depending on comfyui_controlnet_aux. Weights are downloaded from the
HuggingFace hub (lllyasviel/Annotators) on first use and cached locally.

Model files:
    res101.pth         - LeReS ResNeXt101 depth model
    latest_net_G.pth   - pix2pix merge network for boost mode

Architecture:
    - RelDepthModel: ResNeXt101_32x8d encoder + Decoder(FTB/FFM/AO)
    - Pix2Pix4Depth: UNet generator (input_nc=2, output_nc=1, 1024px)

The boost mode (leres++) uses double estimation + adaptive patch refinement
as described in the LeReS paper, merged through the pix2pix network.
"""
from __future__ import annotations

import gc
from collections import OrderedDict
from operator import getitem
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.transforms import Compose, Normalize, ToTensor

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


#region UTIL - HWC3, resize, and image ops (cv2-free, using torch/numpy)
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
    color = x[:, :, 0:3].astype(np.float32)
    alpha = x[:, :, 3:4].astype(np.float32) / 255.0
    y = color * alpha + 255.0 * (1.0 - alpha)
    return y.clip(0, 255).astype(np.uint8)


def _resize_np(img: np.ndarray, w: int, h: int, mode: str = "bilinear") -> np.ndarray:
    """Resize a numpy array (2D or 3D HWC) using torch.nn.functional.interpolate.

    mode: 'bilinear', 'bicubic', 'area', or 'nearest'.
    """
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


def _sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Sobel X and Y gradients using torch convolution."""
    t = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    sy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), sx)
    gy = F.conv2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), sy)
    return gx.squeeze().numpy(), gy.squeeze().numpy()


def _dilate(img: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    """Max dilation using torch max_pool2d."""
    t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pad = ksize // 2
    for _ in range(iterations):
        t = F.max_pool2d(t, ksize, stride=1, padding=pad)
    return t.squeeze().numpy()


def _gaussian_blur(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    """Gaussian blur using torch convolution."""
    half = ksize // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.conv2d(F.pad(t, (half, half, half, half), mode="reflect"), kernel_2d)
    return t.squeeze().numpy()


def _integral(img: np.ndarray) -> np.ndarray:
    """Compute integral image with zero-padded top and left row/col."""
    cum = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)
    return np.pad(cum, ((1, 0), (1, 0)), mode="constant")


def _threshold_tozero(img: np.ndarray, thresh: float) -> np.ndarray:
    """cv2.THRESH_TOZERO equivalent: keep values > thresh, set rest to 0."""
    result = img.copy()
    result[result <= thresh] = 0
    return result
#endregion


def _torch_gc() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped = OrderedDict()
    for key, value in state_dict.items():
        stripped[key.replace(prefix, "")] = value
    return stripped
#endregion


#region RESNEXT - ResNeXt101_32x8d backbone for LeReS
def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNet(nn.Module):
    """ResNeXt backbone that returns 4 feature maps at 1/4, 1/8, 1/16, 1/32."""

    def __init__(self, block, layers, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features


def _resnext101_32x8d() -> "_ResNet":
    return _ResNet(_Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8)
#endregion


#region DECODER - FTB, FFM, AO, Decoder modules
class _FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super().__init__()
        self.in1 = inchannels
        self.mid = midchannels
        self.conv1 = nn.Conv2d(self.in1, self.mid, 3, padding=1, stride=1, bias=True)
        self.conv_branch = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid, self.mid, 3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid, self.mid, 3, padding=1, stride=1, bias=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self._init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        return self.relu(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class _FFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, upfactor=2):
        super().__init__()
        self.ftb1 = _FTB(inchannels, midchannels)
        self.ftb2 = _FTB(midchannels, outchannels)
        self.upsample = nn.Upsample(scale_factor=upfactor, mode="bilinear", align_corners=True)
        self._init_params()

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        x = x + high_x
        x = self.ftb2(x)
        return self.upsample(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class _AO(nn.Module):
    """Adaptive output module."""
    def __init__(self, inchannels, outchannels, upfactor=2):
        super().__init__()
        self.adapt_conv = nn.Sequential(
            nn.Conv2d(inchannels, inchannels // 2, 3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(inchannels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannels // 2, outchannels, 3, padding=1, stride=1, bias=True),
            nn.Upsample(scale_factor=upfactor, mode="bilinear", align_corners=True),
        )
        self._init_params()

    def forward(self, x):
        return self.adapt_conv(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        inchannels = [256, 512, 1024, 2048]
        midchannels = [256, 256, 256, 512]
        upfactors = [2, 2, 2, 2]
        self.conv = _FTB(inchannels[3], midchannels[3])
        self.conv1 = nn.Conv2d(midchannels[3], midchannels[2], 3, padding=1, stride=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=upfactors[3], mode="bilinear", align_corners=True)
        self.ffm2 = _FFM(inchannels[2], midchannels[2], midchannels[2], upfactors[2])
        self.ffm1 = _FFM(inchannels[1], midchannels[1], midchannels[1], upfactors[1])
        self.ffm0 = _FFM(inchannels[0], midchannels[0], midchannels[0], upfactors[0])
        self.outconv = _AO(midchannels[0], 1, upfactor=2)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, features):
        x_32x = self.conv(features[3])
        x_32 = self.conv1(x_32x)
        x_16 = self.upsample(x_32)
        x_8 = self.ffm2(features[2], x_16)
        x_4 = self.ffm1(features[1], x_8)
        x_2 = self.ffm0(features[0], x_4)
        return self.outconv(x_2)


class _DepthNet(nn.Module):
    """Encoder + Decoder: takes RGB, returns depth map."""

    def __init__(self, backbone="resnext101_32x8d"):
        super().__init__()
        if backbone == "resnext101_32x8d":
            self.encoder = _resnext101_32x8d()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.decoder = _Decoder()

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


class RelDepthModel(nn.Module):
    """LeReS relative depth model: ResNeXt101 encoder + Decoder."""

    def __init__(self, backbone="resnext101"):
        super().__init__()
        if backbone == "resnext101":
            encoder = "resnext101_32x8d"
        else:
            encoder = backbone
        self.depth_model = _DepthNet(encoder)

    def inference(self, rgb):
        with torch.no_grad():
            depth = self.depth_model(rgb.to(next(self.depth_model.parameters()).device))
            return depth

    def forward(self, rgb):
        return self.depth_model(rgb)
#endregion


#region PIX2PIX - UNet generator for boost mode merge
class _UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, 4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, 4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """UNet generator for pix2pix depth merge (input_nc=2, output_nc=1, 1024px)."""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        unet_block = _UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                              norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = _UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                  norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = _UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = _UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = _UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = _UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                              outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)


class Pix2Pix4Depth:
    """Lightweight pix2pix merge network wrapper (inference only)."""

    def __init__(self, device: torch.device):
        self.device = device
        # Original pix2pix-depth uses norm='none' (Identity), not BatchNorm/InstanceNorm.
        self.netG = UnetGenerator(2, 1, 10, 64, norm_layer=nn.Identity, use_dropout=False)
        self.netG.to(self.device)
        self.netG.eval()

    def load_network(self, weight_path: str):
        state_dict = torch.load(weight_path, map_location=str(self.device))
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        net = self.netG
        if isinstance(net, nn.DataParallel):
            net = net.module
        net.load_state_dict(state_dict, strict=True)

    def set_input(self, outer: np.ndarray, inner: np.ndarray):
        inner_t = torch.from_numpy(inner).unsqueeze(0).unsqueeze(0)
        outer_t = torch.from_numpy(outer).unsqueeze(0).unsqueeze(0)
        inner_t = (inner_t - torch.min(inner_t)) / (torch.max(inner_t) - torch.min(inner_t))
        outer_t = (outer_t - torch.min(outer_t)) / (torch.max(outer_t) - torch.min(outer_t))
        inner_t = inner_t * 2 - 1
        outer_t = outer_t * 2 - 1
        self.real_A = torch.cat((outer_t, inner_t), 1).to(self.device)

    def test(self):
        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)

    def get_current_visuals(self):
        return {"fake_B": self.fake_B}
#endregion


#region DEPTHMAP - estimation functions (estimateleres, estimateboost)
_whole_size_threshold = 1600
_pix2pixsize = 1024
_factor = 1.0


def _scale_torch(img: np.ndarray) -> torch.Tensor:
    """Scale image to torch tensor with ImageNet normalization."""
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return transform(img.astype(np.float32))
    return torch.from_numpy(img.astype(np.float32))


def estimateleres(img: np.ndarray, model: RelDepthModel, w: int, h: int) -> np.ndarray:
    """Single-pass LeReS depth estimation."""
    device = next(iter(model.parameters())).device
    rgb_c = img[:, :, ::-1].copy()
    a_resize = _resize_np(rgb_c, w, h, mode="bilinear")
    img_torch = _scale_torch(a_resize)[None, :, :, :]
    with torch.no_grad():
        img_torch = img_torch.to(device)
        prediction = model.depth_model(img_torch)
    prediction = prediction.squeeze().cpu().numpy()
    return _resize_np(prediction, img.shape[1], img.shape[0], mode="bicubic")


def _generatemask(size) -> np.ndarray:
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0] / 16)
    k_size = int(2 * np.ceil(2 * int(size[0] / 16)) + 1)
    mask[int(0.15 * size[0]):size[0] - int(0.15 * size[0]),
         int(0.15 * size[1]):size[1] - int(0.15 * size[1])] = 1
    mask = _gaussian_blur(mask, k_size, sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask.astype(np.float32)


def _resizewithpool(img: np.ndarray, size: int) -> np.ndarray:
    """Downsample by taking the max of each n x n block (replaces skimage.measure.block_reduce)."""
    n = int(np.floor(img.shape[0] / size))
    if n <= 1:
        return img
    # Trim to a multiple of n, then reshape and take max per block.
    h, w = img.shape
    h_trim = (h // n) * n
    w_trim = (w // n) * n
    trimmed = img[:h_trim, :w_trim]
    reshaped = trimmed.reshape(h_trim // n, n, w_trim // n, n)
    return reshaped.max(axis=(1, 3))


def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def _calculateprocessingres(img, basesize, confidence=0.1, scale_threshold=3, whole_size_threshold=3000):
    speed_scale = 32
    image_dim = int(min(img.shape[0:2]))
    gray = _rgb2gray(img)
    gx, gy = _sobel_gradients(gray)
    grad = np.abs(gx) + np.abs(gy)
    grad = _resize_np(grad, image_dim, image_dim, mode="area")
    m, M = grad.min(), grad.max()
    middle = m + 0.4 * (M - m)
    grad[grad < middle] = 0
    grad[grad >= middle] = 1
    ksize1 = int(basesize / speed_scale)
    ksize2 = int(basesize / (4 * speed_scale))
    threshold = min(whole_size_threshold, scale_threshold * max(img.shape[:2]))
    outputsize_scale = basesize / speed_scale
    for p_size in range(int(basesize / speed_scale), int(threshold / speed_scale),
                        int(basesize / (2 * speed_scale))):
        grad_resized = _resizewithpool(grad, p_size)
        grad_resized = _resize_np(grad_resized, p_size, p_size, mode="nearest")
        grad_resized[grad_resized >= 0.5] = 1
        grad_resized[grad_resized < 0.5] = 0
        dilated = _dilate(grad_resized, ksize1, iterations=1)
        if (1 - dilated).mean() > confidence:
            break
        outputsize_scale = p_size
    grad_region = _dilate(grad_resized, ksize2, iterations=1)
    patch_scale = grad_region.mean()
    return int(outputsize_scale * speed_scale), patch_scale


def _singleestimate(img, msize, model, net_type):
    return estimateleres(img, model, msize, msize)


def _doubleestimate(img, size1, size2, pix2pixsize, model, net_type, pix2pixmodel):
    estimate1 = _singleestimate(img, size1, model, net_type)
    estimate1 = _resize_np(estimate1, pix2pixsize, pix2pixsize, mode="bicubic")
    estimate2 = _singleestimate(img, size2, model, net_type)
    estimate2 = _resize_np(estimate2, pix2pixsize, pix2pixsize, mode="bicubic")
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals["fake_B"]
    prediction_mapped = (prediction_mapped + 1) / 2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / \
        (torch.max(prediction_mapped) - torch.min(prediction_mapped))
    return prediction_mapped.squeeze().cpu().numpy()


def _applygridpatch(blsize, stride, img, box):
    counter1 = 0
    patch_bound_list = {}
    for k in range(blsize, img.shape[1] - blsize, stride):
        for j in range(blsize, img.shape[0] - blsize, stride):
            patch_bound_list[str(counter1)] = {}
            patchbounds = [j - blsize, k - blsize, j - blsize + 2 * blsize, k - blsize + 2 * blsize]
            patch_bound = [box[0] + patchbounds[1], box[1] + patchbounds[0],
                           patchbounds[3] - patchbounds[1], patchbounds[2] - patchbounds[0]]
            patch_bound_list[str(counter1)]["rect"] = patch_bound
            patch_bound_list[str(counter1)]["size"] = patch_bound[2]
            counter1 += 1
    return patch_bound_list


def _getgf_fromintegral(integralimage, rect):
    x1, y1 = rect[1], rect[0]
    x2, y2 = rect[1] + rect[3], rect[0] + rect[2]
    return integralimage[x2, y2] - integralimage[x1, y2] - integralimage[x2, y1] + integralimage[x1, y1]


def _adaptiveselection(integral_grad, patch_bound_list, gf):
    global _factor
    patchlist = {}
    count = 0
    height, width = integral_grad.shape
    search_step = int(32 / _factor)
    for c in range(len(patch_bound_list)):
        bbox = patch_bound_list[str(c)]["rect"]
        cgf = _getgf_fromintegral(integral_grad, bbox) / (bbox[2] * bbox[3])
        if cgf >= gf:
            bbox_test = bbox.copy()
            patchlist[str(count)] = {}
            while True:
                bbox_test[0] -= int(search_step / 2)
                bbox_test[1] -= int(search_step / 2)
                bbox_test[2] += search_step
                bbox_test[3] += search_step
                if bbox_test[0] < 0 or bbox_test[1] < 0 or \
                   bbox_test[1] + bbox_test[3] >= height or bbox_test[0] + bbox_test[2] >= width:
                    break
                cgf = _getgf_fromintegral(integral_grad, bbox_test) / (bbox[2] * bbox[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()
            patchlist[str(count)]["rect"] = bbox
            patchlist[str(count)]["size"] = bbox[2]
            count += 1
    return patchlist


def _impatch(image, rect):
    return image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]


class _ImageandPatchs:
    def __init__(self, root_dir, name, patchsinfo, rgb_image, scale=1):
        self.root_dir = root_dir
        self.patchsinfo = patchsinfo
        self.name = name
        self.patchs = patchsinfo
        self.scale = scale
        self.rgb_image = _resize_np(rgb_image, round(rgb_image.shape[1] * scale),
                                    round(rgb_image.shape[0] * scale), mode="bicubic")
        self.do_have_estimate = False
        self.estimation_updated_image = None
        self.estimation_base_image = None

    def __len__(self):
        return len(self.patchs)

    def set_base_estimate(self, est):
        self.estimation_base_image = est
        if self.estimation_updated_image is not None:
            self.do_have_estimate = True

    def set_updated_estimate(self, est):
        self.estimation_updated_image = est
        if self.estimation_base_image is not None:
            self.do_have_estimate = True

    def __getitem__(self, index):
        patch_id = int(self.patchs[index][0])
        rect = np.array(self.patchs[index][1]["rect"])
        msize = self.patchs[index][1]["size"]
        rect = np.round(rect * self.scale).astype("int")
        msize = round(msize * self.scale)
        patch_rgb = _impatch(self.rgb_image, rect)
        if self.do_have_estimate:
            patch_whole_estimate_base = _impatch(self.estimation_base_image, rect)
            patch_whole_estimate_updated = _impatch(self.estimation_updated_image, rect)
            return {"patch_rgb": patch_rgb,
                    "patch_whole_estimate_base": patch_whole_estimate_base,
                    "patch_whole_estimate_updated": patch_whole_estimate_updated,
                    "rect": rect, "size": msize, "id": patch_id}
        return {"patch_rgb": patch_rgb, "rect": rect, "size": msize, "id": patch_id}


def _generatepatchs(img, base_size):
    global _factor
    img_gray = _rgb2gray(img)
    gx, gy = _sobel_gradients(img_gray)
    whole_grad = np.abs(gx) + np.abs(gy)
    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0
    gf = whole_grad.sum() / len(whole_grad.reshape(-1))
    grad_integral_image = _integral(whole_grad)
    blsize = int(round(base_size / 2))
    stride = int(round(blsize * 0.75))
    patch_bound_list = _applygridpatch(blsize, stride, img, [0, 0, 0, 0])
    patch_bound_list = _adaptiveselection(grad_integral_image, patch_bound_list, gf)
    return sorted(patch_bound_list.items(), key=lambda x: getitem(x[1], "size"), reverse=True)


def estimateboost(img, model, pix2pixmodel, max_res=512):
    """Boost mode: double estimation + adaptive patch refinement."""
    global _whole_size_threshold, _factor
    net_receptive_field_size = 448
    patch_netsize = 2 * net_receptive_field_size
    gc.collect()
    _torch_gc()
    mask_org = _generatemask((3000, 3000))
    mask = mask_org.copy()
    r_threshold_value = 0.2
    input_resolution = img.shape
    scale_threshold = 3
    whole_image_optimal_size, patch_scale = _calculateprocessingres(
        img, net_receptive_field_size, r_threshold_value, scale_threshold, _whole_size_threshold
    )
    whole_estimate = _doubleestimate(
        img, net_receptive_field_size, whole_image_optimal_size,
        _pix2pixsize, model, 0, pix2pixmodel
    )
    _factor = max(min(1, 4 * patch_scale * whole_image_optimal_size / _whole_size_threshold), 0.2)
    if max_res < whole_image_optimal_size:
        return _resize_np(whole_estimate, input_resolution[1], input_resolution[0], mode="bicubic")
    if img.shape[0] > img.shape[1]:
        a = 2 * whole_image_optimal_size
        b = round(2 * whole_image_optimal_size * img.shape[1] / img.shape[0])
    else:
        a = round(2 * whole_image_optimal_size * img.shape[0] / img.shape[1])
        b = 2 * whole_image_optimal_size
    b = int(round(b / _factor))
    a = int(round(a / _factor))
    img = _resize_np(img, b, a, mode="bicubic")
    base_size = net_receptive_field_size * 2
    patchset = _generatepatchs(img, base_size)
    mergein_scale = input_resolution[0] / img.shape[0]
    imageandpatchs = _ImageandPatchs("", "", patchset, img, mergein_scale)
    whole_estimate_resized = _resize_np(
        whole_estimate,
        round(img.shape[1] * mergein_scale), round(img.shape[0] * mergein_scale),
        mode="bicubic",
    )
    imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
    imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())
    for patch_ind in range(len(imageandpatchs)):
        patch = imageandpatchs[patch_ind]
        patch_rgb = patch["patch_rgb"]
        patch_whole_estimate_base = patch["patch_whole_estimate_base"]
        rect = patch["rect"]
        org_size = patch_whole_estimate_base.shape
        patch_estimation = _doubleestimate(
            patch_rgb, net_receptive_field_size, patch_netsize,
            _pix2pixsize, model, 0, pix2pixmodel
        )
        patch_estimation = _resize_np(patch_estimation, _pix2pixsize, _pix2pixsize, mode="bicubic")
        patch_whole_estimate_base = _resize_np(patch_whole_estimate_base, _pix2pixsize, _pix2pixsize,
                                               mode="bicubic")
        pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)
        pix2pixmodel.test()
        visuals = pix2pixmodel.get_current_visuals()
        prediction_mapped = visuals["fake_B"]
        prediction_mapped = (prediction_mapped + 1) / 2
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()
        p_coef = np.polyfit(prediction_mapped.reshape(-1), patch_whole_estimate_base.reshape(-1), deg=1)
        merged = np.polyval(p_coef, prediction_mapped.reshape(-1)).reshape(prediction_mapped.shape)
        merged = _resize_np(merged, org_size[1], org_size[0], mode="bicubic")
        w1, h1 = rect[0], rect[1]
        w2, h2 = w1 + rect[2], h1 + rect[3]
        if mask.shape != org_size:
            mask = _resize_np(mask_org, org_size[1], org_size[0], mode="bilinear")
        tobemergedto = imageandpatchs.estimation_updated_image
        tobemergedto[h1:h2, w1:w2] = np.multiply(tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
        imageandpatchs.set_updated_estimate(tobemergedto)
    return _resize_np(imageandpatchs.estimation_updated_image,
                      input_resolution[1], input_resolution[0], mode="bicubic")
#endregion


#region DETECTOR - LeresDetector: loads weights, runs inference
class LeresDetector:
    """LeReS depth detector with optional boost (leres++) mode."""

    def __init__(self, model: RelDepthModel, pix2pixmodel: Pix2Pix4Depth):
        self.model = model
        self.pix2pixmodel = pix2pixmodel

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "lllyasviel/Annotators",
        filename: str = "res101.pth",
        pix2pix_filename: str = "latest_net_G.pth",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        device: Optional[torch.device] = None,
    ) -> "LeresDetector":
        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub is required to download LeReS weights. "
                "Install it with: pip install huggingface_hub"
            )
        if device is None:
            device = _get_device()

        model_path = hf_hub_download(
            repo_id, filename, cache_dir=cache_dir, local_files_only=local_files_only
        )
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = _strip_prefix_if_present(checkpoint["depth_model"], "module.")
        # The res101.pth checkpoint stores the encoder under
        # "depth_model.encoder_modules.encoder.*" and the decoder under
        # "depth_model.decoder_modules.*", while _DepthNet uses
        # "depth_model.encoder.*" / "depth_model.decoder.*". Remap so the
        # keys line up with our module names.
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("depth_model.encoder_modules.encoder."):
                new_key = "depth_model.encoder." + key[len("depth_model.encoder_modules.encoder."):]
            elif key.startswith("depth_model.decoder_modules."):
                new_key = "depth_model.decoder." + key[len("depth_model.decoder_modules."):]
            else:
                new_key = key
            remapped[new_key] = value
        model = RelDepthModel(backbone="resnext101")
        model.load_state_dict(remapped, strict=True)
        del checkpoint
        model.to(device)
        model.eval()

        pix2pix_path = hf_hub_download(
            repo_id, pix2pix_filename, cache_dir=cache_dir, local_files_only=local_files_only
        )
        pix2pixmodel = Pix2Pix4Depth(device)
        pix2pixmodel.load_network(pix2pix_path)

        return cls(model, pix2pixmodel)

    def to(self, device: torch.device) -> "LeresDetector":
        self.model.to(device)
        return self

    def __call__(
        self,
        input_image: np.ndarray,
        thr_a: float = 0,
        thr_b: float = 0,
        boost: bool = False,
        detect_resolution: int = 512,
        image_resolution: int = 512,
    ) -> np.ndarray:
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        input_image = _hwc3(input_image)
        input_image = _resize_image(input_image, detect_resolution)
        assert input_image.ndim == 3
        height, width, _ = input_image.shape

        with torch.no_grad():
            if boost:
                depth = estimateboost(input_image, self.model, self.pix2pixmodel, max(width, height))
            else:
                depth = estimateleres(input_image, self.model, width, height)

        # Normalize to 16-bit then convert to 8-bit
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2 ** 16) - 1
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape)
        depth_image = out.astype("uint16")
        depth_image = (depth_image.astype(np.float64) * (255.0 / 65535.0)).clip(0, 255).astype(np.uint8)

        # Remove near
        if thr_a != 0:
            thr_a = (thr_a / 100) * 255
            depth_image = _threshold_tozero(depth_image, thr_a)
        # Remove bg (no inversion - closer objects are lighter)
        if thr_b != 0:
            thr_b = (thr_b / 100) * 255
            depth_image = _threshold_tozero(depth_image, thr_b)

        detected_map = _hwc3(depth_image)
        img = _resize_image(input_image, image_resolution)
        h, w, _ = img.shape
        detected_map = _resize_np(detected_map, w, h, mode="bilinear")
        return detected_map
#endregion
