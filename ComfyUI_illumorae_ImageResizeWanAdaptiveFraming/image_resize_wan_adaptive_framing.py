"""
TITLE::Image Resize WAN Adaptive Framing
DESCRIPTIONSHORT::Auto-selects WAN-friendly vertical/horizontal resolution and applies optional human-centric framing before resize.
VERSION::20260412
IMAGE::comfyui_illumorae_image_resize_wan_adaptive_framing.png
GROUP::Image

NOTES:
- Uses lightweight OpenCV detectors (Haar face + HOG person), not deep checkpoints.
- Chooses a WAN-friendly preset by orientation and aspect/area similarity.
- Applies anchor framing (face/person/center) before resize.
- Supports crop mode (content-preserving framing) and pad mode (fit with soft bars).
- Prioritizes low memory over detector accuracy.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from comfy.utils import common_upscale

MAX_RESOLUTION = 4096


class illumoraeImageResizeWanAdaptiveFramingNode:
    """
    Adaptive image-to-video pre-resizer for WAN-style target resolutions.

    Goals:
    - Automatically map source images to WAN resolution presets.
    - Preserve subject framing using low-cost face/body detection and anchor .
    - Avoid loading any additional neural model checkpoints into VRAM.
    """

    WAN_TIER_PRESETS_LANDSCAPE = {
        "small": [(416, 240)],   # half of main
        "main": [(832, 480)],
        "high": [(1280, 720)],
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "orientation_mode": (["auto", "force_horizontal", "force_vertical"], {"default": "auto"}),
                "framing_mode": (["auto_subject", "human_face", "human_body", "center"], {"default": "auto_subject"}),
                "resize_mode": (["crop", "pad"], {"default": "crop"}),
                "resolution_tier": (["auto", "small", "main", "high"], {"default": "auto"}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "lanczos"}),
                "anchor_strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_bias": ("FLOAT", {"default": 0.18, "min": -0.5, "max": 0.5, "step": 0.01}),
                "face_min_percent": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 40.0, "step": 0.5}),
                "detect_scale_factor": ("FLOAT", {"default": 1.1, "min": 1.01, "max": 1.5, "step": 0.01}),
                "detect_min_neighbors": ("INT", {"default": 5, "min": 1, "max": 12, "step": 1}),
            },
            "optional": {
                "debug_prints": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("resized_image", "debug_image", "width", "height", "selected_preset")
    FUNCTION = "resize_adaptive"
    CATEGORY = "illumorae"
    DESCRIPTION = "Auto-selects WAN-friendly resolution and applies subject/human-centric framing to fit source images for video prep."

    def __init__(self):
        self._face_cascade = None
        self._hog = None

    def _debug_print(self, debug_prints: bool, *args):
        if debug_prints:
            print(*args)

    def _get_face_cascade(self):
        """Lazily load and cache OpenCV Haar face detector."""
        if self._face_cascade is not None:
            return self._face_cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return None
        self._face_cascade = cascade
        return self._face_cascade

    def _get_hog(self):
        """Lazily load and cache OpenCV HOG person detector."""
        if self._hog is not None:
            return self._hog
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._hog = hog
        return self._hog

    def _detect_faces(
        self,
        image_rgb: np.ndarray,
        scale_factor: float,
        min_neighbors: int,
        face_min_percent: float,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect candidate faces via multi-scale Haar cascade.

        Returns area-sorted detections so index 0 is largest face.
        """
        cascade = self._get_face_cascade()
        if cascade is None:
            return []

        h, w = image_rgb.shape[:2]
        min_dim = max(12, int(round(min(h, w) * face_min_percent / 100.0)))
        gray = cv2.cvtColor((np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_dim, min_dim),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        face_list = [(int(x), int(y), int(wf), int(hf)) for (x, y, wf, hf) in faces]
        face_list.sort(key=lambda r: r[2] * r[3], reverse=True)
        return face_list

    def _detect_people(self, image_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect human body boxes with OpenCV HOG descriptor.

        Used as fallback when face detection is unavailable/weak.
        """
        hog = self._get_hog()
        if hog is None:
            return []

        img_u8 = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

        rects, _ = hog.detectMultiScale(
            gray,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        people = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]
        people.sort(key=lambda r: r[2] * r[3], reverse=True)
        return people

    def _choose_orientation(self, src_w: int, src_h: int, orientation_mode: str) -> str:
        """Resolve output orientation: forced horizontal/vertical or auto by source geometry."""
        if orientation_mode == "force_vertical":
            return "vertical"
        if orientation_mode == "force_horizontal":
            return "horizontal"
        return "horizontal" if src_w >= src_h else "vertical"

    def _get_candidates(self, orientation: str, tier: str) -> List[Tuple[int, int]]:
        """
        Build candidate preset list based on orientation and WAN tier.

        Requested WAN tiers:
        - high: 1280x720 (or 720x1280)
        - main: 832x480 (or 480x832)
        - small: half main = 416x240 (or 240x416)

        Vertical presets are derived by transposing landscape presets.
        """
        if tier in self.WAN_TIER_PRESETS_LANDSCAPE:
            landscape_candidates = list(self.WAN_TIER_PRESETS_LANDSCAPE[tier])
        else:
            # auto: allow chooser to match against all known WAN tiers
            landscape_candidates = []
            for tier_name in ("small", "main", "high"):
                landscape_candidates.extend(self.WAN_TIER_PRESETS_LANDSCAPE[tier_name])

        if orientation == "vertical":
            all_candidates = [(h, w) for (w, h) in landscape_candidates]
        else:
            all_candidates = landscape_candidates

        return all_candidates

    def _select_preset(
        self,
        src_w: int,
        src_h: int,
        orientation: str,
        tier: str,
    ) -> Tuple[int, int]:
        """
        Pick best preset using ratio + area scoring.

        Score favors aspect-ratio match first, then closeness of pixel area.
        """
        candidates = self._get_candidates(orientation, tier)
        src_ratio = src_w / src_h
        src_area = float(src_w * src_h)

        best = candidates[0]
        best_score = 1e18

        for tw, th in candidates:
            ratio = tw / th
            ratio_error = abs(src_ratio - ratio)
            area_error = abs(np.log(max(1.0, src_area) / float(max(1, tw * th))))
            score = ratio_error * 4.0 + area_error
            if score < best_score:
                best_score = score
                best = (tw, th)

        return best

    def _compute_crop_size(self, src_w: int, src_h: int, target_w: int, target_h: int) -> Tuple[int, int]:
        """Compute maximal in-bounds crop that matches target aspect ratio."""
        src_ratio = src_w / src_h
        target_ratio = target_w / target_h

        if src_ratio > target_ratio:
            crop_h = src_h
            crop_w = max(1, int(round(crop_h * target_ratio)))
        else:
            crop_w = src_w
            crop_h = max(1, int(round(crop_w / target_ratio)))

        return min(crop_w, src_w), min(crop_h, src_h)

    def _get_anchor(
        self,
        src_w: int,
        src_h: int,
        faces: List[Tuple[int, int, int, int]],
        people: List[Tuple[int, int, int, int]],
        framing_mode: str,
        anchor_strength: float,
        top_bias: float,
    ) -> Tuple[float, float, Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], str]:
        """
        Compute framing anchor from face/person/center heuristics.

        Priority by framing_mode:
        - human_face / auto_subject: largest face center-biased eye-line
        - human_body / auto_subject: largest person upper-body bias
        - center: geometric center fallback
        """
        center_x = src_w * 0.5
        center_y = src_h * 0.5

        primary_face = faces[0] if faces else None
        primary_person = people[0] if people else None

        chosen_type = "center"
        target_x = center_x
        target_y = center_y

        if framing_mode in ("human_face", "auto_subject") and primary_face is not None:
            x, y, w, h = primary_face
            target_x = x + w * 0.5
            target_y = y + h * 0.42
            chosen_type = "face"
        elif framing_mode in ("human_body", "auto_subject") and primary_person is not None:
            x, y, w, h = primary_person
            target_x = x + w * 0.5
            target_y = y + h * 0.32
            chosen_type = "person"

        target_y = target_y + (src_h * top_bias * 0.2)
        anchor_x = center_x * (1.0 - anchor_strength) + target_x * anchor_strength
        anchor_y = center_y * (1.0 - anchor_strength) + target_y * anchor_strength

        return anchor_x, anchor_y, primary_face, primary_person, chosen_type

    def _compute_crop_box(
        self,
        src_w: int,
        src_h: int,
        crop_w: int,
        crop_h: int,
        anchor_x: float,
        anchor_y: float,
    ) -> Tuple[int, int, int, int]:
        """Convert anchor + crop size into clamped integer crop bounds."""
        x0 = int(round(anchor_x - crop_w * 0.5))
        y0 = int(round(anchor_y - crop_h * 0.5))

        x0 = int(np.clip(x0, 0, max(0, src_w - crop_w)))
        y0 = int(np.clip(y0, 0, max(0, src_h - crop_h)))
        return x0, y0, crop_w, crop_h

    def _resize_crop(self, crop_np: np.ndarray, target_w: int, target_h: int, upscale_method: str) -> np.ndarray:
        """Resize a cropped region to exact target size via ComfyUI common_upscale."""
        crop_t = torch.from_numpy(crop_np).float().unsqueeze(0)
        resized = common_upscale(crop_t.movedim(-1, 1), target_w, target_h, upscale_method, None).movedim(1, -1)
        return resized[0].detach().cpu().numpy().astype(np.float32)

    def _resize_pad(
        self,
        src_np: np.ndarray,
        target_w: int,
        target_h: int,
        anchor_x: float,
        anchor_y: float,
        upscale_method: str,
    ) -> np.ndarray:
        """
        Fit source into target with soft-bar padding.

        Anchor influences fit placement; empty bars are filled using blurred full-frame proxy.
        """
        src_h, src_w = src_np.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        fit_w = max(1, int(round(src_w * scale)))
        fit_h = max(1, int(round(src_h * scale)))

        resized_src = self._resize_crop(src_np, fit_w, fit_h, upscale_method)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.float32)
        x_ratio = anchor_x / max(1.0, src_w)
        y_ratio = anchor_y / max(1.0, src_h)

        x0 = int(round((target_w - fit_w) * (0.5 + (x_ratio - 0.5) * 0.6)))
        y0 = int(round((target_h - fit_h) * (0.5 + (y_ratio - 0.5) * 0.6)))
        x0 = int(np.clip(x0, 0, max(0, target_w - fit_w)))
        y0 = int(np.clip(y0, 0, max(0, target_h - fit_h)))

        canvas[y0:y0 + fit_h, x0:x0 + fit_w, :] = resized_src

        # Lightweight bar fill: avoids stark black bars while staying deterministic.
        blur = cv2.GaussianBlur(self._resize_crop(src_np, target_w, target_h, upscale_method), (0, 0), 6.0)
        mask = np.zeros((target_h, target_w), dtype=np.float32)
        mask[y0:y0 + fit_h, x0:x0 + fit_w] = 1.0
        mask_3 = np.stack([mask] * 3, axis=-1)
        return canvas * mask_3 + blur * (1.0 - mask_3)

    def _build_debug(
        self,
        src_np: np.ndarray,
        crop_box: Tuple[int, int, int, int],
        target_w: int,
        target_h: int,
        primary_face: Optional[Tuple[int, int, int, int]],
        primary_person: Optional[Tuple[int, int, int, int]],
        chosen_type: str,
    ) -> np.ndarray:
        """Build debug overlay with crop box, detections, and selected anchor mode."""
        dbg = cv2.cvtColor((np.clip(src_np, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

        x0, y0, cw, ch = crop_box
        cv2.rectangle(dbg, (x0, y0), (x0 + cw, y0 + ch), (255, 180, 0), 2)

        if primary_face is not None:
            x, y, w, h = primary_face
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (40, 220, 40), 2)
        if primary_person is not None:
            x, y, w, h = primary_person
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 180, 255), 2)

        h0, w0 = src_np.shape[:2]
        lines = [
            f"input: {w0}x{h0}",
            f"target: {target_w}x{target_h}",
            f"crop: {cw}x{ch} @ ({x0},{y0})",
            f"anchor: {chosen_type}",
        ]

        y = 24
        for line in lines:
            cv2.putText(dbg, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(dbg, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24

        return cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def resize_adaptive(
        self,
        image: torch.Tensor,
        orientation_mode: str,
        framing_mode: str,
        resize_mode: str,
        resolution_tier: str,
        upscale_method: str,
        anchor_strength: float,
        top_bias: float,
        face_min_percent: float,
        detect_scale_factor: float,
        detect_min_neighbors: int,
        debug_prints: bool = False,
    ):
        """
        Main adaptive resize pipeline.

        Steps per image:
        1) Resolve orientation (auto/forced).
        2) Select WAN-friendly preset resolution.
        3) Detect faces and people with lightweight CV detectors.
        4) Compute anchor and crop box.
        5) Apply crop or pad resize mode.
        6) Return resized image, debug overlay, and chosen preset metadata.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size = image.shape[0]
        outputs = []
        debug_outputs = []

        selected_w = 0
        selected_h = 0
        selected_label = ""

        for b in range(batch_size):
            src_np = image[b].detach().cpu().numpy().astype(np.float32)
            src_h, src_w = src_np.shape[:2]

            orientation = self._choose_orientation(src_w, src_h, orientation_mode)
            target_w, target_h = self._select_preset(src_w, src_h, orientation, resolution_tier)
            selected_w = target_w
            selected_h = target_h
            selected_label = f"{target_w}x{target_h} ({orientation})"

            faces = self._detect_faces(src_np, detect_scale_factor, detect_min_neighbors, face_min_percent)
            people = self._detect_people(src_np)

            anchor_x, anchor_y, primary_face, primary_person, chosen_type = self._get_anchor(
                src_w=src_w,
                src_h=src_h,
                faces=faces,
                people=people,
                framing_mode=framing_mode,
                anchor_strength=anchor_strength,
                top_bias=top_bias,
            )

            crop_w, crop_h = self._compute_crop_size(src_w, src_h, target_w, target_h)
            crop_box = self._compute_crop_box(src_w, src_h, crop_w, crop_h, anchor_x, anchor_y)
            x0, y0, cw, ch = crop_box

            if resize_mode == "pad":
                resized_np = self._resize_pad(src_np, target_w, target_h, anchor_x, anchor_y, upscale_method)
            else:
                crop_np = src_np[y0:y0 + ch, x0:x0 + cw, :]
                resized_np = self._resize_crop(crop_np, target_w, target_h, upscale_method)

            debug_np = self._build_debug(
                src_np=src_np,
                crop_box=crop_box,
                target_w=target_w,
                target_h=target_h,
                primary_face=primary_face,
                primary_person=primary_person,
                chosen_type=chosen_type,
            )

            self._debug_print(
                debug_prints,
                f"batch={b} src={src_w}x{src_h} target={target_w}x{target_h} mode={resize_mode} anchor={chosen_type}",
            )

            outputs.append(torch.from_numpy(resized_np).float())
            debug_outputs.append(torch.from_numpy(debug_np).float())

        return (
            torch.stack(outputs, dim=0),
            torch.stack(debug_outputs, dim=0),
            int(selected_w),
            int(selected_h),
            selected_label,
        )


NODE_CLASS_MAPPINGS = {
    "illumoraeImageResizeWanAdaptiveFramingNode": illumoraeImageResizeWanAdaptiveFramingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeImageResizeWanAdaptiveFramingNode": "Image Resize WAN Adaptive Framing",
}
