"""
TITLE::Image Face Aspect Crop
DESCRIPTIONSHORT::Crops an image to a target aspect ratio with face-detection-biased anchoring and debug overlay output.
VERSION::20260815
IMAGE::comfyui_illumorae_image_face_aspect_crop.png
GROUP::Image
GROUPORDER::1
LISTORDER::20
STATUS::working

Implementation notes (efficiency-first):
- Uses OpenCV Haar Cascade face detection instead of neural detectors.
- Avoids loading external checkpoints / large models into VRAM.
- Prioritizes low-memory, fast, "good-enough" subject framing for preprocessing.
- Uses weighted heuristics (largest face + center/top/secondary-face blending) to stabilize crops.
"""
#region IMPORTS
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

MAX_RESOLUTION = 8192
#endregion


class illumoraeImageFaceAspectCropNode:
    """
    Aspect-ratio crop with face-biased anchoring.

    Design goal:
    - Keep runtime and memory usage light for ComfyUI preprocessing workflows.
    - Prefer deterministic classical CV + heuristics over heavier ML inference.

    Detection strategy:
    - OpenCV Haar cascade (CPU-friendly, bundled XML in cv2.data.haarcascades).
    - Largest detected face is always the primary anchor candidate.
    - Optional secondary-face blending helps multi-person shots frame more naturally.
    """

    #region C-PIPE
    # Main batch pipeline and FUNCTION entry point. Orchestrates the per-image
    # workflow: detect faces -> compute crop size -> compute anchor -> compute
    # box -> slice -> emit cropped + debug overlay tensors.
    def crop_to_aspect(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        face_anchor_strength: float,
        center_bias_weight: float,
        subject_top_bias: float,
        secondary_face_weight: float,
        max_anchor_shift_ratio: float,
        detect_scale_factor: float,
        detect_min_neighbors: int,
        min_face_percent: float,
        debug_prints: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main batch pipeline.

        Steps per image:
        1) Compute ratio-correct crop dimensions.
        2) Detect faces (lightweight Haar cascade).
        3) Compute face-biased crop anchor using weighted heuristics.
        4) Clamp crop box and slice image.
        5) Emit cropped image + debug overlay image.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        batch_size, src_h, src_w, _ = image.shape
        crop_w, crop_h = self._compute_crop_size(src_w, src_h, max(1, width), max(1, height))

        cropped_images = []
        debug_images = []

        for b in range(batch_size):
            image_np = image[b].detach().cpu().numpy().astype(np.float32)
            faces = self._detect_faces(
                image_np,
                detect_scale_factor=detect_scale_factor,
                detect_min_neighbors=detect_min_neighbors,
                min_face_percent=min_face_percent,
            )

            crop_center_x, crop_center_y = self._compute_anchor(
                src_w=src_w,
                src_h=src_h,
                crop_w=crop_w,
                crop_h=crop_h,
                faces=faces,
                face_anchor_strength=face_anchor_strength,
                center_bias_weight=center_bias_weight,
                subject_top_bias=subject_top_bias,
                secondary_face_weight=secondary_face_weight,
                max_anchor_shift_ratio=max_anchor_shift_ratio,
            )

            crop_box = self._compute_crop_box(
                src_w=src_w,
                src_h=src_h,
                crop_w=crop_w,
                crop_h=crop_h,
                crop_center_x=crop_center_x,
                crop_center_y=crop_center_y,
            )

            x0, y0, cw, ch = crop_box
            cropped = image_np[y0:y0 + ch, x0:x0 + cw, :]
            debug_img = self._build_debug_image(image_np, crop_box, faces, width, height)

            self._debug_print(
                debug_prints,
                f"batch={b} input={src_w}x{src_h} crop={cw}x{ch} offset=({x0},{y0}) faces={len(faces)}",
            )

            cropped_images.append(torch.from_numpy(cropped).float())
            debug_images.append(torch.from_numpy(debug_img).float())

        return (torch.stack(cropped_images, dim=0), torch.stack(debug_images, dim=0))
    #endregion

    #region C-DETECT
    # Face detection via OpenCV Haar cascade. Lazily loads and caches the
    # frontal-face classifier; converts to grayscale + equalizes histogram
    # before running detectMultiScale. Returns faces sorted by area descending.
    def _get_face_cascade(self) -> Optional[cv2.CascadeClassifier]:
        """
        Lazily initialize and cache Haar cascade detector.

        Why this approach:
        - Haar cascades are lightweight and do not require loading a neural checkpoint.
        - Cached instance avoids repeated initialization overhead across calls.
        """
        if self._face_cascade is not None:
            return self._face_cascade

        # Uses OpenCV's bundled frontal-face cascade file.
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return None

        self._face_cascade = cascade
        return self._face_cascade

    def _detect_faces(
        self,
        image_rgb: np.ndarray,
        detect_scale_factor: float,
        detect_min_neighbors: int,
        min_face_percent: float,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using classical multi-scale Haar cascade.

        Technique details:
        - Convert float RGB [0..1] to uint8 grayscale for cascade processing.
        - Apply histogram equalization to improve contrast robustness.
        - Run detectMultiScale over an image pyramid.

        Key controls:
        - detect_scale_factor: pyramid step size (lower = finer but slower).
        - detect_min_neighbors: stricter acceptance (higher = fewer false positives).
        - min_face_percent: rejects tiny boxes to reduce noisy detections.

        Returns faces sorted by area descending so faces[0] is always largest.
        """
        cascade = self._get_face_cascade()
        if cascade is None:
            return []

        h, w = image_rgb.shape[:2]
        # Dynamic minimum face size from input dimensions.
        min_dim = max(12, int(round(min(h, w) * (min_face_percent / 100.0))))

        # Lightweight preprocessing for classical detector stability.
        gray = cv2.cvtColor((np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=detect_scale_factor,
            minNeighbors=detect_min_neighbors,
            minSize=(min_dim, min_dim),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        faces_list = [(int(x), int(y), int(wf), int(hf)) for (x, y, wf, hf) in faces]
        faces_list.sort(key=lambda r: r[2] * r[3], reverse=True)
        return faces_list
    #endregion

    #region C-CROP
    # Crop geometry: compute aspect-correct crop dimensions, face-biased anchor
    # center, and integer crop box clamped to image bounds. These are pure
    # functions with no side effects, called in sequence by the pipeline.
    def _compute_crop_size(self, src_w: int, src_h: int, target_w: int, target_h: int) -> Tuple[int, int]:
        """
        Compute the largest in-bounds crop box that matches target aspect ratio.

        This does not resize pixels; it only decides crop dimensions.
        """
        target_ratio = target_w / target_h
        src_ratio = src_w / src_h

        if src_ratio > target_ratio:
            crop_h = src_h
            crop_w = max(1, int(round(crop_h * target_ratio)))
        else:
            crop_w = src_w
            crop_h = max(1, int(round(crop_w / target_ratio)))

        crop_w = min(crop_w, src_w)
        crop_h = min(crop_h, src_h)
        return crop_w, crop_h

    def _compute_anchor(
        self,
        src_w: int,
        src_h: int,
        crop_w: int,
        crop_h: int,
        faces: List[Tuple[int, int, int, int]],
        face_anchor_strength: float,
        center_bias_weight: float,
        subject_top_bias: float,
        secondary_face_weight: float,
        max_anchor_shift_ratio: float,
    ) -> Tuple[float, float]:
        """
        Compute crop center using weighted heuristics.

        Heuristics used:
        1) Largest face as primary anchor (always preferred when available).
        2) Optional blend toward area-weighted center of secondary faces.
        3) Optional pull toward image center to reduce jitter/overreaction.
        4) Optional vertical bias that places the subject higher in the frame.
        5) Hard clamp on max anchor shift from center for stability.

        This intentionally favors robust, predictable framing over detector precision.

        Note on parameter interaction: `center_bias_weight` (step 3) is applied
        to the face anchor BEFORE `face_anchor_strength` blends that anchor with
        the image center (final crop center line). The two are therefore
        multiplicative, not independent. With `face_anchor_strength=1.0` the
        crop center equals the center-pulled anchor, so `center_bias_weight`
        still exerts influence. The effective face weight is approximately
        `face_anchor_strength * (1 - center_bias_weight)`.
        """
        center_x = src_w * 0.5
        center_y = src_h * 0.5

        if faces:
            # Primary anchor: largest detected face.
            largest_x, largest_y, largest_w, largest_h = faces[0]
            anchor_x = largest_x + (largest_w * 0.5)
            # 0.43 places the anchor at ~43% of face height from the top, near the
            # eye line. This is above the face center (50%) and approximates the
            # upper-third compositional line, keeping the crop framed on the eyes
            # rather than the chin/neck.
            anchor_y = largest_y + (largest_h * 0.43)

            if len(faces) > 1 and secondary_face_weight > 0.0:
                sec_sum_x = 0.0
                sec_sum_y = 0.0
                sec_sum_w = 0.0
                for sx, sy, sw, sh in faces[1:]:
                    area = float(sw * sh)
                    sec_sum_x += (sx + sw * 0.5) * area
                    sec_sum_y += (sy + sh * 0.5) * area
                    sec_sum_w += area
                if sec_sum_w > 0.0:
                    sec_center_x = sec_sum_x / sec_sum_w
                    sec_center_y = sec_sum_y / sec_sum_w
                    # Saturating blend function keeps secondary influence bounded.
                    blend = secondary_face_weight / (1.0 + secondary_face_weight)
                    anchor_x = anchor_x * (1.0 - blend) + sec_center_x * blend
                    anchor_y = anchor_y * (1.0 - blend) + sec_center_y * blend

            # Pull anchor toward image center to avoid extreme crops.
            anchor_x = anchor_x + (center_x - anchor_x) * center_bias_weight
            anchor_y = anchor_y + (center_y - anchor_y) * center_bias_weight
            # Positive subject_top_bias shifts the crop center downward, which
            # places the face higher in the resulting crop (upper-third framing).
            # This reduces headroom above the subject. The 0.12 factor scales the
            # shift to ~12% of crop height per unit of bias, keeping the effect
            # proportional to the crop size rather than the source size.
            anchor_y = anchor_y + (crop_h * 0.12 * subject_top_bias)

            crop_center_x = center_x * (1.0 - face_anchor_strength) + anchor_x * face_anchor_strength
            crop_center_y = center_y * (1.0 - face_anchor_strength) + anchor_y * face_anchor_strength
        else:
            # No detection fallback: neutral center crop.
            crop_center_x = center_x
            crop_center_y = center_y

        # Safety clamp: limit max offset from center to prevent unstable framing jumps.
        max_shift_x = (src_w - crop_w) * 0.5 * max_anchor_shift_ratio
        max_shift_y = (src_h - crop_h) * 0.5 * max_anchor_shift_ratio
        crop_center_x = center_x + np.clip(crop_center_x - center_x, -max_shift_x, max_shift_x)
        crop_center_y = center_y + np.clip(crop_center_y - center_y, -max_shift_y, max_shift_y)

        return float(crop_center_x), float(crop_center_y)

    def _compute_crop_box(
        self,
        src_w: int,
        src_h: int,
        crop_w: int,
        crop_h: int,
        crop_center_x: float,
        crop_center_y: float,
    ) -> Tuple[int, int, int, int]:
        """
        Convert crop center + size to integer box and clamp to valid image bounds.
        """
        x0 = int(round(crop_center_x - (crop_w * 0.5)))
        y0 = int(round(crop_center_y - (crop_h * 0.5)))

        max_x0 = max(0, src_w - crop_w)
        max_y0 = max(0, src_h - crop_h)
        x0 = int(np.clip(x0, 0, max_x0))
        y0 = int(np.clip(y0, 0, max_y0))

        return x0, y0, crop_w, crop_h
    #endregion

    #region C-DEBUG
    # Visual diagnostic overlay: draws the chosen crop rectangle, face boxes
    # (green = primary, orange = secondary), and a text summary onto a copy
    # of the source image. Always emitted as the second output tensor.
    def _build_debug_image(
        self,
        image_rgb: np.ndarray,
        crop_box: Tuple[int, int, int, int],
        faces: List[Tuple[int, int, int, int]],
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """
        Build a visual diagnostic overlay.

        Draws:
        - Chosen crop rectangle
        - Largest face (green)
        - Secondary faces (orange)
        - Text summary (input size, target ratio, crop box, face count)
        """
        debug_bgr = cv2.cvtColor((np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

        x0, y0, cw, ch = crop_box
        cv2.rectangle(debug_bgr, (x0, y0), (x0 + cw, y0 + ch), (255, 128, 0), 2)

        if faces:
            lx, ly, lw, lh = faces[0]
            cv2.rectangle(debug_bgr, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
            for sx, sy, sw, sh in faces[1:]:
                cv2.rectangle(debug_bgr, (sx, sy), (sx + sw, sy + sh), (0, 165, 255), 1)

        h, w = image_rgb.shape[:2]
        lines = [
            f"input: {w}x{h}",
            f"target ratio: {target_w}:{target_h}",
            f"crop: {cw}x{ch} @ ({x0},{y0})",
            f"faces: {len(faces)}",
        ]

        text_y = 24
        for line in lines:
            cv2.putText(debug_bgr, line, (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(debug_bgr, line, (12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            text_y += 24

        debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return debug_rgb
    #endregion

    #region C-UTIL
    # Instance state and lightweight helpers. __init__ initializes the lazy
    # cascade cache; _debug_print gates stdout output behind the debug_prints
    # flag.
    def __init__(self):
        self._face_cascade = None

    def _debug_print(self, debug_prints: bool, *args) -> None:
        if debug_prints:
            print(*args)
    #endregion

    #region C-UI
    # ComfyUI-facing declarations: input schema, return types, function binding,
    # category, and human-readable description. Placed last so the core
    # pipeline logic reads first when scanning the file top-down.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 576, "min": 16, "max": MAX_RESOLUTION, "step": 1}),
                "face_anchor_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_bias_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "subject_top_bias": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01}),
                "secondary_face_weight": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_anchor_shift_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
                "detect_scale_factor": ("FLOAT", {"default": 1.1, "min": 1.01, "max": 1.5, "step": 0.01}),
                "detect_min_neighbors": ("INT", {"default": 5, "min": 1, "max": 12, "step": 1}),
                "min_face_percent": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 40.0, "step": 0.5}),
            },
            "optional": {
                "debug_prints": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("cropped_image", "debug_image")
    FUNCTION = "crop_to_aspect"
    CATEGORY = "illumorae"
    DESCRIPTION = "Crops an image to a target aspect ratio, anchoring around the largest detected face, and emits a debug overlay image showing the crop box and detected faces."
    #endregion


#region REG
NODE_CLASS_MAPPINGS = {
    "illumoraeImageFaceAspectCropNode": illumoraeImageFaceAspectCropNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeImageFaceAspectCropNode": "Image Face Aspect Crop",
}
#endregion
