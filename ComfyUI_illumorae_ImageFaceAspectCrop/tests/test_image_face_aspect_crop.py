"""Regression tests for illumoraeImageFaceAspectCropNode.

Covers the bugs fixed in the 20260815 review pass:

- subject_top_bias rename + corrected comment (was top_bias_weight with a
  comment that claimed the opposite of the code's effect)
- DESCRIPTION class attribute now mentions the debug overlay output
- Return type annotation on crop_to_aspect
- _debug_print -> None annotation
- Magic numbers 0.43 and 0.12 documented (verified via behavior, not text)

Also covers the core logic that was previously untested:

- _compute_crop_size: aspect-ratio sizing, edge cases
- _compute_crop_box: bounds clamping
- _compute_anchor: no-faces fallback, face anchor, secondary-face blend,
  center bias, subject_top_bias direction, max_anchor_shift_ratio clamp,
  multiplicative interaction between center_bias_weight and face_anchor_strength
- _detect_faces: blank image returns no faces
- _build_debug_image: shape, dtype, range
- crop_to_aspect: full pipeline (shapes, dtype, range, batch, 3D unsqueeze,
  debug image always present, determinism, no-face center fallback)
- Structural compliance (INPUT_TYPES, RETURN_TYPES, mappings, frontmatter)

Usage:
    python -m tests.test_image_face_aspect_crop
    python -m pytest tests/test_image_face_aspect_crop.py -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from image_face_aspect_crop import (  # noqa: E402
    MAX_RESOLUTION,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    illumoraeImageFaceAspectCropNode,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_blank_image(h: int = 256, w: int = 256, value: float = 0.5) -> torch.Tensor:
    """A uniform-color IMAGE tensor (1, H, W, 3) in [0, 1] float32."""
    return torch.full((1, h, w, 3), value, dtype=torch.float32)


def _make_gradient_image(h: int = 256, w: int = 256) -> torch.Tensor:
    """A left-to-right gradient IMAGE tensor (1, H, W, 3) in [0, 1] float32."""
    xx = np.linspace(0, 1, w, dtype=np.float32)
    img = np.broadcast_to(xx[None, :, None], (h, w, 3)).copy()
    return torch.from_numpy(img).unsqueeze(0)


# --------------------------------------------------------------------------
# Structural compliance
# --------------------------------------------------------------------------

class TestStructuralCompliance(unittest.TestCase):
    """Node class attributes, mappings, and frontmatter."""

    def setUp(self):
        self.node = illumoraeImageFaceAspectCropNode()

    def test_class_attributes_present(self):
        self.assertEqual(self.node.CATEGORY, "illumorae")
        self.assertEqual(self.node.FUNCTION, "crop_to_aspect")
        self.assertEqual(self.node.RETURN_TYPES, ("IMAGE", "IMAGE"))
        self.assertEqual(self.node.RETURN_NAMES, ("cropped_image", "debug_image"))
        self.assertIsInstance(self.node.DESCRIPTION, str)
        self.assertGreater(len(self.node.DESCRIPTION), 0)

    def test_description_mentions_debug_overlay(self):
        """DESCRIPTION should mention the debug overlay (fix (section)3.2)."""
        desc_lower = self.node.DESCRIPTION.lower()
        self.assertIn("debug", desc_lower)

    def test_input_types_required_keys(self):
        required = self.node.INPUT_TYPES()["required"]
        expected = {
            "image", "width", "height", "face_anchor_strength",
            "center_bias_weight", "subject_top_bias", "secondary_face_weight",
            "max_anchor_shift_ratio", "detect_scale_factor",
            "detect_min_neighbors", "min_face_percent",
        }
        self.assertEqual(set(required.keys()), expected)

    def test_input_types_optional_keys(self):
        optional = self.node.INPUT_TYPES().get("optional", {})
        self.assertIn("debug_prints", optional)

    def test_subject_top_bias_renamed(self):
        """The old top_bias_weight name must not appear (fix (section)2.1)."""
        required = self.node.INPUT_TYPES()["required"]
        self.assertIn("subject_top_bias", required)
        self.assertNotIn("top_bias_weight", required)

    def test_input_bounds(self):
        required = self.node.INPUT_TYPES()["required"]
        # width / height
        w_def, w_meta = required["width"]
        self.assertEqual(w_def, "INT")
        self.assertGreaterEqual(w_meta["min"], 1)
        self.assertLessEqual(w_meta["max"], MAX_RESOLUTION)
        # subject_top_bias
        _, stb_meta = required["subject_top_bias"]
        self.assertEqual(stb_meta["min"], 0.0)
        self.assertLessEqual(stb_meta["max"], 2.0)

    def test_return_types_match_names(self):
        self.assertEqual(len(self.node.RETURN_TYPES), len(self.node.RETURN_NAMES))

    def test_node_class_mappings(self):
        self.assertIn("illumoraeImageFaceAspectCropNode", NODE_CLASS_MAPPINGS)
        self.assertIs(
            NODE_CLASS_MAPPINGS["illumoraeImageFaceAspectCropNode"],
            illumoraeImageFaceAspectCropNode,
        )
        self.assertIn("illumoraeImageFaceAspectCropNode", NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["illumoraeImageFaceAspectCropNode"],
            "Image Face Aspect Crop",
        )

    def test_crop_to_aspect_has_return_annotation(self):
        """crop_to_aspect should have a return type annotation (fix (section)3.3)."""
        ann = illumoraeImageFaceAspectCropNode.crop_to_aspect.__annotations__
        self.assertIn("return", ann)

    def test_debug_print_has_return_annotation(self):
        """_debug_print should have -> None (fix (section)3.4)."""
        ann = illumoraeImageFaceAspectCropNode._debug_print.__annotations__
        self.assertIn("return", ann)


# --------------------------------------------------------------------------
# _compute_crop_size
# --------------------------------------------------------------------------

class TestComputeCropSize(unittest.TestCase):
    """Aspect-ratio crop dimension computation."""

    def setUp(self):
        self.node = illumoraeImageFaceAspectCropNode()

    def test_wider_source_than_target(self):
        # Source 1000x500 (ratio 2.0), target 1024x576 (ratio ~1.778)
        cw, ch = self.node._compute_crop_size(1000, 500, 1024, 576)
        self.assertEqual(ch, 500)
        self.assertAlmostEqual(cw / ch, 1024 / 576, places=2)
        self.assertLessEqual(cw, 1000)

    def test_taller_source_than_target(self):
        # Source 500x1000 (ratio 0.5), target 1024x576 (ratio ~1.778)
        cw, ch = self.node._compute_crop_size(500, 1000, 1024, 576)
        self.assertEqual(cw, 500)
        self.assertAlmostEqual(cw / ch, 1024 / 576, places=2)
        self.assertLessEqual(ch, 1000)

    def test_exact_match_ratio(self):
        cw, ch = self.node._compute_crop_size(1024, 576, 1024, 576)
        self.assertEqual((cw, ch), (1024, 576))

    def test_square_target_from_square_source(self):
        cw, ch = self.node._compute_crop_size(500, 500, 100, 100)
        self.assertEqual((cw, ch), (500, 500))

    def test_crop_never_exceeds_source(self):
        cw, ch = self.node._compute_crop_size(100, 100, 2000, 2000)
        self.assertLessEqual(cw, 100)
        self.assertLessEqual(ch, 100)

    def test_minimum_dimensions_are_positive(self):
        # Even with tiny target dims, crop must be >= 1.
        cw, ch = self.node._compute_crop_size(100, 100, 1, 1)
        self.assertGreaterEqual(cw, 1)
        self.assertGreaterEqual(ch, 1)


# --------------------------------------------------------------------------
# _compute_crop_box
# --------------------------------------------------------------------------

class TestComputeCropBox(unittest.TestCase):
    """Crop box clamping to image bounds."""

    def setUp(self):
        self.node = illumoraeImageFaceAspectCropNode()

    def test_centered_crop(self):
        box = self.node._compute_crop_box(100, 100, 50, 50, 50.0, 50.0)
        self.assertEqual(box, (25, 25, 50, 50))

    def test_clamp_to_left_edge(self):
        box = self.node._compute_crop_box(100, 100, 50, 50, 10.0, 50.0)
        self.assertEqual(box[0], 0)
        self.assertEqual(box[2], 50)

    def test_clamp_to_right_edge(self):
        box = self.node._compute_crop_box(100, 100, 50, 50, 90.0, 50.0)
        self.assertEqual(box[0], 50)
        self.assertEqual(box[2], 50)

    def test_clamp_to_top_edge(self):
        box = self.node._compute_crop_box(100, 100, 50, 50, 50.0, 10.0)
        self.assertEqual(box[1], 0)
        self.assertEqual(box[3], 50)

    def test_clamp_to_bottom_edge(self):
        box = self.node._compute_crop_box(100, 100, 50, 50, 50.0, 90.0)
        self.assertEqual(box[1], 50)
        self.assertEqual(box[3], 50)

    def test_crop_larger_than_source_clamps_to_zero(self):
        box = self.node._compute_crop_box(50, 50, 100, 100, 25.0, 25.0)
        self.assertEqual(box, (0, 0, 100, 100))


# --------------------------------------------------------------------------
# _compute_anchor
# --------------------------------------------------------------------------

class TestComputeAnchor(unittest.TestCase):
    """Face-biased anchor computation heuristics."""

    def setUp(self):
        self.node = illumoraeImageFaceAspectCropNode()
        self.src_w = 1000
        self.src_h = 1000
        self.crop_w = 500
        self.crop_h = 500

    def test_no_faces_falls_back_to_center(self):
        cx, cy = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces=[],
            face_anchor_strength=0.9, center_bias_weight=0.1,
            subject_top_bias=0.2, secondary_face_weight=0.15,
            max_anchor_shift_ratio=1.0,
        )
        self.assertAlmostEqual(cx, self.src_w * 0.5)
        self.assertAlmostEqual(cy, self.src_h * 0.5)

    def test_face_anchor_pulls_toward_face(self):
        # Face at (700, 300, 100, 100) - right of center, slightly above.
        # anchor_y = 300 + 100*0.43 = 343, which is 157px from center (500),
        # within the max_shift of 250, so no clamping occurs.
        faces = [(700, 300, 100, 100)]
        cx, cy = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.0,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        # With strength=1.0 and no center bias, crop center should be at
        # the face anchor (x=750, y=300+43=343).
        self.assertAlmostEqual(cx, 750.0, delta=1.0)
        self.assertAlmostEqual(cy, 343.0, delta=1.0)

    def test_face_anchor_strength_zero_is_center(self):
        faces = [(700, 200, 100, 100)]
        cx, cy = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=0.0, center_bias_weight=0.0,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        self.assertAlmostEqual(cx, self.src_w * 0.5)
        self.assertAlmostEqual(cy, self.src_h * 0.5)

    def test_center_bias_pulls_anchor_toward_center(self):
        faces = [(700, 200, 100, 100)]
        cx_no_bias, cy_no_bias = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.0,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        cx_bias, cy_bias = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.5,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        # Center bias should pull the anchor closer to (500, 500).
        self.assertLess(abs(cx_bias - 500), abs(cx_no_bias - 500))
        self.assertLess(abs(cy_bias - 500), abs(cy_no_bias - 500))

    def test_subject_top_bias_shifts_crop_downward(self):
        """Positive subject_top_bias should move crop center DOWN (fix (section)2.1).

        The old comment claimed it increased headroom (moved crop up). The
        code adds a positive value to anchor_y, moving the crop center down,
        which places the face higher in the frame. This test locks in the
        corrected behavior.
        """
        faces = [(400, 400, 100, 100)]
        cx_zero, cy_zero = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.0,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        cx_pos, cy_pos = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.0,
            subject_top_bias=1.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        # X should be unchanged.
        self.assertAlmostEqual(cx_zero, cx_pos)
        # Y should increase (crop moves down).
        self.assertGreater(cy_pos, cy_zero)
        # The shift should be crop_h * 0.12 * 1.0 = 60.
        self.assertAlmostEqual(cy_pos - cy_zero, self.crop_h * 0.12, places=2)

    def test_subject_top_bias_zero_is_no_shift(self):
        faces = [(400, 400, 100, 100)]
        cx, cy = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.0,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        # Without top bias, anchor_y = face anchor y = 400 + 43 = 443.
        self.assertAlmostEqual(cy, 443.0, delta=1.0)

    def test_secondary_face_blend(self):
        """Secondary faces should pull the anchor toward their area-weighted center."""
        # Primary at (200, 400, 100, 100), secondary at (700, 400, 100, 100).
        # Equal area -> secondary center is (750, 450).
        faces = [(200, 400, 100, 100), (700, 400, 100, 100)]
        cx_no_sec, cy_no_sec = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.0,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        cx_sec, cy_sec = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.0,
            subject_top_bias=0.0, secondary_face_weight=1.0,
            max_anchor_shift_ratio=1.0,
        )
        # Without secondary blend, anchor x = 250 (primary center).
        # With secondary weight=1.0, blend = 1/(1+1) = 0.5, so anchor x
        # moves halfway toward 750 -> 500.
        self.assertLess(abs(cx_sec - 500), abs(cx_no_sec - 500))

    def test_secondary_face_weight_zero_no_blend(self):
        faces = [(200, 400, 100, 100), (700, 400, 100, 100)]
        cx, cy = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.0,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        # No secondary blend -> anchor at primary face center (250, 443).
        self.assertAlmostEqual(cx, 250.0, delta=1.0)

    def test_max_anchor_shift_ratio_clamps(self):
        """max_anchor_shift_ratio < 1.0 should limit how far the anchor can move."""
        # Face far to the right.
        faces = [(900, 500, 50, 50)]
        cx_full, _ = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.0,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        cx_clamped, _ = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.0,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=0.1,
        )
        # Full shift: max_shift_x = (1000 - 500) * 0.5 * 1.0 = 250.
        # Clamped: max_shift_x = 250 * 0.1 = 25.
        self.assertLessEqual(abs(cx_clamped - 500), 25.0 + 0.01)
        self.assertGreater(abs(cx_full - 500), abs(cx_clamped - 500))

    def test_center_bias_and_face_strength_are_multiplicative(self):
        """center_bias_weight is applied before face_anchor_strength ((section)3.5).

        With face_anchor_strength=1.0 and center_bias_weight=0.1, the center
        bias still exerts influence - the anchor is NOT purely the face.
        """
        faces = [(700, 200, 100, 100)]
        cx, cy = self.node._compute_anchor(
            self.src_w, self.src_h, self.crop_w, self.crop_h, faces,
            face_anchor_strength=1.0, center_bias_weight=0.1,
            subject_top_bias=0.0, secondary_face_weight=0.0,
            max_anchor_shift_ratio=1.0,
        )
        # Face anchor x = 750. Center bias 0.1 pulls it 10% toward 500 -> 725.
        # With face_anchor_strength=1.0, crop center = 725 (not 750).
        self.assertNotAlmostEqual(cx, 750.0, delta=1.0)
        self.assertAlmostEqual(cx, 725.0, delta=1.0)


# --------------------------------------------------------------------------
# _detect_faces
# --------------------------------------------------------------------------

class TestDetectFaces(unittest.TestCase):
    """Haar cascade face detection wrapper."""

    def setUp(self):
        self.node = illumoraeImageFaceAspectCropNode()

    def test_blank_image_no_faces(self):
        """A uniform blank image should produce no face detections."""
        img = np.full((256, 256, 3), 0.5, dtype=np.float32)
        faces = self.node._detect_faces(
            img, detect_scale_factor=1.1, detect_min_neighbors=5,
            min_face_percent=3.0,
        )
        self.assertEqual(faces, [])

    def test_returns_list_of_tuples(self):
        img = np.full((128, 128, 3), 0.5, dtype=np.float32)
        faces = self.node._detect_faces(
            img, detect_scale_factor=1.1, detect_min_neighbors=5,
            min_face_percent=3.0,
        )
        self.assertIsInstance(faces, list)

    def test_cascade_load_or_none(self):
        """_get_face_cascade returns a classifier or None (if cv2 data missing)."""
        cascade = self.node._get_face_cascade()
        # On a normal OpenCV install this should be a valid classifier.
        # On a stripped install it may be None - both are acceptable.
        if cascade is not None:
            self.assertFalse(cascade.empty())

    def test_faces_sorted_by_area_descending(self):
        """If faces are detected, they must be sorted largest-first."""
        # We can't reliably synthesize a detectable face, so we test the
        # sort contract by monkeypatching the cascade with a stub.
        original = self.node._get_face_cascade

        class StubCascade:
            def empty(self):
                return False

            def detectMultiScale(self, *args, **kwargs):
                # Return faces in random area order.
                return np.array([
                    [10, 10, 20, 20],   # area 400
                    [0, 0, 50, 50],     # area 2500
                    [60, 60, 30, 30],   # area 900
                ], dtype=np.int32)

        self.node._face_cascade = StubCascade()
        try:
            img = np.full((256, 256, 3), 0.5, dtype=np.float32)
            faces = self.node._detect_faces(
                img, detect_scale_factor=1.1, detect_min_neighbors=5,
                min_face_percent=1.0,
            )
            areas = [w * h for (_, _, w, h) in faces]
            self.assertEqual(areas, sorted(areas, reverse=True))
            # Largest first.
            self.assertEqual(faces[0], (0, 0, 50, 50))
        finally:
            self.node._face_cascade = None
            self.node._get_face_cascade = original


# --------------------------------------------------------------------------
# _build_debug_image
# --------------------------------------------------------------------------

class TestBuildDebugImage(unittest.TestCase):
    """Debug overlay image construction."""

    def setUp(self):
        self.node = illumoraeImageFaceAspectCropNode()

    def test_shape_dtype_range(self):
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        debug = self.node._build_debug_image(
            img, crop_box=(10, 10, 50, 50), faces=[(20, 20, 20, 20)],
            target_w=100, target_h=100,
        )
        self.assertEqual(debug.shape, (100, 100, 3))
        self.assertEqual(debug.dtype, np.float32)
        self.assertGreaterEqual(float(debug.min()), 0.0)
        self.assertLessEqual(float(debug.max()), 1.0)

    def test_no_faces_does_not_crash(self):
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        debug = self.node._build_debug_image(
            img, crop_box=(10, 10, 50, 50), faces=[],
            target_w=100, target_h=100,
        )
        self.assertEqual(debug.shape, (100, 100, 3))

    def test_secondary_faces_drawn(self):
        """Multiple faces should not crash the debug overlay."""
        img = np.full((200, 200, 3), 0.5, dtype=np.float32)
        faces = [(20, 20, 40, 40), (100, 100, 30, 30), (150, 50, 20, 20)]
        debug = self.node._build_debug_image(
            img, crop_box=(0, 0, 100, 100), faces=faces,
            target_w=100, target_h=100,
        )
        self.assertEqual(debug.shape, (200, 200, 3))


# --------------------------------------------------------------------------
# crop_to_aspect (full pipeline)
# --------------------------------------------------------------------------

class TestCropToAspect(unittest.TestCase):
    """The main FUNCTION entry point."""

    def setUp(self):
        self.node = illumoraeImageFaceAspectCropNode()

    def _default_kwargs(self, **overrides):
        kw = dict(
            width=1024, height=576,
            face_anchor_strength=0.9, center_bias_weight=0.1,
            subject_top_bias=0.2, secondary_face_weight=0.15,
            max_anchor_shift_ratio=1.0, detect_scale_factor=1.1,
            detect_min_neighbors=5, min_face_percent=3.0,
            debug_prints=False,
        )
        kw.update(overrides)
        return kw

    def test_returns_two_tensors(self):
        img = _make_blank_image(500, 500)
        cropped, debug = self.node.crop_to_aspect(img, **self._default_kwargs())
        self.assertIsInstance(cropped, torch.Tensor)
        self.assertIsInstance(debug, torch.Tensor)

    def test_cropped_image_shape(self):
        img = _make_blank_image(1000, 1000)
        cropped, _ = self.node.crop_to_aspect(img, **self._default_kwargs(width=500, height=500))
        # 1000x1000 source, 1:1 target -> crop is 1000x1000 (full image).
        self.assertEqual(tuple(cropped.shape), (1, 1000, 1000, 3))

    def test_cropped_aspect_ratio_matches_target(self):
        img = _make_blank_image(1000, 500)
        cropped, _ = self.node.crop_to_aspect(img, **self._default_kwargs(width=512, height=512))
        _, h, w, _ = cropped.shape
        # Source is wider than 1:1, so crop_h = 500, crop_w = 500.
        self.assertEqual(h, 500)
        self.assertEqual(w, 500)

    def test_output_dtype_float32(self):
        img = _make_blank_image(100, 100)
        cropped, debug = self.node.crop_to_aspect(img, **self._default_kwargs(width=50, height=50))
        self.assertEqual(cropped.dtype, torch.float32)
        self.assertEqual(debug.dtype, torch.float32)

    def test_output_range_0_to_1(self):
        img = _make_blank_image(100, 100, value=0.7)
        cropped, debug = self.node.crop_to_aspect(img, **self._default_kwargs(width=50, height=50))
        self.assertGreaterEqual(float(cropped.min()), 0.0)
        self.assertLessEqual(float(cropped.max()), 1.0)
        self.assertGreaterEqual(float(debug.min()), 0.0)
        self.assertLessEqual(float(debug.max()), 1.0)

    def test_3d_input_unsqueezed_to_4d(self):
        """A 3D tensor (H, W, C) should be handled by unsqueezing to 4D."""
        img3d = torch.full((100, 100, 3), 0.5, dtype=torch.float32)
        cropped, debug = self.node.crop_to_aspect(img3d, **self._default_kwargs(width=50, height=50))
        self.assertEqual(cropped.dim(), 4)
        self.assertEqual(cropped.shape[0], 1)
        self.assertEqual(debug.dim(), 4)
        self.assertEqual(debug.shape[0], 1)

    def test_batch_processing(self):
        """A batch of N images should produce N cropped + N debug images."""
        img = torch.full((3, 100, 100, 3), 0.5, dtype=torch.float32)
        cropped, debug = self.node.crop_to_aspect(img, **self._default_kwargs(width=50, height=50))
        self.assertEqual(cropped.shape[0], 3)
        self.assertEqual(debug.shape[0], 3)

    def test_debug_image_always_produced(self):
        """The debug overlay is always built, regardless of debug_prints."""
        img = _make_blank_image(100, 100)
        cropped, debug = self.node.crop_to_aspect(img, **self._default_kwargs(debug_prints=False))
        self.assertEqual(tuple(debug.shape), (1, 100, 100, 3))
        # debug_prints=True should also produce a debug image.
        cropped2, debug2 = self.node.crop_to_aspect(img, **self._default_kwargs(debug_prints=True))
        self.assertEqual(tuple(debug2.shape), (1, 100, 100, 3))

    def test_no_faces_center_crop(self):
        """With no detectable faces, the crop should be centered."""
        img = _make_blank_image(1000, 500)
        cropped, _ = self.node.crop_to_aspect(img, **self._default_kwargs(width=500, height=500))
        # Source 1000x500, target 1:1 -> crop 500x500, centered at x=250.
        # Since the image is uniform, we verify shape only (content is identical
        # regardless of offset, but the crop box is centered).
        self.assertEqual(tuple(cropped.shape), (1, 500, 500, 3))

    def test_determinism_same_input(self):
        """Same input + params should produce identical output across calls."""
        img = _make_gradient_image(200, 200)
        kw = self._default_kwargs(width=100, height=100)
        c1, d1 = self.node.crop_to_aspect(img, **kw)
        c2, d2 = self.node.crop_to_aspect(img, **kw)
        self.assertTrue(torch.equal(c1, c2))
        self.assertTrue(torch.equal(d1, d2))

    def test_cropped_is_actual_subregion(self):
        """The cropped image should be a sub-region of the source, not resized."""
        # Create an image with a distinct left half (black) and right half (white).
        img = np.zeros((1, 100, 200, 3), dtype=np.float32)
        img[0, :, 100:] = 1.0
        img_t = torch.from_numpy(img)
        # Target 1:1 crop from a 200x100 source -> crop is 100x100.
        # With no faces, it centers at x=100, so crop spans x=50..150.
        cropped, _ = self.node.crop_to_aspect(
            img_t, **self._default_kwargs(width=100, height=100))
        # The crop should contain both black and white pixels (straddling x=100).
        c = cropped[0].numpy()
        self.assertLess(float(c.min()), 0.1)
        self.assertGreater(float(c.max()), 0.9)

    def test_subject_top_bias_accepted_as_kwarg(self):
        """crop_to_aspect must accept subject_top_bias (not top_bias_weight)."""
        img = _make_blank_image(100, 100)
        # This will raise TypeError if the parameter name is wrong.
        cropped, debug = self.node.crop_to_aspect(
            img, **self._default_kwargs(width=100, height=100, subject_top_bias=0.5))
        self.assertEqual(tuple(cropped.shape), (1, 100, 100, 3))


# --------------------------------------------------------------------------
# Frontmatter / docstring fields
# --------------------------------------------------------------------------

class TestFrontmatter(unittest.TestCase):
    """Module docstring Obsidian-style fields (per AGENTS.md)."""

    def setUp(self):
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        _PARENT = os.path.dirname(_THIS_DIR)
        with open(os.path.join(_PARENT, "image_face_aspect_crop.py"), "r") as f:
            self.docstring = f.read()
        # Extract the module docstring (between the first triple quotes).
        start = self.docstring.find('"""') + 3
        end = self.docstring.find('"""', start)
        self.fields_text = self.docstring[start:end]

    def _field(self, key):
        import re
        m = re.search(rf"^{key}::\s*(.+)$", self.fields_text, re.MULTILINE)
        return m.group(1).strip() if m else None

    def test_required_fields_present(self):
        for key in ("TITLE", "DESCRIPTIONSHORT", "VERSION"):
            self.assertIsNotNone(self._field(key), f"missing {key}::")

    def test_version_is_yyyymmdd(self):
        import re
        v = self._field("VERSION")
        self.assertIsNotNone(v)
        self.assertTrue(re.match(r"^\d{8}$", v), f"VERSION not YYYYMMDD: {v}")

    def test_status_working(self):
        self.assertEqual(self._field("STATUS"), "working")

    def test_group_image(self):
        self.assertEqual(self._field("GROUP"), "Image")


if __name__ == "__main__":
    unittest.main(verbosity=2)
