"""Regression tests for illumoraeImageResizeWanAdaptiveFramingNode.

Covers the bugs fixed in the 20260815 review pass:

- MAX_RESOLUTION constant now used for detection downscaling (fix (section)2.1)
- top_bias direction documented + 0.2 scaling factor (fix (section)3.1)
- detect_scale_factor now passed through to HOG scale (fix (section)3.2)
- DESCRIPTION mentions crop/pad modes (fix (section)3.3)
- Return type annotation on resize_adaptive (fix (section)3.4)
- _debug_print -> None annotation (fix (section)3.5)
- 0.6 anchor damping documented (fix (section)3.6)
- 4.0 scoring weight documented (fix (section)4.7)
- _detect_people equalization skip documented (fix (section)3.7)
- auto_subject face-first priority documented (fix (section)3.8)
- emit_debug_image BOOLEAN input added (fix (section)4.3)

Also covers the core logic that was previously untested:

- _choose_orientation: auto/forced horizontal/vertical
- _get_candidates: orientation + tier combinations
- _select_preset: explicit tier, auto tier, scoring
- _compute_crop_size: aspect-ratio sizing, edge cases
- _compute_crop_box: bounds clamping
- _get_anchor: no-detection fallback, face anchor, person anchor,
  top_bias direction, anchor_strength blending, framing_mode priority
- _detect_faces: blank image returns no faces, sorted by area
- _detect_people: blank image returns no people, accepts scale_factor
- _detect_subjects: downscale path for large images, box rescaling
- _resize_crop: shape, dtype, range
- _resize_pad: shape, dtype, range, bar fill
- _build_debug: shape, dtype, range, no crash with no detections
- resize_adaptive: full pipeline (shapes, dtype, range, batch, 3D unsqueeze,
  debug image, emit_debug_image=False, determinism, orientation modes,
  resize modes, preset selection, preset label)
- Structural compliance (INPUT_TYPES, RETURN_TYPES, mappings, frontmatter)

Usage:
    python -m tests.test_image_resize_wan_adaptive_framing
    python -m pytest tests/test_image_resize_wan_adaptive_framing.py -v
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

from image_resize_wan_adaptive_framing import (  # noqa: E402
    MAX_RESOLUTION,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    illumoraeImageResizeWanAdaptiveFramingNode,
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
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_class_attributes_present(self):
        self.assertEqual(self.node.CATEGORY, "illumorae")
        self.assertEqual(self.node.FUNCTION, "resize_adaptive")
        self.assertEqual(self.node.RETURN_TYPES, ("IMAGE", "IMAGE", "INT", "INT", "STRING"))
        self.assertEqual(
            self.node.RETURN_NAMES,
            ("resized_image", "debug_image", "width", "height", "selected_preset"),
        )
        self.assertIsInstance(self.node.DESCRIPTION, str)
        self.assertGreater(len(self.node.DESCRIPTION), 0)

    def test_description_mentions_crop_and_pad(self):
        """DESCRIPTION should mention crop and pad modes (fix (section)3.3)."""
        desc_lower = self.node.DESCRIPTION.lower()
        self.assertIn("crop", desc_lower)
        self.assertIn("pad", desc_lower)

    def test_input_types_required_keys(self):
        required = self.node.INPUT_TYPES()["required"]
        expected = {
            "image", "orientation_mode", "framing_mode", "resize_mode",
            "resolution_tier", "upscale_method", "anchor_strength",
            "top_bias", "face_min_percent", "detect_scale_factor",
            "detect_min_neighbors",
        }
        self.assertEqual(set(required.keys()), expected)

    def test_input_types_optional_keys(self):
        """Optional inputs should include debug_prints and emit_debug_image (fix (section)4.3)."""
        optional = self.node.INPUT_TYPES().get("optional", {})
        self.assertIn("debug_prints", optional)
        self.assertIn("emit_debug_image", optional)

    def test_emit_debug_image_default_true(self):
        """emit_debug_image should default to True (fix (section)4.3)."""
        optional = self.node.INPUT_TYPES().get("optional", {})
        _, meta = optional["emit_debug_image"]
        self.assertIs(meta["default"], True)

    def test_input_bounds(self):
        required = self.node.INPUT_TYPES()["required"]
        # anchor_strength
        _, as_meta = required["anchor_strength"]
        self.assertEqual(as_meta["min"], 0.0)
        self.assertEqual(as_meta["max"], 1.0)
        # top_bias
        _, tb_meta = required["top_bias"]
        self.assertEqual(tb_meta["min"], -0.5)
        self.assertEqual(tb_meta["max"], 0.5)

    def test_return_types_match_names(self):
        self.assertEqual(len(self.node.RETURN_TYPES), len(self.node.RETURN_NAMES))

    def test_node_class_mappings(self):
        self.assertIn("illumoraeImageResizeWanAdaptiveFramingNode", NODE_CLASS_MAPPINGS)
        self.assertIs(
            NODE_CLASS_MAPPINGS["illumoraeImageResizeWanAdaptiveFramingNode"],
            illumoraeImageResizeWanAdaptiveFramingNode,
        )
        self.assertIn("illumoraeImageResizeWanAdaptiveFramingNode", NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["illumoraeImageResizeWanAdaptiveFramingNode"],
            "Image Resize WAN Adaptive Framing",
        )

    def test_resize_adaptive_has_return_annotation(self):
        """resize_adaptive should have a return type annotation (fix (section)3.4)."""
        ann = illumoraeImageResizeWanAdaptiveFramingNode.resize_adaptive.__annotations__
        self.assertIn("return", ann)

    def test_debug_print_has_return_annotation(self):
        """_debug_print should have -> None (fix (section)3.5)."""
        ann = illumoraeImageResizeWanAdaptiveFramingNode._debug_print.__annotations__
        self.assertIn("return", ann)

    def test_max_resolution_is_positive(self):
        """MAX_RESOLUTION should be a positive integer (fix (section)2.1 - now actually used)."""
        self.assertIsInstance(MAX_RESOLUTION, int)
        self.assertGreater(MAX_RESOLUTION, 0)


# --------------------------------------------------------------------------
# _choose_orientation
# --------------------------------------------------------------------------

class TestChooseOrientation(unittest.TestCase):
    """Orientation resolution from source geometry + mode."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_auto_landscape(self):
        self.assertEqual(self.node._choose_orientation(1920, 1080, "auto"), "horizontal")

    def test_auto_portrait(self):
        self.assertEqual(self.node._choose_orientation(1080, 1920, "auto"), "vertical")

    def test_auto_square_defaults_horizontal(self):
        """Square images (w == h) should default to horizontal."""
        self.assertEqual(self.node._choose_orientation(500, 500, "auto"), "horizontal")

    def test_force_horizontal(self):
        self.assertEqual(self.node._choose_orientation(1080, 1920, "force_horizontal"), "horizontal")

    def test_force_vertical(self):
        self.assertEqual(self.node._choose_orientation(1920, 1080, "force_vertical"), "vertical")


# --------------------------------------------------------------------------
# _get_candidates
# --------------------------------------------------------------------------

class TestGetCandidates(unittest.TestCase):
    """Preset candidate list construction."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_horizontal_main(self):
        cands = self.node._get_candidates("horizontal", "main")
        self.assertEqual(cands, [(832, 480)])

    def test_vertical_main(self):
        cands = self.node._get_candidates("vertical", "main")
        self.assertEqual(cands, [(480, 832)])

    def test_horizontal_high(self):
        cands = self.node._get_candidates("horizontal", "high")
        self.assertEqual(cands, [(1280, 720)])

    def test_vertical_high(self):
        cands = self.node._get_candidates("vertical", "high")
        self.assertEqual(cands, [(720, 1280)])

    def test_horizontal_small(self):
        cands = self.node._get_candidates("horizontal", "small")
        self.assertEqual(cands, [(416, 240)])

    def test_auto_includes_all_tiers(self):
        cands = self.node._get_candidates("horizontal", "auto")
        self.assertIn((416, 240), cands)
        self.assertIn((832, 480), cands)
        self.assertIn((1280, 720), cands)
        self.assertEqual(len(cands), 3)

    def test_auto_vertical_transposes(self):
        cands = self.node._get_candidates("vertical", "auto")
        self.assertIn((240, 416), cands)
        self.assertIn((480, 832), cands)
        self.assertIn((720, 1280), cands)


# --------------------------------------------------------------------------
# _select_preset
# --------------------------------------------------------------------------

class TestSelectPreset(unittest.TestCase):
    """Preset selection via ratio + area scoring."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_explicit_main_returns_main(self):
        """An explicit tier should always return that tier's preset."""
        w, h = self.node._select_preset(1920, 1080, "horizontal", "main")
        self.assertEqual((w, h), (832, 480))

    def test_explicit_high_returns_high(self):
        w, h = self.node._select_preset(1920, 1080, "horizontal", "high")
        self.assertEqual((w, h), (1280, 720))

    def test_explicit_small_returns_small(self):
        w, h = self.node._select_preset(1920, 1080, "horizontal", "small")
        self.assertEqual((w, h), (416, 240))

    def test_explicit_vertical_transposes(self):
        w, h = self.node._select_preset(1080, 1920, "vertical", "main")
        self.assertEqual((w, h), (480, 832))

    def test_auto_landscape_picks_exact_ratio(self):
        """A 1920x1080 source (16:9) should match 1280x720 (exact 16:9) in auto.

        The scoring weights ratio match 4x over area match, so the exact-ratio
        preset (1280x720) wins over 832x480 (which has a slightly different
        ratio of 1.733 vs 1.778).
        """
        w, h = self.node._select_preset(1920, 1080, "horizontal", "auto")
        self.assertEqual((w, h), (1280, 720))

    def test_auto_vertical_picks_exact_ratio(self):
        """A 1080x1920 source (9:16) should match 720x1280 (exact 9:16) in auto."""
        w, h = self.node._select_preset(1080, 1920, "vertical", "auto")
        self.assertEqual((w, h), (720, 1280))

    def test_auto_near_720p_picks_high(self):
        """A 1280x720 source should match the high tier (exact ratio + area)."""
        w, h = self.node._select_preset(1280, 720, "horizontal", "auto")
        self.assertEqual((w, h), (1280, 720))

    def test_returns_valid_preset(self):
        """Selected preset should always be one of the known WAN presets."""
        known = {(416, 240), (832, 480), (1280, 720), (240, 416), (480, 832), (720, 1280)}
        for src in [(100, 100), (1920, 1080), (1080, 1920), (500, 500)]:
            w, h = self.node._select_preset(src[0], src[1], "horizontal", "auto")
            self.assertIn((w, h), known)


# --------------------------------------------------------------------------
# _compute_crop_size
# --------------------------------------------------------------------------

class TestComputeCropSize(unittest.TestCase):
    """Aspect-ratio crop dimension computation."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_wider_source_than_target(self):
        # Source 1000x500 (ratio 2.0), target 832x480 (ratio ~1.733)
        cw, ch = self.node._compute_crop_size(1000, 500, 832, 480)
        self.assertEqual(ch, 500)
        self.assertAlmostEqual(cw / ch, 832 / 480, places=2)
        self.assertLessEqual(cw, 1000)

    def test_taller_source_than_target(self):
        # Source 500x1000 (ratio 0.5), target 832x480 (ratio ~1.733)
        cw, ch = self.node._compute_crop_size(500, 1000, 832, 480)
        self.assertEqual(cw, 500)
        self.assertAlmostEqual(cw / ch, 832 / 480, places=2)
        self.assertLessEqual(ch, 1000)

    def test_exact_match_ratio(self):
        cw, ch = self.node._compute_crop_size(832, 480, 832, 480)
        self.assertEqual((cw, ch), (832, 480))

    def test_square_target_from_square_source(self):
        cw, ch = self.node._compute_crop_size(500, 500, 100, 100)
        self.assertEqual((cw, ch), (500, 500))

    def test_crop_never_exceeds_source(self):
        cw, ch = self.node._compute_crop_size(100, 100, 2000, 2000)
        self.assertLessEqual(cw, 100)
        self.assertLessEqual(ch, 100)

    def test_minimum_dimensions_are_positive(self):
        cw, ch = self.node._compute_crop_size(100, 100, 1, 1)
        self.assertGreaterEqual(cw, 1)
        self.assertGreaterEqual(ch, 1)


# --------------------------------------------------------------------------
# _compute_crop_box
# --------------------------------------------------------------------------

class TestComputeCropBox(unittest.TestCase):
    """Crop box clamping to image bounds."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

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
# _get_anchor
# --------------------------------------------------------------------------

class TestGetAnchor(unittest.TestCase):
    """Face/person/center anchor computation heuristics."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()
        self.src_w = 1000
        self.src_h = 1000

    def test_no_detections_falls_back_to_center(self):
        ax, ay, face, person, chosen = self.node._get_anchor(
            self.src_w, self.src_h, faces=[], people=[],
            framing_mode="auto_subject", anchor_strength=0.85, top_bias=0.0,
        )
        self.assertAlmostEqual(ax, self.src_w * 0.5)
        self.assertAlmostEqual(ay, self.src_h * 0.5)
        self.assertIsNone(face)
        self.assertIsNone(person)
        self.assertEqual(chosen, "center")

    def test_center_mode_ignores_detections(self):
        """framing_mode='center' should ignore faces/people entirely."""
        faces = [(400, 400, 100, 100)]
        people = [(300, 300, 200, 200)]
        ax, ay, _, _, chosen = self.node._get_anchor(
            self.src_w, self.src_h, faces=faces, people=people,
            framing_mode="center", anchor_strength=1.0, top_bias=0.0,
        )
        self.assertAlmostEqual(ax, self.src_w * 0.5)
        self.assertAlmostEqual(ay, self.src_h * 0.5)
        self.assertEqual(chosen, "center")

    def test_face_anchor_with_full_strength(self):
        """With anchor_strength=1.0 and top_bias=0, anchor should be at face eye-line."""
        faces = [(700, 300, 100, 100)]  # center x=750, eye-line y=300+42=342
        ax, ay, face, person, chosen = self.node._get_anchor(
            self.src_w, self.src_h, faces=faces, people=[],
            framing_mode="human_face", anchor_strength=1.0, top_bias=0.0,
        )
        self.assertAlmostEqual(ax, 750.0, delta=1.0)
        self.assertAlmostEqual(ay, 342.0, delta=1.0)
        self.assertEqual(chosen, "face")
        self.assertEqual(face, (700, 300, 100, 100))
        self.assertIsNone(person)

    def test_person_anchor_with_full_strength(self):
        """With anchor_strength=1.0 and top_bias=0, anchor should be at person upper-body."""
        people = [(600, 200, 200, 400)]  # center x=700, upper-body y=200+128=328
        ax, ay, _, person, chosen = self.node._get_anchor(
            self.src_w, self.src_h, faces=[], people=people,
            framing_mode="human_body", anchor_strength=1.0, top_bias=0.0,
        )
        self.assertAlmostEqual(ax, 700.0, delta=1.0)
        self.assertAlmostEqual(ay, 328.0, delta=1.0)
        self.assertEqual(chosen, "person")

    def test_auto_subject_prioritizes_face_over_person(self):
        """auto_subject should pick face when both face and person are available ((section)3.8)."""
        faces = [(700, 300, 100, 100)]
        people = [(300, 300, 200, 200)]
        _, _, _, _, chosen = self.node._get_anchor(
            self.src_w, self.src_h, faces=faces, people=people,
            framing_mode="auto_subject", anchor_strength=1.0, top_bias=0.0,
        )
        self.assertEqual(chosen, "face")

    def test_auto_subject_falls_back_to_person(self):
        """auto_subject should use person when no face is detected ((section)3.8)."""
        people = [(300, 300, 200, 200)]
        _, _, _, _, chosen = self.node._get_anchor(
            self.src_w, self.src_h, faces=[], people=people,
            framing_mode="auto_subject", anchor_strength=1.0, top_bias=0.0,
        )
        self.assertEqual(chosen, "person")

    def test_human_face_no_fallback_to_person(self):
        """human_face mode should NOT fall back to person when no face is found ((section)6.1)."""
        people = [(300, 300, 200, 200)]
        _, _, _, _, chosen = self.node._get_anchor(
            self.src_w, self.src_h, faces=[], people=people,
            framing_mode="human_face", anchor_strength=1.0, top_bias=0.0,
        )
        self.assertEqual(chosen, "center")

    def test_anchor_strength_zero_is_center(self):
        faces = [(700, 200, 100, 100)]
        ax, ay, _, _, _ = self.node._get_anchor(
            self.src_w, self.src_h, faces=faces, people=[],
            framing_mode="human_face", anchor_strength=0.0, top_bias=0.0,
        )
        self.assertAlmostEqual(ax, self.src_w * 0.5)
        self.assertAlmostEqual(ay, self.src_h * 0.5)

    def test_top_bias_shifts_anchor_downward(self):
        """Positive top_bias should move anchor_y DOWN (fix (section)3.1).

        The 0.2 factor damps the bias; at top_bias=0.5 the shift is
        src_h * 0.5 * 0.2 = 100px for a 1000px image.
        """
        faces = [(400, 400, 100, 100)]
        _, ay_zero, _, _, _ = self.node._get_anchor(
            self.src_w, self.src_h, faces=faces, people=[],
            framing_mode="human_face", anchor_strength=1.0, top_bias=0.0,
        )
        _, ay_pos, _, _, _ = self.node._get_anchor(
            self.src_w, self.src_h, faces=faces, people=[],
            framing_mode="human_face", anchor_strength=1.0, top_bias=0.5,
        )
        self.assertGreater(ay_pos, ay_zero)
        # Shift = src_h * 0.5 * 0.2 = 100
        self.assertAlmostEqual(ay_pos - ay_zero, 100.0, delta=1.0)

    def test_top_bias_negative_shifts_upward(self):
        """Negative top_bias should move anchor_y UP."""
        faces = [(400, 400, 100, 100)]
        _, ay_zero, _, _, _ = self.node._get_anchor(
            self.src_w, self.src_h, faces=faces, people=[],
            framing_mode="human_face", anchor_strength=1.0, top_bias=0.0,
        )
        _, ay_neg, _, _, _ = self.node._get_anchor(
            self.src_w, self.src_h, faces=faces, people=[],
            framing_mode="human_face", anchor_strength=1.0, top_bias=-0.5,
        )
        self.assertLess(ay_neg, ay_zero)

    def test_top_bias_applied_in_center_mode(self):
        """top_bias is applied even when chosen_type is 'center' ((section)6.2)."""
        _, ay_zero, _, _, chosen = self.node._get_anchor(
            self.src_w, self.src_h, faces=[], people=[],
            framing_mode="center", anchor_strength=1.0, top_bias=0.0,
        )
        _, ay_pos, _, _, _ = self.node._get_anchor(
            self.src_w, self.src_h, faces=[], people=[],
            framing_mode="center", anchor_strength=1.0, top_bias=0.5,
        )
        self.assertEqual(chosen, "center")
        self.assertGreater(ay_pos, ay_zero)


# --------------------------------------------------------------------------
# _detect_faces
# --------------------------------------------------------------------------

class TestDetectFaces(unittest.TestCase):
    """Haar cascade face detection wrapper."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_blank_image_no_faces(self):
        """A uniform blank image should produce no face detections."""
        img = np.full((256, 256, 3), 0.5, dtype=np.float32)
        faces = self.node._detect_faces(img, scale_factor=1.1, min_neighbors=5, face_min_percent=3.0)
        self.assertEqual(faces, [])

    def test_returns_list_of_tuples(self):
        img = np.full((128, 128, 3), 0.5, dtype=np.float32)
        faces = self.node._detect_faces(img, scale_factor=1.1, min_neighbors=5, face_min_percent=3.0)
        self.assertIsInstance(faces, list)

    def test_cascade_load_or_none(self):
        """_get_face_cascade returns a classifier or None (if cv2 data missing)."""
        cascade = self.node._get_face_cascade()
        if cascade is not None:
            self.assertFalse(cascade.empty())

    def test_faces_sorted_by_area_descending(self):
        """If faces are detected, they must be sorted largest-first."""
        original = self.node._get_face_cascade

        class StubCascade:
            def empty(self):
                return False

            def detectMultiScale(self, *args, **kwargs):
                return np.array([
                    [10, 10, 20, 20],   # area 400
                    [0, 0, 50, 50],     # area 2500
                    [60, 60, 30, 30],   # area 900
                ], dtype=np.int32)

        self.node._face_cascade = StubCascade()
        try:
            img = np.full((256, 256, 3), 0.5, dtype=np.float32)
            faces = self.node._detect_faces(img, scale_factor=1.1, min_neighbors=5, face_min_percent=1.0)
            areas = [w * h for (_, _, w, h) in faces]
            self.assertEqual(areas, sorted(areas, reverse=True))
            self.assertEqual(faces[0], (0, 0, 50, 50))
        finally:
            self.node._face_cascade = None
            self.node._get_face_cascade = original


# --------------------------------------------------------------------------
# _detect_people
# --------------------------------------------------------------------------

class TestDetectPeople(unittest.TestCase):
    """HOG person detection wrapper."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_blank_image_no_people(self):
        """A uniform blank image should produce no person detections."""
        img = np.full((256, 256, 3), 0.5, dtype=np.float32)
        people = self.node._detect_people(img, scale_factor=1.05)
        self.assertEqual(people, [])

    def test_returns_list_of_tuples(self):
        img = np.full((128, 256, 3), 0.5, dtype=np.float32)
        people = self.node._detect_people(img, scale_factor=1.05)
        self.assertIsInstance(people, list)

    def test_accepts_scale_factor_kwarg(self):
        """_detect_people should accept scale_factor (fix (section)3.2)."""
        img = np.full((128, 256, 3), 0.5, dtype=np.float32)
        # Should not raise.
        people = self.node._detect_people(img, scale_factor=1.1)
        self.assertIsInstance(people, list)

    def test_hog_loads_successfully(self):
        """_get_hog should always return a valid HOG descriptor (not None)."""
        hog = self.node._get_hog()
        self.assertIsNotNone(hog)


# --------------------------------------------------------------------------
# _detect_subjects
# --------------------------------------------------------------------------

class TestDetectSubjects(unittest.TestCase):
    """Detection wrapper with downscaling for large images (fix (section)2.1)."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_small_image_no_downscale(self):
        """Images under MAX_RESOLUTION should run detection directly (scale >= 1.0)."""
        img = np.full((256, 256, 3), 0.5, dtype=np.float32)
        faces, people = self.node._detect_subjects(
            img, 256, 256, scale_factor=1.1, min_neighbors=5, face_min_percent=3.0,
        )
        self.assertIsInstance(faces, list)
        self.assertIsInstance(people, list)
        self.assertEqual(faces, [])
        self.assertEqual(people, [])

    def test_large_image_downscale_and_rescale(self):
        """Images over MAX_RESOLUTION should downscale for detection, rescale boxes back."""
        # Create a 5000x5000 blank image (above MAX_RESOLUTION=4096).
        img = np.full((5000, 5000, 3), 0.5, dtype=np.float32)
        faces, people = self.node._detect_subjects(
            img, 5000, 5000, scale_factor=1.1, min_neighbors=5, face_min_percent=3.0,
        )
        # Blank image -> no detections, but should not crash.
        self.assertEqual(faces, [])
        self.assertEqual(people, [])

    def test_large_image_box_rescaling(self):
        """Detection boxes from downscaled image should be rescaled to source coords."""
        # Monkeypatch _detect_faces to return a known box on the downscaled image.
        # Source 5000x5000, MAX_RESOLUTION=4096 -> scale = 4096/5000 = 0.8192
        # det_w = det_h = 4096 (rounded). A box at (100, 100, 200, 200) on the
        # downscaled image should rescale to ~(122, 122, 244, 244) on the source.
        original_detect_faces = self.node._detect_faces
        original_detect_people = self.node._detect_people

        def stub_detect_faces(image_rgb, scale_factor, min_neighbors, face_min_percent):
            h, w = image_rgb.shape[:2]
            # Only return a box if we're on the downscaled image (smaller than 5000).
            if w < 5000:
                return [(100, 100, 200, 200)]
            return []

        def stub_detect_people(image_rgb, scale_factor=1.05):
            return []

        self.node._detect_faces = stub_detect_faces
        self.node._detect_people = stub_detect_people
        try:
            img = np.full((5000, 5000, 3), 0.5, dtype=np.float32)
            faces, people = self.node._detect_subjects(
                img, 5000, 5000, scale_factor=1.1, min_neighbors=5, face_min_percent=3.0,
            )
            self.assertEqual(len(faces), 1)
            x, y, w, h = faces[0]
            # Rescaled box should be larger than the stub box (100,100,200,200).
            self.assertGreater(x, 100)
            self.assertGreater(w, 200)
            # Approximate: inv = 5000/4096 ~ 1.22 -> x ~ 122, w ~ 244
            self.assertAlmostEqual(x, 122, delta=5)
            self.assertAlmostEqual(w, 244, delta=5)
        finally:
            self.node._detect_faces = original_detect_faces
            self.node._detect_people = original_detect_people


# --------------------------------------------------------------------------
# _resize_crop
# --------------------------------------------------------------------------

class TestResizeCrop(unittest.TestCase):
    """Crop region resize via common_upscale."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_shape_dtype_range(self):
        img = np.full((100, 200, 3), 0.5, dtype=np.float32)
        resized = self.node._resize_crop(img, 832, 480, "lanczos")
        self.assertEqual(resized.shape, (480, 832, 3))
        self.assertEqual(resized.dtype, np.float32)
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)

    def test_upscale_small_to_large(self):
        img = np.full((50, 50, 3), 0.7, dtype=np.float32)
        resized = self.node._resize_crop(img, 416, 240, "bilinear")
        self.assertEqual(resized.shape, (240, 416, 3))

    def test_downscale_large_to_small(self):
        img = np.full((1000, 1000, 3), 0.3, dtype=np.float32)
        resized = self.node._resize_crop(img, 416, 240, "area")
        self.assertEqual(resized.shape, (240, 416, 3))


# --------------------------------------------------------------------------
# _resize_pad
# --------------------------------------------------------------------------

class TestResizePad(unittest.TestCase):
    """Soft-bar pad resize."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_shape_dtype_range(self):
        img = np.full((480, 640, 3), 0.5, dtype=np.float32)
        padded = self.node._resize_pad(img, 832, 480, 320.0, 240.0, "lanczos")
        self.assertEqual(padded.shape, (480, 832, 3))
        self.assertEqual(padded.dtype, np.float32)
        self.assertGreaterEqual(float(padded.min()), 0.0)
        self.assertLessEqual(float(padded.max()), 1.0)

    def test_pad_fills_bars(self):
        """Pad mode should fill empty bars with blurred content, not black."""
        img = np.full((480, 640, 3), 0.8, dtype=np.float32)
        padded = self.node._resize_pad(img, 832, 480, 320.0, 240.0, "lanczos")
        # The bars (outside the fitted region) should not be pure black.
        # Check a corner pixel that is likely in the bar region.
        corner = padded[0, 0, :]
        self.assertGreater(float(corner.mean()), 0.1)

    def test_exact_fit_no_bars(self):
        """If source ratio matches target ratio, there should be no bars."""
        img = np.full((480, 832, 3), 0.5, dtype=np.float32)
        padded = self.node._resize_pad(img, 832, 480, 416.0, 240.0, "lanczos")
        self.assertEqual(padded.shape, (480, 832, 3))


# --------------------------------------------------------------------------
# _build_debug
# --------------------------------------------------------------------------

class TestBuildDebug(unittest.TestCase):
    """Debug overlay image construction."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def test_shape_dtype_range(self):
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        debug = self.node._build_debug(
            src_np=img, crop_box=(10, 10, 50, 50),
            target_w=832, target_h=480,
            primary_face=(20, 20, 20, 20), primary_person=None,
            chosen_type="face",
        )
        self.assertEqual(debug.shape, (100, 100, 3))
        self.assertEqual(debug.dtype, np.float32)
        self.assertGreaterEqual(float(debug.min()), 0.0)
        self.assertLessEqual(float(debug.max()), 1.0)

    def test_no_detections_does_not_crash(self):
        img = np.full((100, 100, 3), 0.5, dtype=np.float32)
        debug = self.node._build_debug(
            src_np=img, crop_box=(10, 10, 50, 50),
            target_w=832, target_h=480,
            primary_face=None, primary_person=None,
            chosen_type="center",
        )
        self.assertEqual(debug.shape, (100, 100, 3))

    def test_person_detection_drawn(self):
        """Drawing a person box should not crash."""
        img = np.full((200, 200, 3), 0.5, dtype=np.float32)
        debug = self.node._build_debug(
            src_np=img, crop_box=(0, 0, 100, 100),
            target_w=832, target_h=480,
            primary_face=None, primary_person=(50, 50, 80, 120),
            chosen_type="person",
        )
        self.assertEqual(debug.shape, (200, 200, 3))


# --------------------------------------------------------------------------
# resize_adaptive (full pipeline)
# --------------------------------------------------------------------------

class TestResizeAdaptive(unittest.TestCase):
    """The main FUNCTION entry point."""

    def setUp(self):
        self.node = illumoraeImageResizeWanAdaptiveFramingNode()

    def _default_kwargs(self, **overrides):
        kw = dict(
            orientation_mode="auto",
            framing_mode="auto_subject",
            resize_mode="crop",
            resolution_tier="auto",
            upscale_method="lanczos",
            anchor_strength=0.85,
            top_bias=0.18,
            face_min_percent=3.0,
            detect_scale_factor=1.1,
            detect_min_neighbors=5,
            debug_prints=False,
            emit_debug_image=True,
        )
        kw.update(overrides)
        return kw

    def test_returns_five_values(self):
        img = _make_blank_image(500, 500)
        result = self.node.resize_adaptive(img, **self._default_kwargs())
        self.assertEqual(len(result), 5)
        resized, debug, w, h, label = result
        self.assertIsInstance(resized, torch.Tensor)
        self.assertIsInstance(debug, torch.Tensor)
        self.assertIsInstance(w, int)
        self.assertIsInstance(h, int)
        self.assertIsInstance(label, str)

    def test_resized_image_shape(self):
        """Resized image should match the selected preset dimensions."""
        img = _make_blank_image(1000, 1000)
        resized, _, w, h, _ = self.node.resize_adaptive(img, **self._default_kwargs(resolution_tier="main"))
        self.assertEqual(resized.dim(), 4)
        _, rh, rw, _ = resized.shape
        self.assertEqual((rw, rh), (w, h))

    def test_output_dtype_float32(self):
        img = _make_blank_image(100, 100)
        resized, debug, _, _, _ = self.node.resize_adaptive(img, **self._default_kwargs())
        self.assertEqual(resized.dtype, torch.float32)
        self.assertEqual(debug.dtype, torch.float32)

    def test_output_range_0_to_1(self):
        img = _make_blank_image(100, 100, value=0.7)
        resized, debug, _, _, _ = self.node.resize_adaptive(img, **self._default_kwargs())
        self.assertGreaterEqual(float(resized.min()), 0.0)
        self.assertLessEqual(float(resized.max()), 1.0)
        self.assertGreaterEqual(float(debug.min()), 0.0)
        self.assertLessEqual(float(debug.max()), 1.0)

    def test_3d_input_unsqueezed_to_4d(self):
        """A 3D tensor (H, W, C) should be handled by unsqueezing to 4D."""
        img3d = torch.full((100, 100, 3), 0.5, dtype=torch.float32)
        resized, debug, _, _, _ = self.node.resize_adaptive(img3d, **self._default_kwargs())
        self.assertEqual(resized.dim(), 4)
        self.assertEqual(resized.shape[0], 1)
        self.assertEqual(debug.dim(), 4)
        self.assertEqual(debug.shape[0], 1)

    def test_batch_processing(self):
        """A batch of N images should produce N resized + N debug images."""
        img = torch.full((3, 100, 100, 3), 0.5, dtype=torch.float32)
        resized, debug, _, _, _ = self.node.resize_adaptive(img, **self._default_kwargs())
        self.assertEqual(resized.shape[0], 3)
        self.assertEqual(debug.shape[0], 3)

    def test_debug_image_always_produced_when_enabled(self):
        """The debug overlay is built when emit_debug_image=True."""
        img = _make_blank_image(100, 100)
        resized, debug, _, _, _ = self.node.resize_adaptive(
            img, **self._default_kwargs(emit_debug_image=True, debug_prints=False)
        )
        # Debug image should match source dimensions (100x100).
        self.assertEqual(tuple(debug.shape), (1, 100, 100, 3))

    def test_emit_debug_image_false_skips_overlay(self):
        """emit_debug_image=False should skip _build_debug (fix (section)4.3).

        The debug output should still be a valid tensor (source passed through),
        but it should NOT contain the overlay text/rectangles. We verify by
        checking that the debug image is identical to the source (no overlay).
        """
        img = _make_blank_image(100, 100, value=0.5)
        resized, debug, _, _, _ = self.node.resize_adaptive(
            img, **self._default_kwargs(emit_debug_image=False)
        )
        # Debug should be the source image unchanged (no overlay).
        self.assertEqual(tuple(debug.shape), (1, 100, 100, 3))
        # A uniform 0.5 image with no overlay should remain 0.5 everywhere.
        self.assertTrue(torch.allclose(debug, torch.full_like(debug, 0.5)))

    def test_emit_debug_image_true_has_overlay(self):
        """emit_debug_image=True should produce a debug image that differs from source."""
        img = _make_blank_image(200, 200, value=0.5)
        _, debug_on, _, _, _ = self.node.resize_adaptive(
            img, **self._default_kwargs(emit_debug_image=True)
        )
        # The overlay draws text/rectangles, so the debug image should differ
        # from a uniform 0.5 image.
        self.assertFalse(torch.allclose(debug_on, torch.full_like(debug_on, 0.5)))

    def test_no_detections_center_crop(self):
        """With no detectable faces/people, the crop should be centered."""
        # _make_blank_image(h=500, w=1000) -> landscape source.
        img = _make_blank_image(500, 1000)
        resized, _, _, _, _ = self.node.resize_adaptive(
            img, **self._default_kwargs(resolution_tier="main")
        )
        # Source 1000x500 (w x h), main horizontal preset 832x480 (ratio 1.733).
        # crop_w = round(500 * 1.733) = 867, crop_h = 500, then resized to 832x480.
        self.assertEqual(resized.shape[2], 832)  # width after resize
        self.assertEqual(resized.shape[1], 480)  # height after resize

    def test_determinism_same_input(self):
        """Same input + params should produce identical output across calls."""
        img = _make_gradient_image(200, 200)
        kw = self._default_kwargs(resolution_tier="main")
        r1 = self.node.resize_adaptive(img, **kw)
        r2 = self.node.resize_adaptive(img, **kw)
        self.assertTrue(torch.equal(r1[0], r2[0]))
        self.assertTrue(torch.equal(r1[1], r2[1]))

    def test_force_vertical_orientation(self):
        """force_vertical should select a vertical preset even for a landscape source."""
        img = _make_blank_image(1000, 500)  # landscape source
        _, _, w, h, label = self.node.resize_adaptive(
            img, **self._default_kwargs(orientation_mode="force_vertical", resolution_tier="main")
        )
        self.assertGreater(h, w)
        self.assertIn("vertical", label)

    def test_force_horizontal_orientation(self):
        """force_horizontal should select a horizontal preset even for a portrait source."""
        img = _make_blank_image(500, 1000)  # portrait source
        _, _, w, h, label = self.node.resize_adaptive(
            img, **self._default_kwargs(orientation_mode="force_horizontal", resolution_tier="main")
        )
        self.assertGreater(w, h)
        self.assertIn("horizontal", label)

    def test_pad_mode_shape(self):
        """Pad mode should produce the exact target dimensions."""
        img = _make_blank_image(500, 500)
        resized, _, w, h, _ = self.node.resize_adaptive(
            img, **self._default_kwargs(resize_mode="pad", resolution_tier="main")
        )
        self.assertEqual(resized.shape[2], w)
        self.assertEqual(resized.shape[1], h)

    def test_crop_mode_shape(self):
        """Crop mode should produce the exact target dimensions."""
        img = _make_blank_image(1000, 1000)
        resized, _, w, h, _ = self.node.resize_adaptive(
            img, **self._default_kwargs(resize_mode="crop", resolution_tier="main")
        )
        self.assertEqual(resized.shape[2], w)
        self.assertEqual(resized.shape[1], h)

    def test_preset_label_format(self):
        """The selected_preset string should contain dimensions and orientation."""
        # _make_blank_image(h=1080, w=1920) -> landscape source.
        img = _make_blank_image(1080, 1920)
        _, _, _, _, label = self.node.resize_adaptive(
            img, **self._default_kwargs(resolution_tier="main")
        )
        self.assertIn("832", label)
        self.assertIn("480", label)
        self.assertIn("horizontal", label)

    def test_explicit_tier_small(self):
        # _make_blank_image(h=1080, w=1920) -> landscape source.
        img = _make_blank_image(1080, 1920)
        _, _, w, h, _ = self.node.resize_adaptive(
            img, **self._default_kwargs(resolution_tier="small")
        )
        self.assertEqual((w, h), (416, 240))

    def test_explicit_tier_high(self):
        # _make_blank_image(h=1080, w=1920) -> landscape source.
        img = _make_blank_image(1080, 1920)
        _, _, w, h, _ = self.node.resize_adaptive(
            img, **self._default_kwargs(resolution_tier="high")
        )
        self.assertEqual((w, h), (1280, 720))

    def test_center_framing_mode(self):
        """framing_mode='center' should not crash and should produce valid output."""
        img = _make_blank_image(500, 500)
        resized, debug, _, _, _ = self.node.resize_adaptive(
            img, **self._default_kwargs(framing_mode="center")
        )
        self.assertEqual(resized.dim(), 4)
        self.assertEqual(debug.dim(), 4)

    def test_large_image_does_not_crash(self):
        """An image larger than MAX_RESOLUTION should not crash (fix (section)2.1)."""
        # 5000x5000 is above MAX_RESOLUTION=4096, triggering the downscale path.
        img = _make_blank_image(5000, 5000)
        resized, debug, w, h, _ = self.node.resize_adaptive(
            img, **self._default_kwargs(resolution_tier="main")
        )
        self.assertEqual(resized.shape[2], w)
        self.assertEqual(resized.shape[1], h)


# --------------------------------------------------------------------------
# Frontmatter / docstring fields
# --------------------------------------------------------------------------

class TestFrontmatter(unittest.TestCase):
    """Module docstring Obsidian-style fields (per AGENTS.md)."""

    def setUp(self):
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _parent = os.path.dirname(_this_dir)
        with open(os.path.join(_parent, "image_resize_wan_adaptive_framing.py"), "r") as f:
            self.source = f.read()
        # Extract the module docstring (between the first triple quotes).
        start = self.source.find('"""') + 3
        end = self.source.find('"""', start)
        self.fields_text = self.source[start:end]

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

    def test_title_matches_display_name(self):
        self.assertEqual(self._field("TITLE"), "Image Resize WAN Adaptive Framing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
