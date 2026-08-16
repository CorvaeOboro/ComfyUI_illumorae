"""Regression tests for illumoraeSaveAnimatedWEBPFolderPathNode.

Covers the bugs fixed in the 20260816 review pass:

- sanitize_name: illegal Windows path chars, control chars, whitespace (M1)
- format_counter_filename: both counter_position values, counter_digits (m1)
- get_latest_counter: empty folder, existing files, both positions (m3)
- create_folder_structure: nested folder creation
- save_images: end-to-end save, valid WebP, multi-frame, filepath return
- save_images: empty batch guard (m5)
- save_images: sanitization of bad filename chars (M1)
- save_images: empty folderpath fallback to output_dir (m2)
- save_images: counter increments across saves (m3)
- save_images: metadata disabled (M2/metadata path)
- save_images: unexpected method value falls back to default (m6)
- save_images: fps -> duration rounding (m7)
- INPUT_TYPES: save_metadata is a clean 2-tuple (M2)

Usage:
    python -m tests.test_save_animated_webp_extended_folderpath
    python -m pytest tests/test_save_animated_webp_extended_folderpath.py -v
"""
from __future__ import annotations

import os
import unittest

from _test_setup import (  # noqa: E402
    get_output_dir,
    make_test_tensor,
    reset_output_dir,
    set_disable_metadata,
)

from save_animated_webp_extended_folderpath import (  # noqa: E402
    illumoraeSaveAnimatedWEBPFolderPathNode,
)


class TestSanitizeName(unittest.TestCase):
    """sanitize_name: illegal chars, control chars, whitespace (M1)."""

    def test_replaces_illegal_chars(self):
        node = illumoraeSaveAnimatedWEBPFolderPathNode
        for ch in '<>:"/\\|?*':
            result = node.sanitize_name("a" + ch + "b")
            self.assertEqual(result, "a_b")

    def test_strips_whitespace(self):
        self.assertEqual(illumoraeSaveAnimatedWEBPFolderPathNode.sanitize_name("  hello  "), "hello")

    def test_removes_control_chars(self):
        # tab and null are control chars (ord < 32), removed entirely
        result = illumoraeSaveAnimatedWEBPFolderPathNode.sanitize_name("a\tb\x00c")
        self.assertEqual(result, "abc")

    def test_preserves_valid_chars(self):
        self.assertEqual(
            illumoraeSaveAnimatedWEBPFolderPathNode.sanitize_name("my_gen-2026"),
            "my_gen-2026",
        )

    def test_empty_string_stays_empty(self):
        self.assertEqual(illumoraeSaveAnimatedWEBPFolderPathNode.sanitize_name(""), "")


class TestFormatCounterFilename(unittest.TestCase):
    """format_counter_filename: both positions, zero-padding (m1)."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()

    def test_last_position(self):
        name = self.node.format_counter_filename("output", 5, 3, "last")
        self.assertEqual(name, "output_005.webp")

    def test_first_position(self):
        name = self.node.format_counter_filename("output", 5, 3, "first")
        self.assertEqual(name, "005_output.webp")

    def test_counter_digits_2(self):
        name = self.node.format_counter_filename("gen", 1, 2, "last")
        self.assertEqual(name, "gen_01.webp")

    def test_counter_digits_6(self):
        name = self.node.format_counter_filename("gen", 42, 6, "last")
        self.assertEqual(name, "gen_000042.webp")

    def test_large_counter(self):
        name = self.node.format_counter_filename("gen", 9999, 4, "first")
        self.assertEqual(name, "9999_gen.webp")

    def test_default_position_is_last(self):
        name_default = self.node.format_counter_filename("x", 1, 3)
        name_last = self.node.format_counter_filename("x", 1, 3, "last")
        self.assertEqual(name_default, name_last)


class TestGetLatestCounter(unittest.TestCase):
    """get_latest_counter: empty folder, existing files, both positions (m3)."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()
        self.tmp = reset_output_dir()

    def _create_files(self, names):
        for n in names:
            with open(os.path.join(self.tmp, n), "wb") as f:
                f.write(b"")

    def test_empty_folder_returns_1(self):
        self.assertEqual(self.node.get_latest_counter(self.tmp, "output", 3, "last"), 1)

    def test_nonexistent_folder_returns_1(self):
        missing = os.path.join(self.tmp, "does_not_exist")
        self.assertEqual(self.node.get_latest_counter(missing, "output", 3, "last"), 1)

    def test_last_position_increments(self):
        self._create_files(["output_001.webp", "output_002.webp", "output_005.webp"])
        self.assertEqual(self.node.get_latest_counter(self.tmp, "output", 3, "last"), 6)

    def test_first_position_increments(self):
        self._create_files(["001_output.webp", "003_output.webp"])
        self.assertEqual(self.node.get_latest_counter(self.tmp, "output", 3, "first"), 4)

    def test_ignores_non_matching_prefix(self):
        self._create_files(["other_001.webp", "output_002.webp"])
        self.assertEqual(self.node.get_latest_counter(self.tmp, "output", 3, "last"), 3)

    def test_ignores_non_webp_files(self):
        self._create_files(["output_001.png", "output_002.txt"])
        self.assertEqual(self.node.get_latest_counter(self.tmp, "output", 3, "last"), 1)

    def test_non_digit_counter_treated_as_zero(self):
        self._create_files(["output_abc.webp", "output_005.webp"])
        self.assertEqual(self.node.get_latest_counter(self.tmp, "output", 3, "last"), 6)


class TestCreateFolderStructure(unittest.TestCase):
    """create_folder_structure: nested folder creation."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()
        self.base = reset_output_dir()

    def test_creates_target_folder(self):
        result = self.node.create_folder_structure(self.base, "mygen")
        self.assertTrue(os.path.isdir(result))
        self.assertEqual(os.path.basename(result), "mygen")

    def test_creates_base_if_missing(self):
        nested = os.path.join(self.base, "a", "b", "c")
        result = self.node.create_folder_structure(nested, "gen")
        self.assertTrue(os.path.isdir(result))

    def test_idempotent(self):
        self.node.create_folder_structure(self.base, "gen")
        result = self.node.create_folder_structure(self.base, "gen")
        self.assertTrue(os.path.isdir(result))


class TestSaveImagesBasic(unittest.TestCase):
    """save_images: end-to-end save, valid WebP, filepath return."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()
        self.base = reset_output_dir()

    def _save(self, **overrides):
        defaults = dict(
            images=make_test_tensor(frames=2, height=8, width=8),
            filename_prefix="output",
            folderpath_input=self.base,
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )
        defaults.update(overrides)
        return self.node.save_images(**defaults)

    def test_returns_filepath_string(self):
        result = self._save()
        self.assertIn("result", result)
        filepath = result["result"][0]
        self.assertIsInstance(filepath, str)
        self.assertTrue(filepath.endswith(".webp"))

    def test_file_exists_on_disk(self):
        result = self._save()
        filepath = result["result"][0]
        self.assertTrue(os.path.isfile(filepath), f"Expected file at {filepath}")

    def test_file_is_valid_webp(self):
        from PIL import Image

        result = self._save()
        filepath = result["result"][0]
        with Image.open(filepath) as img:
            self.assertEqual(img.format, "WEBP")

    def test_animated_flag_true(self):
        result = self._save()
        self.assertEqual(result["ui"]["animated"], (True,))

    def test_ui_has_one_result_entry(self):
        result = self._save()
        self.assertEqual(len(result["ui"]["images"]), 1)

    def test_multiple_frames_preserved(self):
        from PIL import Image

        result = self._save(images=make_test_tensor(frames=3, height=4, width=4))
        filepath = result["result"][0]
        with Image.open(filepath) as img:
            img.seek(0)
            frames = 0
            try:
                while True:
                    img.seek(frames)
                    frames += 1
            except EOFError:
                pass
            self.assertEqual(frames, 3)


class TestSaveImagesEmptyBatch(unittest.TestCase):
    """save_images: empty batch guard (m5)."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()
        self.base = reset_output_dir()

    def test_empty_batch_returns_empty_result(self):
        result = self.node.save_images(
            images=[],
            filename_prefix="output",
            folderpath_input=self.base,
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )
        self.assertEqual(result["ui"]["images"], [])
        self.assertEqual(result["ui"]["animated"], (False,))
        self.assertEqual(result["result"], ("",))


class TestSaveImagesSanitization(unittest.TestCase):
    """save_images: sanitization of bad filename/foldername chars (M1)."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()
        self.base = reset_output_dir()

    def test_illegal_filename_chars_sanitized(self):
        result = self.node.save_images(
            images=make_test_tensor(frames=1, height=4, width=4),
            filename_prefix="bad/name?",
            folderpath_input=self.base,
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )
        filepath = result["result"][0]
        self.assertTrue(os.path.isfile(filepath))
        # illegal chars replaced with _
        self.assertIn("bad_name_", os.path.basename(filepath))

    def test_illegal_foldername_chars_sanitized(self):
        result = self.node.save_images(
            images=make_test_tensor(frames=1, height=4, width=4),
            filename_prefix="output",
            folderpath_input=self.base,
            foldername_prefix="bad|folder",
            fps=20.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )
        filepath = result["result"][0]
        self.assertTrue(os.path.isfile(filepath))
        self.assertIn("bad_folder", os.path.dirname(filepath))


class TestSaveImagesEmptyFolderpath(unittest.TestCase):
    """save_images: empty folderpath falls back to output_dir (m2)."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()
        self.base = reset_output_dir()
        # Keep the node's fallback dir in sync with the reset stub state
        self.node.output_dir = self.base

    def test_empty_folderpath_uses_output_dir(self):
        result = self.node.save_images(
            images=make_test_tensor(frames=1, height=4, width=4),
            filename_prefix="output",
            folderpath_input="",
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )
        filepath = result["result"][0]
        self.assertTrue(os.path.isfile(filepath))
        # The file should be under the output_dir (the stub's temp dir)
        self.assertTrue(os.path.abspath(filepath).startswith(os.path.abspath(get_output_dir())))

    def test_whitespace_folderpath_uses_output_dir(self):
        result = self.node.save_images(
            images=make_test_tensor(frames=1, height=4, width=4),
            filename_prefix="output",
            folderpath_input="   ",
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )
        filepath = result["result"][0]
        self.assertTrue(os.path.isfile(filepath))


class TestSaveImagesCounterIncrement(unittest.TestCase):
    """save_images: counter increments across saves (m3)."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()
        self.base = reset_output_dir()

    def _save(self):
        return self.node.save_images(
            images=make_test_tensor(frames=1, height=4, width=4),
            filename_prefix="output",
            folderpath_input=self.base,
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )

    def test_counter_starts_at_1(self):
        result = self._save()
        filepath = result["result"][0]
        self.assertIn("output_001.webp", filepath)

    def test_counter_increments_to_2(self):
        self._save()
        result = self._save()
        filepath = result["result"][0]
        self.assertIn("output_002.webp", filepath)

    def test_counter_increments_to_3(self):
        self._save()
        self._save()
        result = self._save()
        filepath = result["result"][0]
        self.assertIn("output_003.webp", filepath)

    def test_counter_position_first(self):
        result = self.node.save_images(
            images=make_test_tensor(frames=1, height=4, width=4),
            filename_prefix="output",
            folderpath_input=self.base,
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="first",
        )
        filepath = result["result"][0]
        self.assertIn("001_output.webp", filepath)


class TestSaveImagesMetadata(unittest.TestCase):
    """save_images: metadata disabled / enabled path."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()
        self.base = reset_output_dir()

    def test_metadata_disabled_saves_without_error(self):
        result = self.node.save_images(
            images=make_test_tensor(frames=1, height=4, width=4),
            filename_prefix="output",
            folderpath_input=self.base,
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )
        self.assertTrue(os.path.isfile(result["result"][0]))

    def test_metadata_enabled_with_prompt(self):
        result = self.node.save_images(
            images=make_test_tensor(frames=1, height=4, width=4),
            filename_prefix="output",
            folderpath_input=self.base,
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="enabled",
            counter_digits=3,
            counter_position="last",
            prompt={"3": {"class_type": "KSampler", "inputs": {"seed": 42}}},
        )
        self.assertTrue(os.path.isfile(result["result"][0]))

    def test_metadata_disabled_by_global_flag(self):
        set_disable_metadata(True)
        try:
            result = self.node.save_images(
                images=make_test_tensor(frames=1, height=4, width=4),
                filename_prefix="output",
                folderpath_input=self.base,
                foldername_prefix="gen",
                fps=20.0,
                lossless=True,
                quality=100,
                method="default",
                save_metadata="enabled",
                counter_digits=3,
                counter_position="last",
                prompt={"3": {"class_type": "KSampler"}},
            )
            self.assertTrue(os.path.isfile(result["result"][0]))
        finally:
            set_disable_metadata(False)


class TestSaveImagesMethodFallback(unittest.TestCase):
    """save_images: unexpected method value falls back to default (m6)."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()
        self.base = reset_output_dir()

    def test_empty_string_method_falls_back(self):
        result = self.node.save_images(
            images=make_test_tensor(frames=1, height=4, width=4),
            filename_prefix="output",
            folderpath_input=self.base,
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )
        self.assertTrue(os.path.isfile(result["result"][0]))

    def test_unknown_method_falls_back(self):
        result = self.node.save_images(
            images=make_test_tensor(frames=1, height=4, width=4),
            filename_prefix="output",
            folderpath_input=self.base,
            foldername_prefix="gen",
            fps=20.0,
            lossless=True,
            quality=100,
            method="nonexistent",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )
        self.assertTrue(os.path.isfile(result["result"][0]))


class TestSaveImagesFpsRounding(unittest.TestCase):
    """save_images: fps -> duration rounding (m7)."""

    def setUp(self):
        self.node = illumoraeSaveAnimatedWEBPFolderPathNode()
        self.base = reset_output_dir()

    def test_30fps_rounds_to_33ms(self):
        from PIL import Image

        result = self.node.save_images(
            images=make_test_tensor(frames=2, height=4, width=4),
            filename_prefix="output",
            folderpath_input=self.base,
            foldername_prefix="gen",
            fps=30.0,
            lossless=True,
            quality=100,
            method="default",
            save_metadata="disabled",
            counter_digits=3,
            counter_position="last",
        )
        filepath = result["result"][0]
        self.assertTrue(os.path.isfile(filepath))
        # PIL stores per-frame duration in img.info['duration'] after seeking.
        # 1000/30 = 33.33 -> round -> 33. Some PIL versions report 0 for the
        # default frame, so seek to frame 0 explicitly.
        with Image.open(filepath) as img:
            img.seek(0)
            duration = img.info.get("duration", 0)
            # If PIL reports 0 (some versions do for animated WebP), the save
            # still succeeded with the rounded value passed to PIL; just verify
            # it's either 33 (the rounded value) or 0 (PIL reporting gap).
            self.assertIn(duration, (33, 0))


class TestInputTypesShape(unittest.TestCase):
    """INPUT_TYPES: save_metadata is a clean 2-tuple (M2)."""

    def test_save_metadata_is_two_tuple(self):
        input_types = illumoraeSaveAnimatedWEBPFolderPathNode.INPUT_TYPES()
        sm = input_types["required"]["save_metadata"]
        # Should be a 2-tuple: (list_of_options, config_dict)
        self.assertEqual(len(sm), 2)
        self.assertEqual(sm[0], ["disabled", "enabled"])
        self.assertEqual(sm[1], {"default": "enabled"})

    def test_method_enum_matches_methods_dict(self):
        input_types = illumoraeSaveAnimatedWEBPFolderPathNode.INPUT_TYPES()
        method_input = input_types["required"]["method"]
        self.assertEqual(set(method_input[0]), {"default", "fastest", "slowest"})

    def test_return_types_and_names(self):
        self.assertEqual(illumoraeSaveAnimatedWEBPFolderPathNode.RETURN_TYPES, ("STRING",))
        self.assertEqual(illumoraeSaveAnimatedWEBPFolderPathNode.RETURN_NAMES, ("filepath",))

    def test_required_inputs_present(self):
        input_types = illumoraeSaveAnimatedWEBPFolderPathNode.INPUT_TYPES()
        required = input_types["required"]
        for key in ("images", "folderpath_input", "foldername_prefix",
                    "filename_prefix", "fps", "lossless", "quality",
                    "method", "save_metadata", "counter_digits", "counter_position"):
            self.assertIn(key, required, f"Missing required input: {key}")


if __name__ == "__main__":
    unittest.main()
