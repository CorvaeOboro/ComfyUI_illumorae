"""Regression tests for the Save Image Extended FolderPath node.

Covers the bugs fixed in the 2026-08-16 review:

1. Malformed ``save_metadata`` INPUT_TYPES entry (was a 4-tuple, now a 2-tuple).
2. Unused imports ``sys`` and ``locale`` removed.
3. Unused ``resolution`` block removed (F841).
4. Dead job-data feature removed (five static methods + ``save_job_data``
   input + optional prompt inputs + ``save_job_data``/``positive_text_opt``/
   ``negative_text_opt`` parameters from ``save_images``).
5. Redundant double ``sanitize_name`` on ``custom_foldername`` removed.
6. Dead ``custom_filename`` binding/re-sanitization from
   ``get_save_image_path`` return removed.
7. ``get_latest_counter`` now receives the processed ``filename`` and
   ``delimiter_char`` so the scan matches on-disk names.
8. ``get_latest_counter`` ``first`` position now skips
   ``counter_digits + len(delimiter_char)`` instead of hardcoding ``+1``.
9. Metadata construction hoisted above the save loop (was rebuilt per image).
10. Errors are re-raised instead of swallowed.
11. Empty-batch guard added at the top of ``save_images``.

Usage:
    python -m pytest tests/test_save_image_extended_folderpath.py -v
    python -m tests.test_save_image_extended_folderpath
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

# conftest.py (same directory) installs the folder_paths stub and puts the
# package directory on sys.path before this import runs.
from save_image_extended_folderpath import illumoraeSaveImageExtendedFolderPathNode  # noqa: E402

from conftest import make_test_batch  # noqa: E402


class TestInputTypes(unittest.TestCase):
    """Verify INPUT_TYPES is well-formed after the malformed-entry fix."""

    def setUp(self):
        self.node = illumoraeSaveImageExtendedFolderPathNode()
        self.inputs = self.node.INPUT_TYPES()

    def test_save_metadata_is_two_tuple(self):
        """save_metadata must be (options_list, kwargs_dict) — not a 4-tuple."""
        entry = self.inputs["required"]["save_metadata"]
        self.assertEqual(len(entry), 2, f"save_metadata entry has {len(entry)} elements, expected 2")
        options, kwargs = entry
        self.assertIsInstance(options, list)
        self.assertIn("enabled", options)
        self.assertIn("disabled", options)
        self.assertEqual(kwargs["default"], "enabled")

    def test_save_job_data_input_removed(self):
        """The dead save_job_data input must no longer be present."""
        self.assertNotIn("save_job_data", self.inputs["required"])

    def test_optional_prompt_inputs_removed(self):
        """The dead positive_text_opt / negative_text_opt inputs must be gone."""
        self.assertNotIn("optional", self.inputs)

    def test_all_required_inputs_present(self):
        expected = {
            "images", "folderpath_input", "foldername_prefix",
            "filename_prefix", "delimiter", "save_metadata",
            "counter_digits", "counter_position",
        }
        self.assertEqual(set(self.inputs["required"].keys()), expected)

    def test_delimiter_options_have_spaces(self):
        """The delimiter combo options should be properly spaced."""
        options = self.inputs["required"]["delimiter"][0]
        self.assertEqual(options, ["underscore", "dot", "comma"])


class TestSanitizeName(unittest.TestCase):
    """Tests for the static sanitize_name method."""

    def test_replaces_illegal_chars(self):
        result = illumoraeSaveImageExtendedFolderPathNode.sanitize_name('a<b>c:"d/e\\f|g?h*i')
        self.assertEqual(result, "a_b_c__d_e_f_g_h_i")

    def test_strips_whitespace(self):
        result = illumoraeSaveImageExtendedFolderPathNode.sanitize_name("  hello  ")
        self.assertEqual(result, "hello")

    def test_removes_control_chars(self):
        result = illumoraeSaveImageExtendedFolderPathNode.sanitize_name("a\tb\nc")
        self.assertEqual(result, "abc")

    def test_removes_non_ascii(self):
        # Non-ASCII chars are dropped (not in string.printable)
        result = illumoraeSaveImageExtendedFolderPathNode.sanitize_name("caf" + chr(233) + "")
        self.assertEqual(result, "caf")

    def test_preserves_underscore_dot_comma(self):
        """Delimiter chars must survive sanitization."""
        for ch in ["_", ".", ","]:
            result = illumoraeSaveImageExtendedFolderPathNode.sanitize_name(f"a{ch}b")
            self.assertEqual(result, f"a{ch}b")

    def test_empty_string(self):
        self.assertEqual(illumoraeSaveImageExtendedFolderPathNode.sanitize_name(""), "")

    def test_idempotent(self):
        """Running sanitize_name twice produces the same result as once."""
        name = 'test<>:"file?name*'
        once = illumoraeSaveImageExtendedFolderPathNode.sanitize_name(name)
        twice = illumoraeSaveImageExtendedFolderPathNode.sanitize_name(once)
        self.assertEqual(once, twice)


class TestGetSubfolderPath(unittest.TestCase):
    """Tests for get_subfolder_path."""

    def setUp(self):
        self.node = illumoraeSaveImageExtendedFolderPathNode()

    def test_returns_parent_subfolder(self):
        import tempfile
        with tempfile.TemporaryDirectory() as base:
            sub = os.path.join(base, "gen")
            os.makedirs(sub)
            image_path = os.path.join(sub, "output_001.png")
            result = self.node.get_subfolder_path(image_path, base)
            self.assertEqual(result, "gen")

    def test_returns_empty_for_cross_drive(self):
        """When the image is not under output_path, returns empty string."""
        # On Windows, different drive letters cause relative_to to fail.
        # On non-Windows, use two unrelated temp dirs.
        import tempfile
        with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
            image_path = os.path.join(dir1, "test.png")
            result = self.node.get_subfolder_path(image_path, dir2)
            self.assertEqual(result, "")


class TestGetLatestCounter(unittest.TestCase):
    """Tests for get_latest_counter, including the delimiter-aware first-position fix."""

    def setUp(self):
        self.node = illumoraeSaveImageExtendedFolderPathNode()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _create_files(self, names):
        for name in names:
            with open(os.path.join(self.tmp, name), "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")

    def test_empty_folder_returns_1(self):
        counter = self.node.get_latest_counter(self.tmp, "output", 3, "last", "_")
        self.assertEqual(counter, 1)

    def test_nonexistent_folder_returns_1(self):
        counter = self.node.get_latest_counter(
            os.path.join(self.tmp, "nonexistent"), "output", 3, "last", "_"
        )
        self.assertEqual(counter, 1)

    def test_last_position_underscore(self):
        self._create_files(["output_001.png", "output_002.png", "output_005.png"])
        counter = self.node.get_latest_counter(self.tmp, "output", 3, "last", "_")
        self.assertEqual(counter, 6)

    def test_last_position_dot(self):
        self._create_files(["output.001.png", "output.002.png"])
        counter = self.node.get_latest_counter(self.tmp, "output", 3, "last", ".")
        self.assertEqual(counter, 3)

    def test_last_position_comma(self):
        self._create_files(["output,001.png", "output,002.png"])
        counter = self.node.get_latest_counter(self.tmp, "output", 3, "last", ",")
        self.assertEqual(counter, 3)

    def test_first_position_underscore(self):
        self._create_files(["001_output.png", "002_output.png", "005_output.png"])
        counter = self.node.get_latest_counter(self.tmp, "output", 3, "first", "_")
        self.assertEqual(counter, 6)

    def test_first_position_dot(self):
        """The bug fix: first position with dot delimiter must skip
        counter_digits + len(delimiter_char) = 3 + 1 = 4 chars."""
        self._create_files(["001.output.png", "002.output.png"])
        counter = self.node.get_latest_counter(self.tmp, "output", 3, "first", ".")
        self.assertEqual(counter, 3)

    def test_first_position_comma(self):
        self._create_files(["001,output.png", "002,output.png"])
        counter = self.node.get_latest_counter(self.tmp, "output", 3, "first", ",")
        self.assertEqual(counter, 3)

    def test_ignores_non_matching_prefix(self):
        self._create_files(["output_001.png", "other_010.png"])
        counter = self.node.get_latest_counter(self.tmp, "output", 3, "last", "_")
        self.assertEqual(counter, 2)

    def test_ignores_non_png_files(self):
        self._create_files(["output_001.png"])
        with open(os.path.join(self.tmp, "output_002.txt"), "w") as f:
            f.write("not an image")
        counter = self.node.get_latest_counter(self.tmp, "output", 3, "last", "_")
        self.assertEqual(counter, 2)

    def test_invalid_position_defaults_to_last(self):
        self._create_files(["output_001.png", "output_002.png"])
        counter = self.node.get_latest_counter(self.tmp, "output", 3, "invalid", "_")
        self.assertEqual(counter, 3)

    def test_different_counter_digits(self):
        self._create_files(["output_00001.png", "output_00002.png"])
        counter = self.node.get_latest_counter(self.tmp, "output", 5, "last", "_")
        self.assertEqual(counter, 3)


class TestSaveImages(unittest.TestCase):
    """Integration tests for the save_images method."""

    def setUp(self):
        self.node = illumoraeSaveImageExtendedFolderPathNode()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _save(self, **overrides):
        """Call save_images with sensible defaults, overridden by kwargs."""
        defaults = dict(
            counter_digits=3,
            counter_position="last",
            delimiter="underscore",
            save_metadata="enabled",
            images=make_test_batch(frames=2, height=8, width=8),
            filename_prefix="output",
            foldername_prefix="gen",
            folderpath_input=self.tmp,
            extra_pnginfo=None,
            prompt=None,
        )
        defaults.update(overrides)
        return self.node.save_images(**defaults)

    def test_basic_save(self):
        result = self._save()
        self.assertIn("ui", result)
        self.assertIn("images", result["ui"])
        self.assertEqual(len(result["ui"]["images"]), 2)
        for entry in result["ui"]["images"]:
            self.assertTrue(entry["filename"].endswith(".png"))
            self.assertEqual(entry["type"], "output")

    def test_files_actually_written(self):
        self._save()
        gen_dir = os.path.join(self.tmp, "gen")
        files = [f for f in os.listdir(gen_dir) if f.endswith(".png")]
        self.assertEqual(len(files), 2)

    def test_counter_increments_within_batch(self):
        result = self._save()
        filenames = [e["filename"] for e in result["ui"]["images"]]
        self.assertIn("output_001.png", filenames)
        self.assertIn("output_002.png", filenames)

    def test_counter_continues_across_calls(self):
        self._save()
        result = self._save()
        filenames = [e["filename"] for e in result["ui"]["images"]]
        self.assertIn("output_003.png", filenames)
        self.assertIn("output_004.png", filenames)

    def test_counter_position_first(self):
        result = self._save(counter_position="first")
        filenames = [e["filename"] for e in result["ui"]["images"]]
        self.assertIn("001_output.png", filenames)
        self.assertIn("002_output.png", filenames)

    def test_delimiter_dot(self):
        result = self._save(delimiter="dot")
        filenames = [e["filename"] for e in result["ui"]["images"]]
        self.assertIn("output.001.png", filenames)

    def test_delimiter_comma(self):
        result = self._save(delimiter="comma")
        filenames = [e["filename"] for e in result["ui"]["images"]]
        self.assertIn("output,001.png", filenames)

    def test_metadata_disabled(self):
        from PIL import Image
        self._save(save_metadata="disabled")
        gen_dir = os.path.join(self.tmp, "gen")
        files = [f for f in os.listdir(gen_dir) if f.endswith(".png")]
        img = Image.open(os.path.join(gen_dir, files[0]))
        # No prompt metadata should be embedded
        self.assertIsNone(img.info.get("prompt"))

    def test_metadata_enabled_with_prompt(self):
        from PIL import Image
        test_prompt = {"3": {"class_type": "KSampler", "inputs": {"seed": 42}}}
        self._save(prompt=test_prompt)
        gen_dir = os.path.join(self.tmp, "gen")
        files = [f for f in os.listdir(gen_dir) if f.endswith(".png")]
        img = Image.open(os.path.join(gen_dir, files[0]))
        embedded = json.loads(img.info["prompt"])
        self.assertEqual(embedded, test_prompt)

    def test_metadata_enabled_with_extra_pnginfo(self):
        from PIL import Image
        extra = {"workflow": {"nodes": [1, 2, 3]}}
        self._save(extra_pnginfo=extra)
        gen_dir = os.path.join(self.tmp, "gen")
        files = [f for f in os.listdir(gen_dir) if f.endswith(".png")]
        img = Image.open(os.path.join(gen_dir, files[0]))
        embedded = json.loads(img.info["workflow"])
        self.assertEqual(embedded, {"nodes": [1, 2, 3]})

    def test_empty_folderpath_falls_back_to_output_dir(self):
        """When folderpath_input is empty, the node falls back to output_dir."""
        import conftest
        old = conftest._STATE["output_dir"]
        try:
            conftest._STATE["output_dir"] = self.tmp
            result = self._save(folderpath_input="")
            self.assertIn("ui", result)
        finally:
            conftest._STATE["output_dir"] = old

    def test_empty_batch_returns_empty_list(self):
        """The empty-batch guard must return an empty images list."""
        result = self._save(images=[])
        self.assertEqual(result, {"ui": {"images": []}})

    def test_none_images_returns_empty_list(self):
        result = self._save(images=None)
        self.assertEqual(result, {"ui": {"images": []}})

    def test_sanitizes_filename_prefix(self):
        """Illegal characters in filename_prefix are replaced."""
        result = self._save(filename_prefix='test<>:"file')
        filenames = [e["filename"] for e in result["ui"]["images"]]
        for fn in filenames:
            self.assertTrue(fn.startswith("test____file"))

    def test_sanitizes_foldername_prefix(self):
        """Illegal characters in foldername_prefix are replaced."""
        self._save(foldername_prefix='bad<>name')
        gen_dir = os.path.join(self.tmp, "bad__name")
        self.assertTrue(os.path.isdir(gen_dir))

    def test_creates_nested_folder(self):
        """The node creates the output folder if it doesn't exist."""
        new_path = os.path.join(self.tmp, "nested", "deep")
        result = self._save(folderpath_input=new_path)
        gen_dir = os.path.join(new_path, "gen")
        self.assertTrue(os.path.isdir(gen_dir))
        self.assertEqual(len(result["ui"]["images"]), 2)

    def test_counter_digits_4(self):
        result = self._save(counter_digits=4)
        filenames = [e["filename"] for e in result["ui"]["images"]]
        self.assertIn("output_0001.png", filenames)

    def test_error_is_raised_not_swallowed(self):
        """Errors must be re-raised, not silently swallowed.

        We force an error by making the output path unwritable.
        """
        # Create a file where a directory is expected, causing makedirs to fail
        blocking_file = os.path.join(self.tmp, "gen")
        with open(blocking_file, "w") as f:
            f.write("blocking file")
        with self.assertRaises(Exception):
            self._save()


class TestDeadCodeRemoval(unittest.TestCase):
    """Verify the dead job-data helpers are no longer present on the class."""

    def test_find_keys_recursively_removed(self):
        self.assertFalse(hasattr(illumoraeSaveImageExtendedFolderPathNode, "find_keys_recursively"))

    def test_remove_file_extension_removed(self):
        self.assertFalse(hasattr(illumoraeSaveImageExtendedFolderPathNode, "remove_file_extension"))

    def test_find_parameter_values_removed(self):
        self.assertFalse(hasattr(illumoraeSaveImageExtendedFolderPathNode, "find_parameter_values"))

    def test_generate_custom_name_removed(self):
        self.assertFalse(hasattr(illumoraeSaveImageExtendedFolderPathNode, "generate_custom_name"))

    def test_save_job_to_json_removed(self):
        self.assertFalse(hasattr(illumoraeSaveImageExtendedFolderPathNode, "save_job_to_json"))


class TestSaveImagesSignature(unittest.TestCase):
    """Verify save_images no longer accepts the removed parameters."""

    def test_no_save_job_data_param(self):
        import inspect
        sig = inspect.signature(illumoraeSaveImageExtendedFolderPathNode.save_images)
        self.assertNotIn("save_job_data", sig.parameters)

    def test_no_positive_text_opt_param(self):
        import inspect
        sig = inspect.signature(illumoraeSaveImageExtendedFolderPathNode.save_images)
        self.assertNotIn("positive_text_opt", sig.parameters)

    def test_no_negative_text_opt_param(self):
        import inspect
        sig = inspect.signature(illumoraeSaveImageExtendedFolderPathNode.save_images)
        self.assertNotIn("negative_text_opt", sig.parameters)


class TestNodeMappings(unittest.TestCase):
    """Verify the node class mappings are correct."""

    def test_class_mappings_present(self):
        from save_image_extended_folderpath import NODE_CLASS_MAPPINGS
        self.assertIn("illumoraeSaveImageExtendedFolderPathNode", NODE_CLASS_MAPPINGS)

    def test_display_name_mappings_present(self):
        from save_image_extended_folderpath import NODE_DISPLAY_NAME_MAPPINGS
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["illumoraeSaveImageExtendedFolderPathNode"],
            "Save Image Extended FolderPath",
        )


if __name__ == "__main__":
    unittest.main()
