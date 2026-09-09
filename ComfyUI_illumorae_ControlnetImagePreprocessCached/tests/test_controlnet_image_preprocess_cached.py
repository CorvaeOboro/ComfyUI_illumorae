"""Tests for illumoraeControlnetImagePreprocessCachedNode.

Covers:
- _cache_path_for: derives "<source_dir>/<stem>/controlnet/<process>.png".
- _is_cache_fresh: missing source -> False; missing cache -> False;
  cache newer-or-equal -> True; cache older -> False; with mtime override;
  with check_newer=False (always fresh if cache exists).
- execute cache hit: pre-existing fresh cache -> loads PNG, returns
  cache_hit=True, preprocessor NOT called.
- execute cache miss: no cache -> runs preprocessor, writes cache PNG,
  returns cache_hit=False.
- execute force_refresh: fresh cache but force_refresh=True -> re-runs
  preprocessor and overwrites cache.
- execute update_cache_if_newer=False: stale cache but toggle off -> uses
  cache anyway, preprocessor NOT called.
- execute source_date_modified override: uses provided timestamp instead
  of file mtime for freshness comparison.
- execute leres: dispatches to the leres handler with correct kwargs.
- execute missing source_path: no caching attempted, preprocessor runs.
- execute unknown preprocessor: raises ValueError.
- IS_CHANGED: stable key across repeated calls with same mtimes; different
  when source mtime changes; different when force_refresh toggles;
  different when update_cache_if_newer toggles; includes leres params.
- Registration: NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS and class
  attribute arity are consistent.

The canny/lineart/leres handlers are mocked so no model dependencies are needed.

Usage:
    python -m tests.test_controlnet_image_preprocess_cached
    python -m pytest tests/test_controlnet_image_preprocess_cached.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch  # noqa: E402

from controlnet_image_preprocess_cached import (  # noqa: E402
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    illumoraeControlnetImagePreprocessCachedNode,
    _cache_path_for,
    _is_cache_fresh,
    _load_image_from_path,
    _png_to_tensor,
    _resolve_source,
    _tensor_to_png,
)


def _make_image(h=8, w=8):
    """A deterministic (1, H, W, 3) float32 image in [0, 1]."""
    t = torch.linspace(0, 1, h * w * 3).reshape(1, h, w, 3).float()
    return t


class TestCachePath(unittest.TestCase):
    def test_beside_source_with_preprocessor_suffix(self):
        p = _cache_path_for("C:/imgs/foo.png", "canny")
        self.assertEqual(p, Path("C:/imgs/foo/controlnet/canny.png"))

    def test_strips_extension(self):
        p = _cache_path_for("/data/sub/bar.jpg", "lineart")
        self.assertEqual(p, Path("/data/sub/bar/controlnet/lineart.png"))

    def test_leres_maps_to_depth_leres(self):
        p = _cache_path_for("/data/sub/bar.png", "leres")
        self.assertEqual(p, Path("/data/sub/bar/controlnet/depth_leres.png"))

    def test_source_name_combines_with_source_path(self):
        p = _cache_path_for("/data/imgs", "canny", source_name="item_01")
        self.assertEqual(p, Path("/data/imgs/item_01/controlnet/canny.png"))

    def test_source_name_leres_maps_to_depth_leres(self):
        p = _cache_path_for("/data/imgs", "leres", source_name="item_01")
        self.assertEqual(p, Path("/data/imgs/item_01/controlnet/depth_leres.png"))


class TestResolveSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "myitem.png"
        _tensor_to_png(_make_image(), self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_file_path_derives_stem_and_mtime(self):
        fp, stem, mtime, parent = _resolve_source(str(self.src), "", 0.0)
        self.assertEqual(fp, self.src)
        self.assertEqual(stem, "myitem")
        self.assertTrue(mtime > 0)
        self.assertEqual(parent, self.dir / "myitem")

    def test_folder_plus_name_finds_file(self):
        fp, stem, mtime, parent = _resolve_source(str(self.dir), "myitem", 0.0)
        self.assertEqual(fp, self.src)
        self.assertEqual(stem, "myitem")
        self.assertEqual(parent, self.dir / "myitem")

    def test_source_date_modified_override(self):
        fp, stem, mtime, parent = _resolve_source(str(self.src), "", 99999.0)
        self.assertEqual(mtime, 99999.0)

    def test_missing_file_returns_zero_mtime(self):
        fp, stem, mtime, parent = _resolve_source(
            str(self.dir / "nope.png"), "", 0.0)
        self.assertEqual(mtime, 0.0)

    def test_bare_stem_resolves_with_extension(self):
        """A path without extension should find the file via common extensions."""
        fp, stem, mtime, parent = _resolve_source(
            str(self.dir / "myitem"), "", 0.0)
        self.assertEqual(fp, self.src)
        self.assertEqual(stem, "myitem")
        self.assertTrue(mtime > 0)
        self.assertEqual(parent, self.dir / "myitem")


class TestLoadImageFromPath(unittest.TestCase):
    def test_loads_image_as_tensor(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            d = Path(tmp.name)
            src = d / "test.png"
            _tensor_to_png(_make_image(4, 6), src)
            t = _load_image_from_path(src)
            self.assertEqual(t.shape, (1, 4, 6, 3))
            self.assertTrue(t.dtype == torch.float32)
        finally:
            tmp.cleanup()


class TestIsCacheFresh(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "src.png"
        self.src.write_bytes(b"x")
        self.cache = self.dir / "src" / "controlnet" / "canny.png"
        self.cache.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_cache_is_not_fresh(self):
        self.assertFalse(_is_cache_fresh(str(self.src), self.cache))

    def test_missing_source_is_not_fresh(self):
        self.cache.write_bytes(b"y")
        self.assertFalse(_is_cache_fresh(str(self.dir / "nope.png"), self.cache))

    def test_empty_source_is_not_fresh(self):
        self.cache.write_bytes(b"y")
        self.assertFalse(_is_cache_fresh("", self.cache))

    def test_cache_newer_or_equal_is_fresh(self):
        self.cache.write_bytes(b"y")
        # Ensure cache mtime >= source mtime.
        os.utime(self.src, (time.time() - 5, time.time() - 5))
        os.utime(self.cache, (time.time(), time.time()))
        self.assertTrue(_is_cache_fresh(str(self.src), self.cache))

    def test_cache_older_is_not_fresh(self):
        self.cache.write_bytes(b"y")
        os.utime(self.cache, (time.time() - 10, time.time() - 10))
        os.utime(self.src, (time.time(), time.time()))
        self.assertFalse(_is_cache_fresh(str(self.src), self.cache))

    def test_source_mtime_override_used(self):
        self.cache.write_bytes(b"y")
        # Cache is older than the real file, but the override says source
        # is very old -> cache should be fresh.
        os.utime(self.cache, (time.time() - 10, time.time() - 10))
        os.utime(self.src, (time.time(), time.time()))
        old_mtime = time.time() - 100
        self.assertTrue(_is_cache_fresh(str(self.src), self.cache,
                                         source_mtime_override=old_mtime))

    def test_source_mtime_override_newer_than_cache(self):
        self.cache.write_bytes(b"y")
        os.utime(self.cache, (time.time() - 100, time.time() - 100))
        # Override says source is newer than cache -> not fresh.
        new_mtime = time.time()
        self.assertFalse(_is_cache_fresh(str(self.src), self.cache,
                                          source_mtime_override=new_mtime))

    def test_check_newer_false_always_fresh_if_cache_exists(self):
        self.cache.write_bytes(b"y")
        # Cache is much older than source, but check_newer=False.
        os.utime(self.cache, (time.time() - 100, time.time() - 100))
        os.utime(self.src, (time.time(), time.time()))
        self.assertTrue(_is_cache_fresh(str(self.src), self.cache,
                                         check_newer=False))

    def test_check_newer_false_still_false_if_no_cache(self):
        self.assertFalse(_is_cache_fresh(str(self.src), self.cache,
                                         check_newer=False))


class TestExecute(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "src.png"
        _tensor_to_png(_make_image(), self.src)
        self.node = illumoraeControlnetImagePreprocessCachedNode()
        self.fake_result = torch.ones(1, 4, 4, 3)

        def _fake_canny(image, **kw):
            return self.fake_result.clone()

        def _fake_lineart(image, **kw):
            return self.fake_result.clone()

        def _fake_leres(image, **kw):
            self._leres_kwargs = kw
            return self.fake_result.clone()

        self._patch_canny = mock.patch(
            "controlnet_image_preprocess_cached.PREPROCESSOR_DISPATCH",
            {"canny": _fake_canny, "lineart": _fake_lineart, "leres": _fake_leres},
        )
        self._patch_canny.start()

    def tearDown(self):
        self._patch_canny.stop()
        self.tmp.cleanup()

    def test_cache_miss_runs_preprocessor_and_writes_cache(self):
        img = _make_image()
        result, hit = self.node.execute(img, str(self.src), "canny")
        self.assertFalse(hit)
        self.assertTrue(torch.allclose(result, self.fake_result))
        cache = _cache_path_for(str(self.src), "canny")
        self.assertTrue(cache.is_file())

    def test_cache_hit_loads_png_and_skips_preprocessor(self):
        img = _make_image()
        # First run: writes cache.
        self.node.execute(img, str(self.src), "canny")
        cache = _cache_path_for(str(self.src), "canny")
        # Make cache newer than source.
        os.utime(self.src, (time.time() - 5, time.time() - 5))
        os.utime(cache, (time.time(), time.time()))
        # Second run: should hit cache.
        with mock.patch("controlnet_image_preprocess_cached.PREPROCESSOR_DISPATCH",
                        {"canny": self._fail_if_called("canny"),
                         "lineart": self._fail_if_called("lineart"),
                         "leres": self._fail_if_called("leres")}):
            result, hit = self.node.execute(img, str(self.src), "canny")
        self.assertTrue(hit)
        loaded = _png_to_tensor(cache)
        self.assertEqual(tuple(result.shape), tuple(loaded.shape))

    def test_force_refresh_ignores_fresh_cache(self):
        img = _make_image()
        self.node.execute(img, str(self.src), "canny")
        cache = _cache_path_for(str(self.src), "canny")
        os.utime(self.src, (time.time() - 5, time.time() - 5))
        os.utime(cache, (time.time(), time.time()))
        called = {"n": 0}

        def _spy(image, **kw):
            called["n"] += 1
            return self.fake_result.clone()

        with mock.patch("controlnet_image_preprocess_cached.PREPROCESSOR_DISPATCH",
                        {"canny": _spy, "lineart": _spy, "leres": _spy}):
            result, hit = self.node.execute(img, str(self.src), "canny",
                                            force_refresh=True)
        self.assertFalse(hit)
        self.assertEqual(called["n"], 1)

    def test_missing_source_path_runs_without_caching(self):
        img = _make_image()
        result, hit = self.node.execute(img, "", "canny")
        self.assertFalse(hit)
        # No cache file written anywhere in the temp dir tree.
        caches = list(self.dir.rglob("canny.png"))
        self.assertEqual(caches, [])

    def test_unknown_preprocessor_raises(self):
        with self.assertRaises(ValueError):
            self.node.execute(_make_image(), str(self.src), "nope")

    def test_update_cache_if_newer_false_uses_stale_cache(self):
        img = _make_image()
        self.node.execute(img, str(self.src), "canny")
        cache = _cache_path_for(str(self.src), "canny")
        # Make cache much older than source (stale).
        os.utime(self.cache if hasattr(self, 'cache') else cache,
                 (time.time() - 100, time.time() - 100))
        os.utime(self.src, (time.time(), time.time()))
        # With update_cache_if_newer=False, should use the stale cache.
        with mock.patch("controlnet_image_preprocess_cached.PREPROCESSOR_DISPATCH",
                        {"canny": self._fail_if_called("canny"),
                         "lineart": self._fail_if_called("lineart"),
                         "leres": self._fail_if_called("leres")}):
            result, hit = self.node.execute(img, str(self.src), "canny",
                                            update_cache_if_newer=False)
        self.assertTrue(hit)

    def test_source_date_modified_override(self):
        img = _make_image()
        self.node.execute(img, str(self.src), "canny")
        cache = _cache_path_for(str(self.src), "canny")
        # Make cache older than the real file mtime.
        os.utime(cache, (time.time() - 100, time.time() - 100))
        os.utime(self.src, (time.time(), time.time()))
        # But provide a source_date_modified that's even older than the cache.
        old_ts = time.time() - 200
        with mock.patch("controlnet_image_preprocess_cached.PREPROCESSOR_DISPATCH",
                        {"canny": self._fail_if_called("canny"),
                         "lineart": self._fail_if_called("lineart"),
                         "leres": self._fail_if_called("leres")}):
            result, hit = self.node.execute(img, str(self.src), "canny",
                                            source_date_modified=old_ts)
        self.assertTrue(hit)

    def test_source_date_modified_override_newer_regenerates(self):
        img = _make_image()
        self.node.execute(img, str(self.src), "canny")
        cache = _cache_path_for(str(self.src), "canny")
        # Make cache older than source.
        os.utime(cache, (time.time() - 100, time.time() - 100))
        os.utime(self.src, (time.time() - 50, time.time() - 50))
        # Provide a source_date_modified that's newer than the cache.
        new_ts = time.time()
        called = {"n": 0}

        def _spy(image, **kw):
            called["n"] += 1
            return self.fake_result.clone()

        with mock.patch("controlnet_image_preprocess_cached.PREPROCESSOR_DISPATCH",
                        {"canny": _spy, "lineart": _spy, "leres": _spy}):
            result, hit = self.node.execute(img, str(self.src), "canny",
                                            source_date_modified=new_ts)
        self.assertFalse(hit)
        self.assertEqual(called["n"], 1)

    def test_leres_dispatches_with_correct_kwargs(self):
        img = _make_image()
        self.node.execute(img, str(self.src), "leres",
                          rm_nearest=5.0, rm_background=10.0,
                          boost="enable", resolution=768)
        kw = getattr(self, "_leres_kwargs", {})
        self.assertEqual(kw.get("rm_nearest"), 5.0)
        self.assertEqual(kw.get("rm_background"), 10.0)
        self.assertEqual(kw.get("boost"), "enable")
        self.assertEqual(kw.get("resolution"), 768)
        cache = _cache_path_for(str(self.src), "leres")
        self.assertTrue(cache.is_file())

    def test_debug_mode_logs_to_console(self):
        img = _make_image()
        with mock.patch("controlnet_image_preprocess_cached.print") as mock_print:
            self.node.execute(img, str(self.src), "canny", debug=True)
        self.assertTrue(mock_print.called)
        logged = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("ControlnetImagePreprocessCached", logged)

    def test_debug_off_does_not_log(self):
        img = _make_image()
        with mock.patch("controlnet_image_preprocess_cached.print") as mock_print:
            self.node.execute(img, str(self.src), "canny", debug=False)
        self.assertFalse(mock_print.called)

    def test_source_name_writes_cache_to_subfolder(self):
        """When source_name is provided, cache goes to <source_path>/<source_name>/controlnet/."""
        img = _make_image()
        # source_path is the temp dir (a folder), source_name is the item stem.
        result, hit = self.node.execute(
            img, str(self.dir), "canny",
            source_name="myitem", source_date_modified=time.time())
        self.assertFalse(hit)
        expected = self.dir / "myitem" / "controlnet" / "canny.png"
        self.assertTrue(expected.is_file())

    def test_source_name_cache_hit(self):
        """Cache written with source_name should be found on the second run."""
        img = _make_image()
        ts = time.time()
        self.node.execute(img, str(self.dir), "canny",
                          source_name="myitem2", source_date_modified=ts)
        with mock.patch("controlnet_image_preprocess_cached.PREPROCESSOR_DISPATCH",
                        {"canny": self._fail_if_called("canny"),
                         "lineart": self._fail_if_called("lineart"),
                         "leres": self._fail_if_called("leres")}):
            result, hit = self.node.execute(
                img, str(self.dir), "canny",
                source_name="myitem2", source_date_modified=ts)
        self.assertTrue(hit)

    def test_self_contained_loads_from_path_no_image_input(self):
        """When no IMAGE tensor is wired, the node loads the image from source_path."""
        # Write a source image to disk.
        src_file = self.dir / "standalone.png"
        _tensor_to_png(_make_image(), src_file)
        result, hit = self.node.execute(
            source_path=str(src_file), preprocessor="canny")
        self.assertFalse(hit)
        cache = self.dir / "standalone" / "controlnet" / "canny.png"
        self.assertTrue(cache.is_file())

    def test_self_contained_no_image_no_path_raises(self):
        """Without an IMAGE input or a valid source_path, the node raises."""
        with self.assertRaises(ValueError):
            self.node.execute(source_path="", preprocessor="canny")

    @staticmethod
    def _fail_if_called(name):
        def _fn(image, **kw):
            raise AssertionError("{} handler called during a cache hit".format(name))
        return _fn


class TestIsChanged(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "src.png"
        _tensor_to_png(_make_image(), self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stable_key_for_same_inputs(self):
        img = _make_image()
        k1 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(img, str(self.src), "canny")
        k2 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(img, str(self.src), "canny")
        self.assertEqual(k1, k2)

    def test_key_changes_when_source_mtime_changes(self):
        img = _make_image()
        k1 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(img, str(self.src), "canny")
        os.utime(self.src, (time.time() + 10, time.time() + 10))
        k2 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(img, str(self.src), "canny")
        self.assertNotEqual(k1, k2)

    def test_key_changes_when_force_refresh_toggles(self):
        img = _make_image()
        k1 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(
            img, str(self.src), "canny", force_refresh=False)
        k2 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(
            img, str(self.src), "canny", force_refresh=True)
        self.assertNotEqual(k1, k2)

    def test_key_changes_when_update_cache_if_newer_toggles(self):
        img = _make_image()
        k1 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(
            img, str(self.src), "canny", update_cache_if_newer=True)
        k2 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(
            img, str(self.src), "canny", update_cache_if_newer=False)
        self.assertNotEqual(k1, k2)

    def test_key_changes_when_source_date_modified_changes(self):
        img = _make_image()
        k1 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(
            img, str(self.src), "canny", source_date_modified=0.0)
        k2 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(
            img, str(self.src), "canny", source_date_modified=1234567890.0)
        self.assertNotEqual(k1, k2)

    def test_key_changes_when_leres_params_change(self):
        img = _make_image()
        k1 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(
            img, str(self.src), "leres", rm_nearest=0.0, boost="disable")
        k2 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(
            img, str(self.src), "leres", rm_nearest=5.0, boost="enable")
        self.assertNotEqual(k1, k2)

    def test_key_changes_when_cache_written(self):
        img = _make_image()
        k1 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(img, str(self.src), "canny")
        cache = _cache_path_for(str(self.src), "canny")
        _tensor_to_png(_make_image(), cache)
        k2 = illumoraeControlnetImagePreprocessCachedNode.IS_CHANGED(img, str(self.src), "canny")
        self.assertNotEqual(k1, k2)


class TestRegistration(unittest.TestCase):
    def test_mappings_consistent(self):
        self.assertIn("illumoraeControlnetImagePreprocessCachedNode", NODE_CLASS_MAPPINGS)
        self.assertIn("illumoraeControlnetImagePreprocessCachedNode", NODE_DISPLAY_NAME_MAPPINGS)
        cls = NODE_CLASS_MAPPINGS["illumoraeControlnetImagePreprocessCachedNode"]
        self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE", "BOOLEAN"))
        self.assertEqual(cls.RETURN_NAMES, ("image", "cache_hit"))
        self.assertEqual(cls.FUNCTION, "execute")
        self.assertEqual(cls.CATEGORY, "illumorae")


if __name__ == "__main__":
    unittest.main()
