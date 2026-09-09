"""
Tests for deterministic file ordering (review item 2.3).

The determinism guarantee lives in _get_all_loras_recursive, which sorts its
output by (rel_path, base_dir) before returning. find_matching_lora_pair
iterates that list as given and keeps the first candidate on a score tie
(strict > comparison), so a sorted input makes ties resolve to the
lexicographically smallest path consistently regardless of filesystem walk
order.

Verifies that:
- _get_all_loras_recursive returns a sorted list (by rel_path then base_dir).
- find_matching_lora_pair resolves a score-90 tie deterministically when fed
  the sorted output the walker produces in production.

Run from the repo root:
    python ComfyUI_illumorae_Wan22LoraLoaderByText/tests/test_deterministic_ordering.py
"""
import os
import tempfile
import unittest

from _test_stub import load_node_module

_mod, Node = load_node_module()


class TestWalkerOrdering(unittest.TestCase):
    def test_walker_output_is_sorted(self):
        with tempfile.TemporaryDirectory() as base:
            # Create files in non-sorted order.
            for name in ['zeta.safetensors', 'alpha.safetensors', 'mid.safetensors']:
                with open(os.path.join(base, name), 'w'):
                    pass
            result = Node._get_all_loras_recursive([base])
            rel_paths = [rel for rel, _ in result]
            self.assertEqual(rel_paths, sorted(rel_paths, key=str.lower))

    def test_walker_sort_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as base:
            for name in ['Beta.safetensors', 'alpha.safetensors']:
                with open(os.path.join(base, name), 'w'):
                    pass
            result = Node._get_all_loras_recursive([base])
            rel_paths = [rel for rel, _ in result]
            self.assertEqual(rel_paths, ['alpha.safetensors', 'Beta.safetensors'])


class TestTieBreakingDeterministic(unittest.TestCase):
    def _files(self, *paths):
        return [(p, '/fake/loras') for p in paths]

    def test_score_tie_picks_first_in_sorted_input(self):
        """Two no-variant files both match 'mystyle' as a meaningful substring
        at score 90. Fed in sorted order (as the walker produces), the first
        (lexicographically smaller) path wins as the primary."""
        files = self._files(
            'packs/aaa_mystyle.safetensors',
            'packs/zzz_mystyle.safetensors',
        )
        result = Node.find_matching_lora_pair(
            'mystyle', files,
            model_type_mode='NONE', fallback_classification='HIGH',
        )
        self.assertIsNotNone(result['high_match'])
        self.assertEqual(result['high_match'][0], 'packs/aaa_mystyle.safetensors')

    def test_walker_then_match_is_deterministic(self):
        """End-to-end: the walker sorts, and feeding its output to the matcher
        resolves a tie to the smallest path. Repeating the walk yields the
        same match regardless of creation order on disk."""
        with tempfile.TemporaryDirectory() as base:
            for name in ['zzz_mystyle.safetensors', 'aaa_mystyle.safetensors']:
                with open(os.path.join(base, name), 'w'):
                    pass
            collected = Node._get_all_loras_recursive([base])
            result = Node.find_matching_lora_pair(
                'mystyle', collected,
                model_type_mode='NONE', fallback_classification='HIGH',
            )
            self.assertIsNotNone(result['high_match'])
            self.assertEqual(result['high_match'][0], 'aaa_mystyle.safetensors')


if __name__ == '__main__':
    unittest.main(verbosity=2)
