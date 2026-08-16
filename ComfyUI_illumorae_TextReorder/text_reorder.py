"""
TITLE::Text Reorder
DESCRIPTIONSHORT::Reorders prompt text sections (comma/sentence/paragraph) while preserving parenthesis-enclosed blocks; supports seeded shuffle and distance constraints.
VERSION::20260127
IMAGE::comfyui_illumorae_text_reorder.png
GROUP::Text
GROUPORDER::5
LISTORDER::70
STATUS::working
"""
import random
from typing import List, Tuple

#region CLASS
class illumoraeTextReorderNode:
    """
    A ComfyUI node that reorders text sections based on parenthesis enclosures.

    Features:
    - Parses text into sections (enclosed in parenthesis or orphaned text between enclosures)
    - Randomizes section order based on seed
    - Optional distance constraint to limit how far sections can move from original position
    - Preserves section content and structure
    """

    VALID_REORDER_MODES = ("paragraph", "sentence", "comma")
    #endregion

    #region PARSE
    def parse_sections_mode(self, text: str, reorder_mode: str) -> Tuple[List[str], List[int]]:
        """Parse text into enclosed (parenthesis) and orphaned sections.

        Tracks paren depth and angle-bracket depth so that '(' inside
        <lora:...> tags does not start an enclosed section. Orphaned text
        is split by the active delimiter (comma, sentence, paragraph).
        Whitespace-only segments are preserved for round-trip fidelity.
        """
        sections = []
        positions = []

        current_pos = 0
        i = 0
        depth = 0
        angle_depth = 0
        section_start = 0
        in_enclosed_section = False

        while i < len(text):
            char = text[i]

            if char == '<':
                angle_depth += 1
            elif char == '>':
                angle_depth = max(0, angle_depth - 1)

            if char == '(' and depth == 0 and angle_depth == 0:
                if i > section_start:
                    orphaned = text[section_start:i]
                    if orphaned:  # preserve whitespace-only segments for round-trip fidelity
                        sections.append(orphaned)
                        positions.append(current_pos)
                        current_pos += 1

                section_start = i
                in_enclosed_section = True
                depth = 1
                i += 1

                while i < len(text):
                    if text[i] == '<':
                        angle_depth += 1
                    elif text[i] == '>':
                        angle_depth = max(0, angle_depth - 1)
                    elif text[i] == '(':
                        depth += 1
                    elif text[i] == ')':
                        depth -= 1
                        if depth == 0:
                            enclosed_section = text[section_start:i+1]
                            sections.append(enclosed_section)
                            positions.append(current_pos)
                            current_pos += 1
                            section_start = i + 1
                            in_enclosed_section = False
                            break
                    i += 1

                if depth != 0:
                    orphaned = text[section_start:]
                    if orphaned:  # preserve whitespace-only segments
                        sections.append(orphaned)
                        positions.append(current_pos)
                    break

                continue

            if depth == 0 and angle_depth == 0 and not in_enclosed_section:
                if reorder_mode == "comma" and (char == ',' or char == '\n'):
                    segment = text[section_start:i+1]
                    if segment:  # preserve whitespace-only segments for round-trip fidelity
                        sections.append(segment)
                        positions.append(current_pos)
                        current_pos += 1
                    section_start = i + 1
                elif reorder_mode == "sentence" and char == '\n':
                    segment = text[section_start:i+1]
                    if segment:  # preserve whitespace-only segments for round-trip fidelity
                        sections.append(segment)
                        positions.append(current_pos)
                        current_pos += 1
                    section_start = i + 1
                elif reorder_mode == "paragraph" and char == '\n':
                    j = i + 1
                    while j < len(text) and text[j] == '\r':
                        j += 1

                    k = j
                    while k < len(text) and text[k] in (' ', '\t'):
                        k += 1
                    if k < len(text) and text[k] == '\n':
                        sep_end = k + 1
                        while sep_end < len(text):
                            m = sep_end
                            while m < len(text) and text[m] == '\r':
                                m += 1
                            n = m
                            while n < len(text) and text[n] in (' ', '\t'):
                                n += 1
                            if n < len(text) and text[n] == '\n':
                                sep_end = n + 1
                                continue
                            break

                        segment = text[section_start:sep_end]
                        if segment:  # preserve whitespace-only segments for round-trip fidelity
                            sections.append(segment)
                            positions.append(current_pos)
                            current_pos += 1
                        section_start = sep_end
                        i = sep_end - 1

            i += 1

        if section_start < len(text) and not in_enclosed_section:
            orphaned = text[section_start:]
            if orphaned:  # preserve whitespace-only segments for round-trip fidelity
                sections.append(orphaned)
                positions.append(current_pos)

        return sections, positions
    #endregion

    #region SHUFFLE
    def reorder_completely_random(self, sections: List[str], seed: int) -> List[Tuple[int, str]]:
        """Completely randomize the order of sections.

        Returns a list of (original_index, section) tuples.
        """
        rng = random.Random(seed) if seed != 0 else random.Random()

        reordered = list(enumerate(sections))
        rng.shuffle(reordered)
        return reordered
    #endregion

    #region DISTANCE
    def reorder_distance_constrained(
        self,
        sections: List[str],
        max_distance: int,
        seed: int
    ) -> List[Tuple[int, str]]:
        """Reorder sections with distance constraints.

        Each section can only move up to max_distance positions from its
        original location. When no in-range slot is available for a section
        (greedy crowding), the closest available slot is used so the
        displacement approximates max_distance as closely as the remaining
        slots allow.

        Returns a list of (original_index, section) tuples.
        """
        rng = random.Random(seed) if seed != 0 else random.Random()

        n = len(sections)
        if n <= 1:
            return list(enumerate(sections))

        result: List[Tuple[int, str]] = [None] * n  # type: ignore[assignment]
        available_positions = list(range(n))

        # Shuffle the order in which we process sections
        processing_order = list(range(n))
        rng.shuffle(processing_order)

        for original_idx in processing_order:
            section = sections[original_idx]

            # Calculate valid range for this section
            min_pos = max(0, original_idx - max_distance)
            max_pos = min(n - 1, original_idx + max_distance)

            # Find available positions within the valid range
            valid_positions = [p for p in available_positions if min_pos <= p <= max_pos]

            if not valid_positions:
                # No in-range slot left: pick the closest available position to
                # original_idx (ties broken randomly) so the displacement
                # approximates max_distance as closely as the remaining slots allow.
                valid_positions = sorted(
                    available_positions,
                    key=lambda p: (abs(p - original_idx), rng.random()),
                )
                chosen_pos = valid_positions[0]
            else:
                # Randomly choose from valid positions
                chosen_pos = rng.choice(valid_positions)
            result[chosen_pos] = (original_idx, section)
            available_positions.remove(chosen_pos)

        return result
    #endregion

    #region ENTRY
    def reorder(
        self,
        text: str,
        reorder_mode: str = "comma",
        seed: int = 0,
        completely_random: bool = True,
        distance_constrained: bool = False,
        max_distance: int = 2,
    ) -> Tuple[str, str]:
        """Main processing function that reorders text sections.

        Returns:
            - Reordered text string
            - Section information string
        """
        # Validate reorder_mode (clamp unknown values to "comma")
        if reorder_mode not in self.VALID_REORDER_MODES:
            print(f"[TextReorder] Warning: unknown reorder_mode '{reorder_mode}', falling back to 'comma'")
            reorder_mode = "comma"

        # Parse text into sections (positions unused - original indices are
        # carried through the reorder step via enumerate tuples)
        sections, _positions = self.parse_sections_mode(text, reorder_mode)

        if len(sections) == 0:
            return (text, "No sections found in text.")

        # Create section info report
        info_lines = [
            "=== SECTION ANALYSIS ===",
            f"Total sections found: {len(sections)}",
            f"Split mode: {reorder_mode}",
            "",
            "Original sections:"
        ]

        for i, section in enumerate(sections):
            section_preview = section[:50].replace('\n', ' ')
            if len(section) > 50:
                section_preview += "..."
            section_type = "ENCLOSED" if section.strip().startswith('(') else "ORPHANED"
            info_lines.append(f"  [{i}] {section_type}: {section_preview}")

        info_lines.append("")

        # Reorder sections based on mode. Each branch returns a list of
        # (original_index, section) tuples so duplicate sections can be tracked
        # without an O(n^2) sections.index() lookup.
        if completely_random and not distance_constrained:
            reordered = self.reorder_completely_random(sections, seed)
            info_lines.append("Mode: COMPLETELY RANDOM")
        elif distance_constrained:
            reordered = self.reorder_distance_constrained(sections, max_distance, seed)
            info_lines.append(f"Mode: DISTANCE CONSTRAINED (max distance: {max_distance})")
        else:
            # No reordering
            reordered = list(enumerate(sections))
            info_lines.append("Mode: NO REORDERING (both options disabled)")

        info_lines.append("")
        info_lines.append("Reordered sections:")

        for i, (original_idx, section) in enumerate(reordered):
            section_preview = section[:50].replace('\n', ' ')
            if len(section) > 50:
                section_preview += "..."
            section_type = "ENCLOSED" if section.strip().startswith('(') else "ORPHANED"
            info_lines.append(f"  [{i}] (was [{original_idx}]) {section_type}: {section_preview}")

        section_info = "\n".join(info_lines)

        # Reconstruct text from reordered sections
        reordered_text = "".join(section for _, section in reordered)

        print(f"[TextReorder] Reordered {len(sections)} sections using mode {reorder_mode} and seed {seed}")

        return (reordered_text, section_info)
    #endregion

    #region UI
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "reorder_mode": (["paragraph", "sentence", "comma"], {"default": "comma"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "completely_random": ("BOOLEAN", {"default": True}),
                "distance_constrained": ("BOOLEAN", {"default": False}),
                "max_distance": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
            },
            "optional": {},
            "hidden": {},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("reordered_text", "section_info")
    FUNCTION = "reorder"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Reorders prompt text sections (comma/sentence/paragraph) while preserving parenthesis-enclosed blocks; supports seeded shuffle and distance constraints."
    #endregion

#region REGISTER
NODE_CLASS_MAPPINGS = {
    "illumoraeTextReorderNode": illumoraeTextReorderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeTextReorderNode": "Text Reorder",
}
#endregion
