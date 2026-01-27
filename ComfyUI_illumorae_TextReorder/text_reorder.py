"""
TITLE::Text Reorder
DESCRIPTIONSHORT::Reorders prompt text sections (comma/sentence/paragraph) while preserving parenthesis-enclosed blocks; supports seeded shuffle and distance constraints.
VERSION::20260127
IMAGE::comfyui_illumorae_text_reorder.png
GROUP::Text
"""
import re
import random
from typing import List, Tuple

class illumoraeTextReorderNode:
    """
    A ComfyUI node that reorders text sections based on parenthesis enclosures.
    
    Features:
    - Parses text into sections (enclosed in parenthesis or orphaned text between enclosures)
    - Randomizes section order based on seed
    - Optional distance constraint to limit how far sections can move from original position
    - Preserves section content and structure
    """

    def __init__(self):
        """Initialize the text reorder node."""
        pass

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

    def parse_sections(self, text: str) -> Tuple[List[str], List[int]]:
        """
        Parse text into sections based on parenthesis enclosures.
        
        Sections are:
        1. Text enclosed in parenthesis (including the parenthesis)
        2. Orphaned text between enclosed sections (text not in parenthesis)
        
        Returns:
            - List of section strings
            - List of original positions (for tracking)
        """
        sections = []
        positions = []
        current_pos = 0
        i = 0
        depth = 0
        section_start = 0
        in_enclosed_section = False
        
        while i < len(text):
            char = text[i]
            
            if char == '(' and depth == 0:
                # Start of a new enclosed section
                # First, save any orphaned text before this
                if i > section_start:
                    orphaned = text[section_start:i]
                    if orphaned.strip():  # Only add non-empty orphaned sections
                        sections.append(orphaned)
                        positions.append(current_pos)
                        current_pos += 1
                
                # Now start tracking the enclosed section
                section_start = i
                in_enclosed_section = True
                depth = 1
                i += 1
                
                # Find the matching closing parenthesis
                while i < len(text):
                    if text[i] == '(':
                        depth += 1
                    elif text[i] == ')':
                        depth -= 1
                        if depth == 0:
                            # Found matching close
                            enclosed_section = text[section_start:i+1]
                            sections.append(enclosed_section)
                            positions.append(current_pos)
                            current_pos += 1
                            section_start = i + 1
                            in_enclosed_section = False
                            break
                    i += 1
                
                if depth != 0:
                    # Unclosed parenthesis - treat rest as orphaned
                    orphaned = text[section_start:]
                    if orphaned.strip():
                        sections.append(orphaned)
                        positions.append(current_pos)
                    break
            else:
                i += 1
        
        # Handle any remaining orphaned text at the end
        if section_start < len(text) and not in_enclosed_section:
            orphaned = text[section_start:]
            if orphaned.strip():
                sections.append(orphaned)
                positions.append(current_pos)
        
        return sections, positions

    def reorder_completely_random(self, sections: List[str], seed: int) -> List[str]:
        """
        Completely randomize the order of sections.
        
        Args:
            sections: List of text sections
            seed: Random seed for reproducibility
            
        Returns:
            Reordered list of sections
        """
        if seed != 0:
            random.seed(seed)
        
        reordered = sections.copy()
        random.shuffle(reordered)
        return reordered

    def reorder_distance_constrained(
        self, 
        sections: List[str], 
        positions: List[int], 
        max_distance: int, 
        seed: int
    ) -> List[str]:
        """
        Reorder sections with distance constraints.
        Each section can only move up to max_distance positions from its original location.
        
        Args:
            sections: List of text sections
            positions: Original positions of sections
            max_distance: Maximum distance a section can move
            seed: Random seed for reproducibility
            
        Returns:
            Reordered list of sections
        """
        if seed != 0:
            random.seed(seed)
        
        n = len(sections)
        if n <= 1:
            return sections
        
        # Create a list of (section, original_index) tuples
        indexed_sections = list(enumerate(sections))
        result = [None] * n
        available_positions = list(range(n))
        
        # Shuffle the order in which we process sections
        processing_order = list(range(n))
        random.shuffle(processing_order)
        
        for original_idx in processing_order:
            section = sections[original_idx]
            
            # Calculate valid range for this section
            min_pos = max(0, original_idx - max_distance)
            max_pos = min(n - 1, original_idx + max_distance)
            
            # Find available positions within the valid range
            valid_positions = [p for p in available_positions if min_pos <= p <= max_pos]
            
            if not valid_positions:
                # If no valid positions, find the closest available position
                valid_positions = available_positions
            
            # Randomly choose from valid positions
            chosen_pos = random.choice(valid_positions)
            result[chosen_pos] = section
            available_positions.remove(chosen_pos)
        
        return result

    def parse_sections_mode(self, text: str, reorder_mode: str) -> Tuple[List[str], List[int]]:
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
                angle_depth -= 1

            if char == '(' and depth == 0 and angle_depth == 0:
                if i > section_start:
                    orphaned = text[section_start:i]
                    if orphaned.strip():
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
                        angle_depth -= 1
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
                    if orphaned.strip():
                        sections.append(orphaned)
                        positions.append(current_pos)
                    break

                continue

            if depth == 0 and angle_depth == 0 and not in_enclosed_section:
                if reorder_mode == "comma" and (char == ',' or char == '\n'):
                    segment = text[section_start:i+1]
                    if segment.strip():
                        sections.append(segment)
                        positions.append(current_pos)
                        current_pos += 1
                    section_start = i + 1
                elif reorder_mode == "sentence" and char == '\n':
                    segment = text[section_start:i+1]
                    if segment.strip():
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
                        if segment.strip():
                            sections.append(segment)
                            positions.append(current_pos)
                            current_pos += 1
                        section_start = sep_end
                        i = sep_end - 1

            i += 1

        if section_start < len(text) and not in_enclosed_section:
            orphaned = text[section_start:]
            if orphaned.strip():
                sections.append(orphaned)
                positions.append(current_pos)

        return sections, positions

    def reorder(
        self,
        text: str,
        reorder_mode: str = "comma",
        seed: int = 0,
        completely_random: bool = True,
        distance_constrained: bool = False,
        max_distance: int = 2,
        *args,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Main processing function that reorders text sections.
        
        Returns:
            - Reordered text string
            - Section information string
        """
        
        # Parse text into sections
        sections, positions = self.parse_sections_mode(text, reorder_mode)
        
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
        
        # Reorder sections based on mode
        if completely_random and not distance_constrained:
            reordered_sections = self.reorder_completely_random(sections, seed)
            info_lines.append("Mode: COMPLETELY RANDOM")
        elif distance_constrained:
            reordered_sections = self.reorder_distance_constrained(sections, positions, max_distance, seed)
            info_lines.append(f"Mode: DISTANCE CONSTRAINED (max distance: {max_distance})")
        else:
            # No reordering
            reordered_sections = sections
            info_lines.append("Mode: NO REORDERING (both options disabled)")
        
        info_lines.append("")
        info_lines.append("Reordered sections:")
        
        for i, section in enumerate(reordered_sections):
            section_preview = section[:50].replace('\n', ' ')
            if len(section) > 50:
                section_preview += "..."
            
            # Find original index
            original_idx = sections.index(section) if section in sections else -1
            section_type = "ENCLOSED" if section.strip().startswith('(') else "ORPHANED"
            info_lines.append(f"  [{i}] (was [{original_idx}]) {section_type}: {section_preview}")
        
        section_info = "\n".join(info_lines)
        
        # Reconstruct text from reordered sections
        reordered_text = "".join(reordered_sections)
        
        print(f"[TextReorder] Reordered {len(sections)} sections using mode {reorder_mode} and seed {seed}")
        
        return (reordered_text, section_info)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "illumoraeTextReorderNode": illumoraeTextReorderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeTextReorderNode": "Text Reorder",
}
