"""
TITLE::Text Truncate Protect
DESCRIPTIONSHORT::Truncates text to a character limit with protection options for words, paragraphs, and enclosures.
VERSION::20260206
IMAGE::comfyui_illumorae_text_truncate_protect.png
GROUP::Text
"""
from typing import Tuple, List, Dict


class illumoraeTextTruncateProtectNode:
    """
    A ComfyUI node that truncates text to a target character limit with configurable protection.
    
    Features:
    - Truncates text based on character limit
    - Protect Words: Only truncates at space/newline boundaries (preserves whole words including hyphenated)
    - Protect Paragraphs: Only truncates at newline boundaries
    - Protect Enclosures: Respects enclosure boundaries: (), {}, <>, []
    - Outputs truncated text and statistics
    """

    def __init__(self):
        """Initialize the node."""
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "character_limit": ("INT", {"default": 500, "min": 1, "max": 100000, "step": 1}),
                "threshold": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
            },
            "optional": {
                "protect_words": ("BOOLEAN", {"default": False}),
                "protect_paragraphs": ("BOOLEAN", {"default": False}),
                "protect_enclosures": ("BOOLEAN", {"default": False}),
                "enclosure_types": (["all", "parentheses", "braces", "brackets", "angle_brackets"], {"default": "all"}),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("truncated_text", "statistics", "final_length", "truncated_chars")
    FUNCTION = "truncate_text"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Truncates text to a character limit with protection options for words, paragraphs, and enclosures."

    def get_enclosure_pairs(self, enclosure_types: str) -> List[Tuple[str, str]]:
        """
        Get the list of enclosure pairs to consider based on the selected type.
        
        Args:
            enclosure_types: Type of enclosures to consider
            
        Returns:
            List of (opening, closing) character pairs
        """
        all_pairs = [
            ('(', ')'),
            ('{', '}'),
            ('[', ']'),
            ('<', '>')
        ]
        
        if enclosure_types == "all":
            return all_pairs
        elif enclosure_types == "parentheses":
            return [('(', ')')]
        elif enclosure_types == "braces":
            return [('{', '}')]
        elif enclosure_types == "brackets":
            return [('[', ']')]
        elif enclosure_types == "angle_brackets":
            return [('<', '>')]
        
        return all_pairs

    def parse_enclosures(self, text: str, enclosure_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """
        Parse text to identify enclosure positions and depths.
        
        Args:
            text: Input text to parse
            enclosure_pairs: List of (opening, closing) character pairs
            
        Returns:
            List of character info dicts with position, char, depth, and enclosure type
        """
        char_info = []
        depth_stacks = {pair: [] for pair in enclosure_pairs}
        opening_chars = {pair[0]: pair for pair in enclosure_pairs}
        closing_chars = {pair[1]: pair for pair in enclosure_pairs}
        
        for i, char in enumerate(text):
            total_depth = sum(len(stack) for stack in depth_stacks.values())
            enclosure_type = None
            is_opening = False
            is_closing = False
            
            if char in opening_chars:
                pair = opening_chars[char]
                depth_stacks[pair].append(i)
                enclosure_type = pair
                is_opening = True
                total_depth = sum(len(stack) for stack in depth_stacks.values())
                
            elif char in closing_chars:
                pair = closing_chars[char]
                if depth_stacks[pair]:
                    depth_stacks[pair].pop()
                    enclosure_type = pair
                    is_closing = True
                    total_depth = sum(len(stack) for stack in depth_stacks.values())
            
            char_info.append({
                'pos': i,
                'char': char,
                'depth': total_depth,
                'enclosure_type': enclosure_type,
                'is_opening': is_opening,
                'is_closing': is_closing
            })
        
        return char_info

    def find_word_boundary(self, text: str, position: int, threshold: int, search_backward: bool = True) -> int:
        """
        Find the nearest word boundary (space or newline) from the given position.
        
        Args:
            text: Input text
            position: Starting position to search from
            threshold: Maximum distance to search
            search_backward: If True, search backward first; otherwise search forward
            
        Returns:
            Position of the word boundary
        """
        if position >= len(text):
            return len(text)
        
        min_pos = max(0, position - threshold)
        max_pos = min(len(text), position + threshold)
        
        if search_backward:
            for pos in range(position, min_pos - 1, -1):
                if pos < len(text) and text[pos] in ' \n\r\t':
                    return pos
            for pos in range(position + 1, max_pos + 1):
                if pos < len(text) and text[pos] in ' \n\r\t':
                    return pos
        else:
            for pos in range(position, max_pos + 1):
                if pos < len(text) and text[pos] in ' \n\r\t':
                    return pos
            for pos in range(position - 1, min_pos - 1, -1):
                if pos < len(text) and text[pos] in ' \n\r\t':
                    return pos
        
        return position

    def find_paragraph_boundary(self, text: str, position: int, threshold: int, search_backward: bool = True) -> int:
        """
        Find the nearest paragraph boundary (newline) from the given position.
        
        Args:
            text: Input text
            position: Starting position to search from
            threshold: Maximum distance to search
            search_backward: If True, search backward first; otherwise search forward
            
        Returns:
            Position of the paragraph boundary
        """
        if position >= len(text):
            return len(text)
        
        min_pos = max(0, position - threshold)
        max_pos = min(len(text), position + threshold)
        
        if search_backward:
            for pos in range(position, min_pos - 1, -1):
                if pos < len(text) and text[pos] in '\n\r':
                    return pos
            for pos in range(position + 1, max_pos + 1):
                if pos < len(text) and text[pos] in '\n\r':
                    return pos
        else:
            for pos in range(position, max_pos + 1):
                if pos < len(text) and text[pos] in '\n\r':
                    return pos
            for pos in range(position - 1, min_pos - 1, -1):
                if pos < len(text) and text[pos] in '\n\r':
                    return pos
        
        return position

    def find_enclosure_safe_point(
        self,
        text: str,
        char_info: List[Dict],
        character_limit: int,
        threshold: int
    ) -> int:
        """
        Find the optimal truncation point that respects enclosure boundaries.
        
        Args:
            text: Input text
            char_info: Parsed character information
            character_limit: Target character limit
            threshold: Allowed deviation from character limit
            
        Returns:
            Position to truncate at 
        """
        if len(text) <= character_limit:
            return len(text)
        
        min_pos = max(0, character_limit - threshold)
        max_pos = min(len(text), character_limit + threshold)
        
        best_pos = character_limit
        best_depth = float('inf')
        
        for pos in range(min_pos, max_pos + 1):
            if pos >= len(char_info):
                break
                
            depth = char_info[pos]['depth']
            
            if depth == 0:
                # If this position is a closing bracket, depth==0 reflects state
                # AFTER the pop, so we must include the bracket itself (pos+1)
                # because truncate_pos is exclusive in text[:truncate_pos].
                candidate_pos = pos + 1 if char_info[pos]['is_closing'] else pos
                distance_from_limit = abs(candidate_pos - character_limit)
                if depth < best_depth or (depth == best_depth and distance_from_limit < abs(best_pos - character_limit)):
                    best_pos = candidate_pos
                    best_depth = depth
        
        if best_depth == 0:
            return best_pos
        
        for pos in range(character_limit, max_pos + 1):
            if pos >= len(char_info):
                break
            
            if char_info[pos]['is_closing']:
                depth_after = char_info[pos]['depth']
                if depth_after < best_depth:
                    best_pos = pos + 1
                    best_depth = depth_after
                    if depth_after == 0:
                        break
        
        if best_depth > 0 and best_pos > character_limit:
            for pos in range(character_limit, min_pos - 1, -1):
                if pos < 0:
                    break
                if pos < len(char_info) and char_info[pos]['depth'] == 0:
                    best_pos = pos
                    break
        
        return min(best_pos, len(text))

    def truncate_text(
        self,
        text: str,
        character_limit: int = 500,
        threshold: int = 50,
        protect_words: bool = False,
        protect_paragraphs: bool = False,
        protect_enclosures: bool = False,
        enclosure_types: str = "all",
        *args,
        **kwargs
    ) -> Tuple[str, str, int, int]:
        """
        Main processing function that truncates text with configurable protection.
        
        Args:
            text: Input text to truncate
            character_limit: Target character limit
            threshold: Allowed deviation from limit for protection
            protect_words: If True, only truncate at word boundaries
            protect_paragraphs: If True, only truncate at paragraph boundaries
            protect_enclosures: If True, respect enclosure boundaries
            enclosure_types: Types of enclosures to respect when protect_enclosures is True
            
        Returns:
            - Truncated text
            - Statistics string
            - Final length
            - Number of characters truncated
        """
        original_length = len(text)
        
        if original_length <= character_limit:
            stats = self._generate_statistics(
                original_length,
                original_length,
                0,
                character_limit,
                threshold,
                protect_words,
                protect_paragraphs,
                protect_enclosures,
                False
            )
            return (text, stats, original_length, 0)
        
        truncate_pos = character_limit
        protection_applied = []
        
        if protect_paragraphs:
            truncate_pos = self.find_paragraph_boundary(text, truncate_pos, threshold)
            protection_applied.append("paragraphs")
        elif protect_words:
            truncate_pos = self.find_word_boundary(text, truncate_pos, threshold)
            protection_applied.append("words")
        
        if protect_enclosures:
            enclosure_pairs = self.get_enclosure_pairs(enclosure_types)
            char_info = self.parse_enclosures(text, enclosure_pairs)
            truncate_pos = self.find_enclosure_safe_point(text, char_info, truncate_pos, threshold)
            protection_applied.append("enclosures")
        
        truncated_text = text[:truncate_pos]
        final_length = len(truncated_text)
        truncated_chars = original_length - final_length
        
        was_adjusted = abs(final_length - character_limit) > 0 and len(protection_applied) > 0
        
        stats = self._generate_statistics(
            original_length,
            final_length,
            truncated_chars,
            character_limit,
            threshold,
            protect_words,
            protect_paragraphs,
            protect_enclosures,
            was_adjusted
        )
        
        protection_str = ", ".join(protection_applied) if protection_applied else "none"
        print(f"[TextTruncateProtect] Truncated from {original_length} to {final_length} chars ({truncated_chars} removed), protection: {protection_str}")
        
        return (truncated_text, stats, final_length, truncated_chars)

    def _generate_statistics(
        self,
        original_length: int,
        final_length: int,
        truncated_chars: int,
        character_limit: int,
        threshold: int,
        protect_words: bool,
        protect_paragraphs: bool,
        protect_enclosures: bool,
        was_adjusted: bool
    ) -> str:
        """Generate statistics report string."""
        protection_list = []
        if protect_words:
            protection_list.append("Words")
        if protect_paragraphs:
            protection_list.append("Paragraphs")
        if protect_enclosures:
            protection_list.append("Enclosures")
        protection_str = ", ".join(protection_list) if protection_list else "None"
        
        stats_lines = [
            "=== TRUNCATION STATISTICS ===",
            f"Original length: {original_length} characters",
            f"Final length: {final_length} characters",
            f"Truncated: {truncated_chars} characters",
            f"Character target: {character_limit}",
            f"Threshold: ±{threshold}",
            f"Protection: {protection_str}",
            ""
        ]
        
        if truncated_chars > 0:
            percentage = (truncated_chars / original_length) * 100
            stats_lines.append(f"Removed: {percentage:.1f}% of original text")
            
            if was_adjusted:
                deviation = final_length - character_limit
                if deviation > 0:
                    stats_lines.append(f"Extended by: {deviation} characters for protection")
                elif deviation < 0:
                    stats_lines.append(f"Shortened by: {abs(deviation)} characters for protection")
        else:
            stats_lines.append("No truncation needed - text within limit")
        
        return "\n".join(stats_lines)


NODE_CLASS_MAPPINGS = {
    "illumoraeTextTruncateProtectNode": illumoraeTextTruncateProtectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeTextTruncateProtectNode": "Text Truncate Protect",
}
