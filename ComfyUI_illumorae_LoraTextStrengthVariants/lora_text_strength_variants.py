"""
TITLE::LoRA Strength Variants
DESCRIPTIONSHORT::Randomizes or highlights LoRA strengths in <lora:name:strength> tags with optional seed control.
VERSION::20260816
IMAGE::comfyui_illumorae_lora_strength_randomize.png
GROUP::Lora
GROUPORDER::4
LISTORDER::2
STATUS::working
"""
#region IMPORTS
import re
import random
from typing import Dict, List
#endregion


#region CLASS
class illumoraeLoRARandomizeStrengthOnTextNode:
    """
    A ComfyUI node that parses LoRA strings, randomizes their strengths, highlights a single LoRA, or passes through unmodified.

    Modes:
        - Randomize: Randomize strengths while adhering to max total and individual strengths.
        - Highlight: Highlight a single LoRA, setting others to a low strength.
        - Pass-through: Leave the input unchanged if both modes are off.

    ``randomize`` and ``highlight`` are mutually exclusive in effect: when both
    are enabled, ``randomize`` takes precedence and ``highlight`` is skipped.

    Surrounding prompt text is preserved; only matched ``<lora:...>`` tags are
    rewritten in place.

    The total-strength budget is an approximate upper bound: each draw is
    capped by the remaining budget, so the realized sum is typically less than
    ``TOTAL_STRENGTH`` rather than equal to it. LoRAs that receive no budget
    are emitted at ``0.00`` (present but disabled) rather than removed.
    """

    #region CONFIG
    # Strength configuration (class-level constants; not exposed as inputs).
    TOTAL_STRENGTH = 1.5
    MAX_INDIVIDUAL_STRENGTH = 0.9
    HIGHLIGHT_STRENGTH = 0.9
    DIM_STRENGTH = 0.01

    # Matches <lora:name:strength>. Strength accepts N, N.NN, .NN, or N. (trailing dot).
    _LORA_PATTERN = re.compile(r"<lora:([^:<>]+):([0-9]+\.?[0-9]*|\.[0-9]+)>")
    #endregion

    #region CORE
    def process(self, text: str, randomize: bool = False, highlight: bool = False, seed: int = 0, debug_prints: bool = False) -> tuple[str]:
        """
        Process the input text based on the selected modes.

        ``randomize`` and ``highlight`` are mutually exclusive: when both are
        enabled, ``randomize`` takes precedence. Surrounding prompt text is
        preserved; only the matched ``<lora:...>`` tags are rewritten.
        """
        self._debug_print(debug_prints, "Input Text:", text)

        #region C_GATE
        # Pass-through when both modes are off, or when no <lora:...> tags are
        # present in the input (nothing to rewrite).
        if not randomize and not highlight:
            self._debug_print(debug_prints, "Pass-through mode: No changes made.")
            return (text,)  # Pass through if both modes are off

        matches = list(self._LORA_PATTERN.finditer(text))
        if not matches:
            self._debug_print(debug_prints, "No <lora:...> tags found; returning input unchanged.")
            return (text,)
        #endregion

        #region C_DISPATCH
        # Build a local RNG (does not perturb the global random module) and
        # dispatch to the selected transform. randomize wins over highlight
        # when both flags are on.
        names = [m.group(1) for m in matches]
        rng = random.Random(seed)

        if randomize:
            strengths = self.randomize_strengths(names, rng, debug_prints=debug_prints)
        else:  # highlight
            strengths = self.highlight_random_lora(names, rng, debug_prints=debug_prints)
        #endregion

        #region C_REWRITE
        # Rewrite each matched tag in place, preserving all surrounding text.
        def _replace(match: re.Match) -> str:
            name = match.group(1)
            return f"<lora:{name}:{strengths[name]:.2f}>"

        final_text = self._LORA_PATTERN.sub(_replace, text)
        self._debug_print(debug_prints, f"FINAL LoRA Output: {final_text}")
        return (final_text,)  # Return as a tuple
        #endregion
    #endregion

    #region C_PARSE
    def parse_lora_syntax(self, text: str, debug_prints: bool = False) -> Dict[str, float]:
        """
        Parse LoRA strings into a dictionary of {lora_name: strength}.

        Malformed ``<lora:...>`` tags that do not match the strength pattern
        are skipped; when ``debug_prints`` is on, any ``<lora:`` substring that
        was not matched is logged so users notice malformed tags.
        """
        parsed_loras = {m.group(1).strip(): float(m.group(2)) for m in self._LORA_PATTERN.finditer(text)}
        self._debug_print(debug_prints, "Parsed LoRA syntax:", parsed_loras)

        if debug_prints:
            # Report any <lora:...> occurrences that the strict pattern did not consume.
            loose = re.findall(r"<lora:[^<>]*>", text)
            matched_tags = self._LORA_PATTERN.findall(text)
            if len(loose) != len(matched_tags):
                self._debug_print(debug_prints, "Skipped malformed <lora:...> tags:", loose)

        return parsed_loras
    #endregion

    #region C_RAND
    def randomize_strengths(self, names: List[str], rng: random.Random, debug_prints: bool = False) -> Dict[str, float]:
        """
        Randomize the strengths of LoRAs while keeping the sum under TOTAL_STRENGTH.

        ``rng`` is a local ``random.Random`` instance so the global RNG is not
        perturbed. The budget is an approximate upper bound; the realized sum
        is typically less than ``TOTAL_STRENGTH``. LoRAs that receive no
        budget are emitted at ``0.00``.
        """
        total_available = self.TOTAL_STRENGTH
        randomized_loras: Dict[str, float] = {}

        # Shuffle so budget allocation order is unbiased across calls.
        order = list(names)
        rng.shuffle(order)

        for name in order:
            strength = min(rng.uniform(0, self.MAX_INDIVIDUAL_STRENGTH), total_available)
            randomized_loras[name] = strength
            total_available -= strength
            if total_available <= 0:
                break

        # LoRAs that received no budget are emitted at 0.00 (present but disabled).
        for name in names:
            if name not in randomized_loras:
                randomized_loras[name] = 0.0

        self._debug_print(debug_prints, "Randomized LoRA strengths:", randomized_loras)
        return randomized_loras
    #endregion

    #region C_HIGH
    def highlight_random_lora(self, names: List[str], rng: random.Random, debug_prints: bool = False) -> Dict[str, float]:
        """
        Highlight a single LoRA at HIGHLIGHT_STRENGTH and dim the rest to DIM_STRENGTH.

        ``rng`` is a local ``random.Random`` instance so the global RNG is not
        perturbed. Returns an empty dict when ``names`` is empty.
        """
        if not names:
            self._debug_print(debug_prints, "No LoRAs to highlight. Returning empty dictionary.")
            return {}

        chosen_lora = rng.choice(names)
        highlighted_loras = {name: self.DIM_STRENGTH for name in names}
        highlighted_loras[chosen_lora] = self.HIGHLIGHT_STRENGTH
        self._debug_print(debug_prints, f"Highlighted LoRA: {chosen_lora}", highlighted_loras)
        return highlighted_loras
    #endregion

    #region FORMAT
    def format_lora_syntax(self, loras: Dict[str, float], debug_prints: bool = False) -> str:
        """
        Format a dictionary of {lora_name: strength} back into LoRA syntax.
        Returns an empty string when the dictionary is empty.

        Standalone helper: ``process`` rewrites tags in place via ``re.sub``
        and does not call this method. Kept for callers that build a tag
        string from a parsed dict.
        """
        if not loras:
            self._debug_print(debug_prints, "No LoRAs to format. Returning empty string.")
            return ""

        formatted = " ".join(f"<lora:{name}:{strength:.2f}>" for name, strength in loras.items())
        self._debug_print(debug_prints, "Formatted LoRA syntax:", formatted)
        return formatted
    #endregion

    #region UTIL
    def _debug_print(self, debug_prints, *args, **kwargs):
        if debug_prints:
            print(*args, **kwargs)
    #endregion

    #region UI
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "randomize": ("BOOLEAN", {"default": False}),
                "highlight": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0}),
            },
            "optional": {
                "debug_prints": ("BOOLEAN", {"default": False}),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("STRING",)  # We return a single modified string
    RETURN_NAMES = ("modified_text",)  # Name for the output
    FUNCTION = "process"  # The method in this class that is called
    CATEGORY = "illumorae"  # Category name for where this node appears
    DESCRIPTION = "Randomizes or highlights LoRA strengths in <lora:name:strength> tags with optional seed control."
    #endregion
#endregion


#region REGISTER
# ComfyUI needs to know which classes to load when scanning your .py file
NODE_CLASS_MAPPINGS = {
    "illumoraeLoRARandomizeStrengthOnTextNode": illumoraeLoRARandomizeStrengthOnTextNode,
}

# (Optional) Provide a human-readable display name for your node
NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeLoRARandomizeStrengthOnTextNode": "LoRA Randomize Strength on Text",
}
#endregion
