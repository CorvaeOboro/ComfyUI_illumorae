"""
TITLE::LoRA Strength Variants
DESCRIPTIONSHORT::Randomizes or highlights LoRA strengths in <lora:name:strength> tags with optional seed control.
VERSION::20260113
IMAGE::comfyui_illumorae_lora_strength_randomize.png
GROUP::Lora
"""
import re
import random
from typing import Dict

class illumoraeLoraRandomizerNode:
    """
    A ComfyUI node that parses LoRA strings, randomizes their strengths, highlights a single LoRA, or passes through unmodified.

    Modes:
        - Randomize: Randomize strengths while adhering to max total and individual strengths.
        - Highlight: Highlight a single LoRA, setting others to a low strength.
        - Pass-through: Leave the input unchanged if both modes are off.
    """

    def __init__(self):
        """
        Initialize node with default strength limits.
        """
        self.total_strength = 1.5
        self.max_individual_strength = 0.9

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

    # ComfyUI expects certain class variables that define how the node behaves
    RETURN_TYPES = ("STRING",)  # We return a single modified string
    RETURN_NAMES = ("modified_text",) # Name for the output
    FUNCTION = "process"       # The method in this class that is called
    CATEGORY = "illumorae"          # Category name for where this node appears
    OUTPUT_NODE = False         # Whether this node can terminate a workflow
    DESCRIPTION = "Randomizes LoRA strengths in a text string."

    def _debug_print(self, debug_prints, *args, **kwargs):
        if debug_prints:
            print(*args, **kwargs)

    def parse_lora_syntax(self, text: str, debug_prints: bool = False) -> Dict[str, float]:
        """
        Parse LoRA strings into a dictionary of {lora_name: strength}.
        """
        lora_pattern = r"<lora:([^:<>]+):([0-9]*\.?[0-9]+)>"
        matches = re.findall(lora_pattern, text)
        parsed_loras = {name.strip(): float(strength) for name, strength in matches}
        self._debug_print(debug_prints, "Parsed LoRA syntax:", parsed_loras)
        return parsed_loras

    def format_lora_syntax(self, loras: Dict[str, float], debug_prints: bool = False) -> str:
        """
        Format a dictionary of {lora_name: strength} back into LoRA syntax.
        Ensure proper formatting even if the dictionary is empty.
        """
        if not loras:
            self._debug_print(debug_prints, "No LoRAs to format. Returning empty string.")
            return ""

        formatted = " ".join(f"<lora:{name}:{strength:.2f}>" for name, strength in loras.items())
        self._debug_print(debug_prints, "Formatted LoRA syntax:", formatted)
        return formatted

    def randomize_strengths(self, loras: Dict[str, float], seed: int, debug_prints: bool = False) -> Dict[str, float]:
        """
        Randomize the strengths of LoRAs while keeping total under total_strength.
        Connect the random seed to the main seed for reproducibility.
        """
        if seed != 0:
            random.seed(seed)

        total_available = self.total_strength
        randomized_loras = {}
        
        # Convert keys to list and shuffle to randomize the order of processing
        lora_names = list(loras.keys())
        random.shuffle(lora_names)
        
        for name in lora_names:
            strength = min(random.uniform(0, self.max_individual_strength), total_available)
            randomized_loras[name] = strength
            total_available -= strength
            if total_available <= 0:
                break

        # Fill remaining LoRAs with 0 strength
        for name in loras.keys():
            if name not in randomized_loras:
                randomized_loras[name] = 0.0

        self._debug_print(debug_prints, "Randomized LoRA strengths:", randomized_loras)
        return randomized_loras

    def highlight_random_lora(self, loras: Dict[str, float], seed: int, debug_prints: bool = False) -> Dict[str, float]:
        """
        Highlight a single LoRA by setting it to a high strength and others to a low strength.
        Connect the random seed to the main seed for reproducibility.
        """
        if seed != 0:
            random.seed(seed)

        if not loras:
            self._debug_print(debug_prints, "No LoRAs to highlight. Returning empty dictionary.")
            return {}

        chosen_lora = random.choice(list(loras.keys()))
        highlighted_loras = {name: 0.01 for name in loras.keys()}
        highlighted_loras[chosen_lora] = 0.9
        self._debug_print(debug_prints, f"Highlighted LoRA: {chosen_lora}", highlighted_loras)
        return highlighted_loras

    def process(self, text: str, randomize: bool = False, highlight: bool = False, seed: int = 0, debug_prints: bool = False, *args, **kwargs) -> tuple:
        """
        Process the input text based on the selected modes.
        """
        self._debug_print(debug_prints, "Input Text:", text)

        if not randomize and not highlight:
            self._debug_print(debug_prints, "Pass-through mode: No changes made.")
            return (text,)  # Pass through if both modes are off

        loras = self.parse_lora_syntax(text, debug_prints=debug_prints)

        if randomize:
            loras = self.randomize_strengths(loras, seed, debug_prints=debug_prints)

        if highlight:
            loras = self.highlight_random_lora(loras, seed, debug_prints=debug_prints)

        final_loras = self.format_lora_syntax(loras, debug_prints=debug_prints)

        if not isinstance(final_loras, str):
            self._debug_print(debug_prints, "Error: Final output is not a string.")
            final_loras = ""

        self._debug_print(debug_prints, f"FINAL LoRA Output: {final_loras}")
        return (final_loras,)  # Return as a tuple

# ComfyUI needs to know which classes to load when scanning your .py file
NODE_CLASS_MAPPINGS = {
    "illumoraeLoraRandomizerNode": illumoraeLoraRandomizerNode,
}

# (Optional) Provide a human-readable display name for your node
NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeLoraRandomizerNode": "LoRA Randomize Strength on Text",
}
