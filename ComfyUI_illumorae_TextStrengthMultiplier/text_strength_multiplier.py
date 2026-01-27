"""
TITLE::Text Strength Multiplier
DESCRIPTIONSHORT::Multiplies prompt section weights while preserving <lora:...> tags; supports individual caps/minimums and total cap scaling.
VERSION::20260113
IMAGE::comfyui_illumorae_text_strength_multiplier.png
GROUP::Text
"""
import re
from typing import Dict

class illumoraeTextStrengthMultiplierNode:
    """
    A ComfyUI node that processes a prompt’s “sections” (i.e. weighted groups or plain text blocks)
    and applies a strength multiplier. It is careful not to alter any LoRA tags (i.e. anything inside
    <lora:…>) while updating sections that already have weight syntax (e.g. "( fruit:1.6)") or wrapping
    plain text sections (which are assumed to have a default strength of 1.0).
    
    The processing steps are:
      1. Temporarily “protect” any LoRA tags (in the form <lora:name:strength>) so they remain unchanged.
      2. Process any weighted groups of the form:
             ( some text [ :strength] )
         If a strength is already present, multiply it by the given multiplier.
         Otherwise, assume a base strength of 1.0 and add a strength (1.0 * multiplier).
         (Individual caps/minimums are applied if enabled.)
      3. For any plain text paragraphs that do not contain weighted groups (and that are not LoRA tags),
         wrap the entire paragraph in parentheses and append the multiplier as a strength.
      4. If total–cap enforcement is enabled, scan all weighted groups, compute their sum, and if the sum
         exceeds the total cap then scale each group’s strength proportionally.
      5. Finally, restore the LoRA tags.
    
    The node also accepts options to limit each section’s new strength (via an individual cap and optional
    minimum).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.0001, "round": 0.0001}),
                "individual_cap_enabled": ("BOOLEAN", {"default": False}),
                "individual_cap": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.0001, "round": 0.0001}),
                "individual_min_enabled": ("BOOLEAN", {"default": False}),
                "individual_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.0001, "round": 0.0001}),
                "total_cap_enabled": ("BOOLEAN", {"default": False}),
                "total_cap": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.0001, "round": 0.0001}),
            },
            "optional": {},
            "hidden": {},
        }

    RETURN_TYPES = ("STRING",)  # The node returns a single modified string.
    RETURN_NAMES = ("modified_text",)
    FUNCTION = "process"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Multiplies prompt section weights while preserving <lora:...> tags; supports individual caps/minimums and total cap scaling."

    def process(
        self,
        text: str,
        multiplier: float,
        individual_cap_enabled: bool,
        individual_cap: float,
        individual_min_enabled: bool,
        individual_min: float,
        total_cap_enabled: bool,
        total_cap: float,
        *args,
        **kwargs,
    ) -> tuple:
        # --- STEP 1: Protect any LoRA tags so we don’t process them.
        # We'll replace any <lora:...> occurrence with a unique placeholder.
        lora_pattern = re.compile(r'(<lora:[^>]+>)')
        lora_placeholders: Dict[str, str] = {}
        def lora_replacer(match):
            tag = match.group(1)
            placeholder = f"@@LORA_{len(lora_placeholders)}@@"
            lora_placeholders[placeholder] = tag
            return placeholder

        text_protected = lora_pattern.sub(lora_replacer, text)

        # --- Helper to apply individual cap and minimum.
        def apply_caps(value: float) -> float:
            if individual_cap_enabled:
                value = min(value, individual_cap)
            if individual_min_enabled:
                value = max(value, individual_min)
            return value

        # --- STEP 2: Process weighted groups.
        # A “weighted group” is assumed to be in the form:
        #    ( some text [ :strength] )
        # We capture the inner text and (optional) strength.
        weighted_pattern = re.compile(
            r'\(\s*(.*?)\s*(?::\s*([0-9]*\.?[0-9]+))?\s*\)'
        )
        def replace_weighted(match):
            content = match.group(1)
            strength_str = match.group(2)
            if strength_str:
                try:
                    original_strength = float(strength_str)
                except Exception:
                    original_strength = 1.0
                new_strength = original_strength * multiplier
            else:
                # No explicit strength; assume a base value of 1.0.
                new_strength = 1.0 * multiplier
            new_strength = apply_caps(new_strength)
            # Format strength with 2 decimals.
            return f"({content} :{new_strength:.2f})"

        text_processed = weighted_pattern.sub(replace_weighted, text_protected)

        # --- STEP 3: For any plain text paragraphs (separated by double-newlines)
        # that do not already contain a weighted group, wrap the entire paragraph.
        # (This makes sure sections that originally had no weight are given one.)
        def wrap_plain_text(paragraph: str) -> str:
            if paragraph.strip() == "":
                return paragraph
            # If the paragraph already contains a weighted group (i.e. a "(" with a matching ")"),
            # then we leave it unchanged.
            if re.search(r'\([^)]*\)', paragraph):
                return paragraph
            else:
                # Wrap the entire paragraph. Here we assume a default base strength of 1.0.
                new_strength = apply_caps(1.0 * multiplier)
                return f"({paragraph} :{new_strength:.2f})"
        # Split by double newline (to treat separate paragraphs).
        paragraphs = text_processed.split('\n\n')
        wrapped_paragraphs = [wrap_plain_text(p) for p in paragraphs]
        text_final = "\n\n".join(wrapped_paragraphs)

        # --- STEP 4: If total cap is enabled, adjust all weighted groups so that the sum
        # of their strength values does not exceed the total cap.
        if total_cap_enabled:
            # Find all occurrences of " :<number>)" inside weighted groups.
            strength_num_pattern = re.compile(r':\s*([0-9]*\.?[0-9]+)\)')
            all_strengths = strength_num_pattern.findall(text_final)
            total_strength = sum(float(s) for s in all_strengths) if all_strengths else 0.0
            if total_strength > total_cap and total_strength > 0:
                factor = total_cap / total_strength
                # Adjust each weighted group.
                def adjust_total(match):
                    content = match.group(1)
                    strength_str = match.group(2)
                    if strength_str:
                        new_strength = float(strength_str) * factor
                        new_strength = apply_caps(new_strength)
                        return f"({content} :{new_strength:.2f})"
                    else:
                        return match.group(0)
                text_final = weighted_pattern.sub(adjust_total, text_final)

        # --- STEP 5: Restore the LoRA tags.
        for placeholder, tag in lora_placeholders.items():
            text_final = text_final.replace(placeholder, tag)


        return (text_final,)

# ComfyUI needs to know which classes to load when scanning your .py file.
NODE_CLASS_MAPPINGS = {
    "illumoraeTextStrengthMultiplierNode": illumoraeTextStrengthMultiplierNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeTextStrengthMultiplierNode": "Text Strength Multiplier",
}
