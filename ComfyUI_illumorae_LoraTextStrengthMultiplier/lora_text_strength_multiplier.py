"""
TITLE::LoRA Strength Multiplier on Text
DESCRIPTIONSHORT::Parses <lora:name:unet[:clip]> tags, applies a multiplier with optional individual and total caps, preserving surrounding text.
VERSION::20260816
IMAGE::comfyui_illumorae_lora_strength_multiplier.png
GROUP::Lora
GROUPORDER::4
LISTORDER::1
STATUS::working
"""
#region IMPORTS
import re
#endregion

#region CONFIG
# LoRA tag syntax: <lora:name:unet[:clip]>.
# Strengths accept an optional leading sign and a trailing dot (e.g. 1., .5, -0.5).
_LORA_NUM = r"-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)"
_LORA_PATTERN = re.compile(rf"<lora:([^:<>]+):({_LORA_NUM})(?::({_LORA_NUM}))?>")
#endregion


#region CLASS
class illumoraeLoRAStrengthMultiplierOnTextNode:
    """
    A ComfyUI node that applies a multiplier to LoRA strengths and optionally enforces hard caps on:
        - Individual LoRA strengths.
        - The total combined strength of all LoRAs.

    Supported tag syntax: <lora:name:unet> and <lora:name:unet:clip> (separate unet/clip
    strengths). Negative strengths are accepted. All non-LoRA text in the input is
    preserved; only the matched tags are rewritten in place.

    Workflow:
        1. Find every <lora:name:unet[:clip]> tag in the input text.
        2. Multiply each strength (unet and clip) by the provided multiplier.
        3. If individual cap enforcement is enabled, clamp each strength to the specified
           maximum (upper bound only; negative strengths are unaffected).
        4. If total cap enforcement is enabled, and the sum of all strengths exceeds the
           maximum, scale all strengths proportionally so the total matches the cap.
        5. Rebuild the text with each tag replaced by its rewritten form; surrounding
           text is left untouched.

    Notes:
        - A cap of 0.0 with enforcement enabled zeros out all strengths (individual cap
          clamps each value to 0; total cap scales by 0/total = 0).
        - The individual cap is an upper bound, so negative strengths pass through unchanged
          when the cap is enabled.
    """

    #region CORE
    def process(
        self,
        text: str,
        multiplier: float,
        individual_cap_enabled: bool,
        individual_cap: float,
        total_cap_enabled: bool,
        total_cap: float,
        debug_prints: bool = False,
    ) -> tuple[str]:
        """
        Process the input text by applying the multiplier and enforcing the specified caps.
        LoRA tags are rewritten in place; all surrounding text is preserved.
        """
        self._debug_print(debug_prints, "Input Text:", text)

        #region C_GATE
        # No <lora:...> tags in the input: nothing to rewrite, return unchanged.
        matches = list(_LORA_PATTERN.finditer(text))
        if not matches:
            self._debug_print(debug_prints, "No LoRA tags found; returning text unchanged.")
            return (text,)
        #endregion

        #region C_MULT
        # Apply the multiplier to each LoRA strength (unet and clip).
        finals = []
        for m in matches:
            unet = float(m.group(2)) * multiplier
            clip = float(m.group(3)) * multiplier if m.group(3) is not None else None
            finals.append([unet, clip])
        self._debug_print(debug_prints, "After applying multiplier:", finals)
        #endregion

        #region C_CAPS
        # Enforce individual strength cap if enabled (upper bound; negatives are unaffected).
        if individual_cap_enabled:
            for pair in finals:
                pair[0] = min(pair[0], individual_cap)
                if pair[1] is not None:
                    pair[1] = min(pair[1], individual_cap)
            self._debug_print(debug_prints, "After enforcing individual cap:", finals)

        # Enforce total strength cap if enabled. The total sums unet + clip across all
        # tags; when it exceeds the cap, every strength is scaled down proportionally.
        if total_cap_enabled:
            total_strength = sum(u + (c or 0.0) for u, c in finals)
            if total_strength > total_cap and total_strength > 0:
                scale_factor = total_cap / total_strength
                for pair in finals:
                    pair[0] *= scale_factor
                    if pair[1] is not None:
                        pair[1] *= scale_factor
                self._debug_print(debug_prints, "After enforcing total cap, scale factor applied:", scale_factor)
                self._debug_print(debug_prints, "LoRAs after total cap enforcement:", finals)
            else:
                self._debug_print(debug_prints, "Total strength within cap, no scaling needed:", total_strength)
        #endregion

        #region C_REWRITE
        # Rebuild the text in place, replacing each matched tag with its rewritten form
        # and preserving all surrounding text.
        out = []
        last = 0
        for m, (u, c) in zip(matches, finals):
            out.append(text[last:m.start()])
            out.append(self.format_lora_tag(m.group(1), u, c, debug_prints=debug_prints))
            last = m.end()
        out.append(text[last:])
        final_text = "".join(out)
        self._debug_print(debug_prints, "Final output text:", final_text)
        return (final_text,)
        #endregion
    #endregion

    #region C_PARSE
    def parse_lora_syntax(self, text: str, debug_prints: bool = False) -> list:
        """
        Parse LoRA tags from the input text.
        Expected syntax: <lora:name:unet> or <lora:name:unet:clip>.
        Returns a list of (name, unet_strength, clip_strength) tuples in order of
        appearance; clip_strength is None when only one strength is given. Duplicate
        names are preserved as separate records.
        """
        records = []
        for m in _LORA_PATTERN.finditer(text):
            name = m.group(1)
            unet = float(m.group(2))
            clip = float(m.group(3)) if m.group(3) is not None else None
            records.append((name, unet, clip))
        self._debug_print(debug_prints, "Parsed LoRA syntax:", records)
        return records
    #endregion

    #region FORMAT
    def format_lora_tag(self, name: str, unet: float, clip, debug_prints: bool = False) -> str:
        """
        Format a single LoRA tag. Emits the two-strength form only when clip is not None.
        """
        if clip is None:
            tag = f"<lora:{name}:{unet:.4f}>"
        else:
            tag = f"<lora:{name}:{unet:.4f}:{clip:.4f}>"
        self._debug_print(debug_prints, "Formatted LoRA tag:", tag)
        return tag
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
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.0001, "round": 0.0001}),
                "individual_cap_enabled": ("BOOLEAN", {"default": False}),
                "individual_cap": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.0001, "round": 0.0001}),
                "total_cap_enabled": ("BOOLEAN", {"default": False}),
                "total_cap": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.0001, "round": 0.0001}),
            },
            "optional": {
                "debug_prints": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)  # The node returns a single modified string.
    RETURN_NAMES = ("modified_text",)
    FUNCTION = "process"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Parses <lora:name:unet[:clip]> tags, applies a multiplier with optional individual and total caps, preserving surrounding text."
    #endregion
#endregion


#region REG
# ComfyUI needs to know which classes to load when scanning your .py file.
NODE_CLASS_MAPPINGS = {
    "illumoraeLoRAStrengthMultiplierOnTextNode": illumoraeLoRAStrengthMultiplierOnTextNode,
}

# (Optional) Provide a human-readable display name for your node.
NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeLoRAStrengthMultiplierOnTextNode": "LoRA Strength Multiplier on Text",
}
#endregion
