"""
TITLE::Text Strength Multiplier
DESCRIPTIONSHORT::Multiplies prompt section weights while preserving <lora:...> tags; supports individual caps/minimums and total cap scaling.
VERSION::20260816
IMAGE::comfyui_illumorae_text_strength_multiplier.png
GROUP::Text
GROUPORDER::5
LISTORDER::90
STATUS::working
"""
#region IMPORTS
import re
import uuid
from typing import List, Tuple
#endregion

#region CONST
# Matches an optional trailing ":strength" at the end of a group's inner
# text, e.g. "fruit:1.5" -> content "fruit", strength "1.5".
_STRENGTH_SUFFIX = re.compile(r':\s*([0-9]*\.?[0-9]+)\s*$')
#endregion

#region HELPERS
def _find_weighted_groups(text: str) -> List[Tuple[int, int, str, str]]:
    """Find top-level balanced ``(...)`` groups in ``text``.

    Returns a list of ``(start, end, content, strength_str)`` tuples where
    ``start``/``end`` are the absolute indices of the opening and closing
    parentheses, ``content`` is the inner text with any trailing
    ``:strength`` stripped, and ``strength_str`` is the matched strength
    token (or ``""`` when no explicit strength is present).

    Only top-level groups (paren depth 1) are returned; nested groups are
    left intact inside their parent's ``content`` and are not re-parsed.
    Unbalanced opening parentheses are skipped (the scan resumes without
    emitting a group), so malformed input does not raise.
    """
    groups: List[Tuple[int, int, str, str]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    inner = text[start + 1:i]
                    m = _STRENGTH_SUFFIX.search(inner)
                    if m:
                        strength_str = m.group(1)
                        content = inner[:m.start()].rstrip()
                    else:
                        strength_str = ""
                        content = inner.strip()
                    groups.append((start, i, content, strength_str))
                    start = -1
    return groups


def _rewrite_weighted_groups(
    text: str, transform
) -> Tuple[str, List[Tuple[str, str]]]:
    """Rewrite every top-level weighted group in ``text``.

    ``transform`` is called as ``transform(content, strength_str)`` for each
    group and must return the new inner text to place between the
    parentheses (the caller is responsible for any ``:strength`` suffix).

    Returns the rebuilt string and a list of ``(content, strength_str)``
    tuples in match order (useful for the total-cap pass).
    """
    groups = _find_weighted_groups(text)
    if not groups:
        return text, []
    parts: List[str] = []
    last = 0
    parsed: List[Tuple[str, str]] = []
    for start, end, content, strength_str in groups:
        parts.append(text[last:start])
        parts.append("(" + transform(content, strength_str) + ")")
        parsed.append((content, strength_str))
        last = end + 1
    parts.append(text[last:])
    return "".join(parts), parsed
#endregion

#region CLASS
class illumoraeTextStrengthMultiplierNode:
    """
    A ComfyUI node that processes a prompt's "sections" (i.e. weighted groups or plain text blocks)
    and applies a strength multiplier. It is careful not to alter any LoRA tags (i.e. anything inside
    <lora:...>) while updating sections that already have weight syntax (e.g. "(fruit:1.6)") or wrapping
    plain text sections (which are assumed to have a default strength of 1.0).

    The processing steps are:
      1. Temporarily "protect" any LoRA tags (in the form <lora:name:strength>) so they remain unchanged.
         Paragraphs that contain a LoRA tag are left unwrapped so the restored tag is not nested inside
         weighted syntax.
      2. Process any weighted groups of the form:
             ( some text [ :strength] )
         A depth-aware scanner finds top-level balanced groups, so nested parentheses such as
         "(a (b:1.5):2.0)" are handled correctly (the outer weight is multiplied, the inner group is
         preserved). If a strength is already present, multiply it by the given multiplier. Otherwise,
         assume a base strength of 1.0 and add a strength (1.0 * multiplier). (Individual caps/minimums
         are applied if enabled.)
      3. For any plain text paragraphs that do not contain weighted groups (and that are not LoRA tags),
         wrap the entire paragraph in parentheses and append the multiplier as a strength.
      4. If total-cap enforcement is enabled, scan all weighted groups, compute their sum, and if the sum
         exceeds the total cap then scale each group's strength proportionally. The individual minimum is
         treated as a hard floor and the total cap as a hard ceiling: sections pinned at their minimum are
         fixed there and the remaining sections are rescaled against the remaining budget. If the minimums
         alone meet or exceed the total cap, every section is set to its minimum (the closest feasible
         point).
      5. Finally, restore the LoRA tags.

    The node also accepts options to limit each section's new strength (via an individual cap and optional
    minimum). When both are enabled, the cap is applied first and the minimum second, so a configuration
    with cap < minimum resolves to the minimum for every section.
    """
    #region PROCESS
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
    ) -> tuple[str]:
        #region C_LORA
        # --- STEP 1: Protect any LoRA tags so we don't process them.
        # We'll replace any <lora:...> occurrence with a unique placeholder.
        # A UUID-based prefix avoids collisions with literal "@@LORA_n@@"
        # text a user might have in their prompt.
        lora_pattern = re.compile(r'(<lora:[^>]+>)')
        lora_placeholders: dict[str, str] = {}
        placeholder_prefix = f"@@LORA_{uuid.uuid4().hex}_"
        def lora_replacer(match):
            tag = match.group(1)
            placeholder = f"{placeholder_prefix}{len(lora_placeholders)}@@"
            lora_placeholders[placeholder] = tag
            return placeholder

        text_protected = lora_pattern.sub(lora_replacer, text)
        # Normalize CRLF/CR to LF so paragraph splitting and wrapping do not
        # leave stray "\r" characters in the output (e.g. prompts pasted from
        # Windows editors).
        text_protected = text_protected.replace("\r\n", "\n").replace("\r", "\n")
        #endregion

        #region C_CAPS
        # --- Helper to apply individual cap and minimum.
        # Cap is applied first, minimum second; cap < min resolves to min.
        def apply_caps(value: float) -> float:
            if individual_cap_enabled:
                value = min(value, individual_cap)
            if individual_min_enabled:
                value = max(value, individual_min)
            return value
        #endregion

        #region C_WEIGHTS
        # --- STEP 2: Process weighted groups.
        # A "weighted group" is in the form "( some text [ :strength] )".
        # A depth-aware scanner finds top-level balanced groups so nested
        # parens (e.g. "(a (b:1.5):2.0)") are preserved instead of being
        # truncated at the first ")".
        def replace_weighted(content: str, strength_str: str) -> str:
            if strength_str:
                try:
                    original_strength = float(strength_str)
                except ValueError:
                    original_strength = 1.0
                new_strength = original_strength * multiplier
            else:
                # No explicit strength; assume a base value of 1.0.
                new_strength = 1.0 * multiplier
            new_strength = apply_caps(new_strength)
            # Format strength with 2 decimals. No space before the colon to
            # match the A1111/ComfyUI weighted-text convention "(text:1.5)".
            return f"{content}:{new_strength:.2f}"

        text_processed, _ = _rewrite_weighted_groups(text_protected, replace_weighted)
        #endregion

        #region C_WRAP
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
            # If the paragraph contains a LoRA placeholder, leave it unchanged so
            # the restored <lora:...> tag is not nested inside weighted syntax.
            if placeholder_prefix in paragraph:
                return paragraph
            else:
                # Wrap the entire paragraph. Here we assume a default base strength of 1.0.
                new_strength = apply_caps(1.0 * multiplier)
                return f"({paragraph}:{new_strength:.2f})"
        # Split by double newline (to treat separate paragraphs).
        paragraphs = text_processed.split('\n\n')
        wrapped_paragraphs = [wrap_plain_text(p) for p in paragraphs]
        text_final = "\n\n".join(wrapped_paragraphs)
        #endregion

        #region C_TOTALCAP
        # --- STEP 4: If total cap is enabled, adjust all weighted groups so that the sum
        # of their strength values does not exceed the total cap.
        # Precedence: the individual minimum is a hard floor and the total
        # cap is a hard ceiling. Sections are rescaled proportionally; any
        # section that would fall below its minimum is fixed at the minimum
        # and the remaining sections are rescaled against the remaining
        # budget. If the minimums alone meet or exceed the total cap, every
        # section is set to its minimum (the closest feasible point).
        if total_cap_enabled:
            # Collect (content, strength) for every weighted group, in order.
            found = _find_weighted_groups(text_final)
            groups = [(c, float(s)) for _, _, c, s in found if s]
            total_strength = sum(s for _, s in groups) if groups else 0.0
            if total_strength > total_cap and groups:
                # Iterative rescale that respects the individual minimum.
                final_strengths = [None] * len(groups)
                pending = list(range(len(groups)))
                budget = float(total_cap)
                while pending:
                    pending_sum = sum(groups[i][1] for i in pending)
                    if pending_sum <= 0:
                        # All remaining are zero; assign 0 (clamped by caps).
                        for i in pending:
                            final_strengths[i] = apply_caps(0.0)
                        break
                    factor = budget / pending_sum
                    # Apply the factor; clamp via apply_caps (cap then min).
                    scaled = {i: apply_caps(groups[i][1] * factor) for i in pending}
                    # Find sections pinned at the minimum by the clamp.
                    pinned = []
                    for i in pending:
                        if (individual_min_enabled
                                and scaled[i] <= individual_min + 1e-12):
                            pinned.append(i)
                    if not pinned:
                        for i in pending:
                            final_strengths[i] = scaled[i]
                        break
                    # Fix pinned sections at the minimum, subtract from budget.
                    for i in pinned:
                        final_strengths[i] = individual_min
                        budget -= individual_min
                        pending.remove(i)
                    if budget < 0:
                        # Minimums alone exceed the cap; clamp remaining to min.
                        for i in pending:
                            final_strengths[i] = apply_caps(0.0)
                        break
                # Re-emit each weighted group with its final strength. Only
                # groups that contributed to ``groups`` (those with an
                # explicit strength) are updated; others are left as-is.
                final_iter = iter(final_strengths)

                def adjust_total(content: str, strength_str: str) -> str:
                    if strength_str:
                        new_strength = next(final_iter)
                        return f"{content}:{new_strength:.2f}"
                    return content

                text_final, _ = _rewrite_weighted_groups(text_final, adjust_total)
        #endregion

        #region C_RESTORE
        # --- STEP 5: Restore the LoRA tags.
        for placeholder, tag in lora_placeholders.items():
            text_final = text_final.replace(placeholder, tag)
        #endregion

        return (text_final,)
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
                "individual_min_enabled": ("BOOLEAN", {"default": False}),
                "individual_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.0001, "round": 0.0001}),
                "total_cap_enabled": ("BOOLEAN", {"default": False}),
                "total_cap": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.0001, "round": 0.0001}),
            },
        }

    RETURN_TYPES = ("STRING",)  # The node returns a single modified string.
    RETURN_NAMES = ("modified_text",)
    FUNCTION = "process"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Multiplies prompt section weights while preserving <lora:...> tags; supports individual caps/minimums and total cap scaling."
    #endregion
#endregion

#region REGISTER
# ComfyUI needs to know which classes to load when scanning your .py file.
NODE_CLASS_MAPPINGS = {
    "illumoraeTextStrengthMultiplierNode": illumoraeTextStrengthMultiplierNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeTextStrengthMultiplierNode": "Text Strength Multiplier",
}
#endregion
