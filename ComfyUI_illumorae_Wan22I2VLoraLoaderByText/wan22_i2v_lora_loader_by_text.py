"""
illumorae Wan2.2 I2V LoRA Loader By Text - a ComfyUI custom_node

TITLE::Wan2.2 I2V Lora Loader By Text
DESCRIPTIONSHORT::Loads Wan2.2 I2V LoRA models (HIGH or LOW) by fuzzy text string matching from lora syntax tags.
VERSION::20260414
GROUP::Lora

MATCHING RULES:
  - Searches for I2V LoRA files matching the input lora name
  - Filters by HIGH or LOW variant based on node type
  - Minimum score of 90 required to accept a match

MODEL TYPE MODE (toggle on the node):
  - I2V_ONLY (default): Accept files with I2V or no T2V/I2V marker. REJECT files with T2V only.
  - T2V_ONLY: Accept files with T2V or no T2V/I2V marker. REJECT files with I2V only.
  - INCLUSIVE: Accept files with BOTH T2V and I2V in name, or NEITHER. Reject files with only one.
  - NONE: No model type filtering at all. Accept any file regardless of T2V/I2V markers.
  
  Skipped files due to model type are shown in lora_info output for accurate feedback.

HIGH/LOW KEYWORDS (using separator-aware matching):
  - HIGH variants: highnoise, highres, highfreq, high_noise, high, hn (as whole words)
  - LOW variants: lownoise, lowres, lowfreq, low_noise, low, ln (as whole words)

DEDUPLICATION:
  - Tracks which LoRA files have been applied
  - If a LoRA file is already applied, it will be SKIPPED (not applied twice)

NOTES: currently this I2V nodes are separate from the T2V lora node in order to separate the loading of low and high , previously i2v process was loading both and OOM
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import folder_paths
import nodes


def norm(s):
    """Normalize path separators and case."""
    return s.replace("\\", "/").lower()


def is_high_variant(text):
    """Check if text contains HIGH variant keywords using separator-aware matching."""
    text_lower = text.lower()
    text_spaced = " " + text_lower.replace("_", " ").replace("-", " ") + " "
    high_keywords = ["highnoise", "highres", "highfreq", "high noise", "high", "hn"]
    for kw in high_keywords:
        if f" {kw} " in text_spaced:
            return True
    if re.search(r"[\s_-]h[\s_.-]", text_lower) or re.search(r"[\s_-]h$", text_lower):
        return True
    return False


def is_low_variant(text):
    """Check if text contains LOW variant keywords using separator-aware matching."""
    text_lower = text.lower()
    text_spaced = " " + text_lower.replace("_", " ").replace("-", " ") + " "
    low_keywords = ["lownoise", "lowres", "lowfreq", "low noise", "low", "ln"]
    for kw in low_keywords:
        if f" {kw} " in text_spaced:
            return True
    if re.search(r"[\s_-]l[\s_.-]", text_lower) or re.search(r"[\s_-]l$", text_lower):
        return True
    return False


def extract_model_type(text):
    """Extract model type markers like T2V, I2V, etc. Returns set of found markers."""
    text_upper = text.upper()
    markers = set()
    if "T2V" in text_upper:
        markers.add("T2V")
    if "I2V" in text_upper:
        markers.add("I2V")
    if "14B" in text_upper:
        markers.add("14B")
    if "1.3B" in text_upper or "1_3B" in text_upper:
        markers.add("1.3B")
    return markers


def model_types_compatible(file_markers, mode):
    """
    Check if file model type markers are compatible with the selected mode.
    
    Modes:
    - I2V_ONLY: Accept files with I2V or no T2V/I2V marker. Reject files with T2V only.
    - T2V_ONLY: Accept files with T2V or no T2V/I2V marker. Reject files with I2V only.
    - INCLUSIVE: Accept files with BOTH T2V and I2V, or NEITHER. Reject files with only one.
    - NONE: No filtering, accept all files.
    
    Returns: (is_compatible, skip_reason) tuple
    """
    if mode == "NONE":
        return (True, None)
    
    t2v_in_file = "T2V" in file_markers
    i2v_in_file = "I2V" in file_markers
    
    if mode == "I2V_ONLY":
        if t2v_in_file and not i2v_in_file:
            return (False, "T2V-only file rejected in I2V_ONLY mode")
        return (True, None)
    elif mode == "T2V_ONLY":
        if i2v_in_file and not t2v_in_file:
            return (False, "I2V-only file rejected in T2V_ONLY mode")
        return (True, None)
    elif mode == "INCLUSIVE":
        has_both = t2v_in_file and i2v_in_file
        has_neither = not t2v_in_file and not i2v_in_file
        if has_both or has_neither:
            return (True, None)
        else:
            marker = "T2V" if t2v_in_file else "I2V"
            return (False, f"File has only {marker}, rejected in INCLUSIVE mode (requires both or neither)")
    return (True, None)


def strip_lora_extension(name):
    """Strip only known lora file extensions, not arbitrary dots like in 'Wan2.2'."""
    known_extensions = [".safetensors", ".ckpt", ".pt", ".pth", ".bin"]
    name_lower = name.lower()
    for ext in known_extensions:
        if name_lower.endswith(ext):
            return name[: -len(ext)]
    return name


def normalize_separators(text):
    """Normalize all separators to underscores for uniform comparison."""
    result = text.replace(" ", "_").replace("-", "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def remove_variant_keywords(text):
    """Remove HIGH/LOW variant keywords to get the base pattern."""
    result = text
    removal_patterns = [
        r"[_\s-]?highnoise[_\s-]?",
        r"[_\s-]?highres[_\s-]?",
        r"[_\s-]?highfreq[_\s-]?",
        r"[_\s-]?lownoise[_\s-]?",
        r"[_\s-]?lowres[_\s-]?",
        r"[_\s-]?lowfreq[_\s-]?",
        r"[_\s-]?high[_\s-]?noise[_\s-]?",
        r"[_\s-]?low[_\s-]?noise[_\s-]?",
        r"[_\s-]high(?=[_\s-]|$)",
        r"[_\s-]low(?=[_\s-]|$)",
        r"^high[_\s-]",
        r"^low[_\s-]",
        r"[_\s-]hn(?=[_\s.-]|$)",
        r"[_\s-]ln(?=[_\s.-]|$)",
        r"[_\s-]h(?=[_\s.-]|$)",
        r"[_\s-]l(?=[_\s.-]|$)",
    ]
    for pattern in removal_patterns:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    result = result.replace("_", " ").replace("-", " ")
    while "  " in result:
        result = result.replace("  ", " ")
    result = result.strip()
    return normalize_separators(result)


def get_all_loras_recursive(base_dirs):
    """Recursively collect all lora files under all base_dirs."""
    files = []
    exts = (".safetensors", ".sft", ".pt", ".ckpt")
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, filenames in os.walk(base_dir):
            for f in filenames:
                if any(f.endswith(ext) for ext in exts):
                    rel_path = os.path.relpath(os.path.join(root, f), base_dir)
                    files.append((rel_path.replace("\\", "/"), base_dir))
    return files


def find_matching_i2v_lora(
    lora_name: str,
    filenames_with_dirs: List[Tuple[str, str]],
    target_variant: str,
    model_type_mode: str = "I2V_ONLY",
    debug_func=None,
    DEBUG_MODE: bool = False,
) -> Dict:
    """
    Find matching I2V lora file for the given lora name.
    
    Returns: dict with keys:
        - match: (rel_path, base_dir) or None
        - skipped_model_type: list of (file_path, reason, file_markers) for files skipped
        - search_mode: the model_type_mode used
    """
    lora_name_norm = norm(lora_name)
    lora_base_norm = norm(strip_lora_extension(os.path.basename(lora_name)))
    
    input_base_pattern = remove_variant_keywords(lora_base_norm)
    input_base_normalized = normalize_separators(input_base_pattern)

    if debug_func:
        debug_func(f"=== Searching for {target_variant.upper()} variant ===", DEBUG_MODE)
        debug_func(f"  Input: '{lora_name}'", DEBUG_MODE)
        debug_func(f"  Input base norm: '{lora_base_norm}'", DEBUG_MODE)
        debug_func(f"  Input base pattern (no HIGH/LOW): '{input_base_normalized}'", DEBUG_MODE)
        debug_func(f"  Total files to search: {len(filenames_with_dirs)}", DEBUG_MODE)

    best_match = None
    best_score = 0
    skipped_model_type = []
    candidates_checked = 0

    for rel_path, base_dir in filenames_with_dirs:
        rel_path_norm = norm(rel_path)
        base_name_norm = norm(strip_lora_extension(os.path.basename(rel_path)))
        file_model_types = extract_model_type(rel_path)

        file_is_high = is_high_variant(base_name_norm)
        file_is_low = is_low_variant(base_name_norm)

        if target_variant == "high" and not file_is_high:
            continue
        if target_variant == "low" and not file_is_low:
            continue

        candidates_checked += 1
        
        is_compatible, skip_reason = model_types_compatible(file_model_types, model_type_mode)
        if not is_compatible:
            if lora_base_norm in base_name_norm or base_name_norm in lora_base_norm:
                skipped_model_type.append((rel_path, skip_reason, file_model_types))
            if debug_func:
                debug_func(f"  Skipped (model type): {rel_path} | {skip_reason}", DEBUG_MODE)
            continue

        file_base_pattern = remove_variant_keywords(base_name_norm)
        file_base_normalized = normalize_separators(file_base_pattern)

        match_score = 0
        match_reason = ""
        if lora_name_norm == rel_path_norm:
            match_score = 100
            match_reason = "exact path"
        elif lora_base_norm == base_name_norm:
            match_score = 100
            match_reason = "exact base name"
        elif input_base_normalized == file_base_normalized and input_base_normalized:
            match_score = 95
            match_reason = "base pattern match"
        elif input_base_normalized and file_base_normalized:
            if input_base_normalized in file_base_normalized or file_base_normalized in input_base_normalized:
                match_score = 92
                match_reason = "base pattern substring"
        
        if match_score == 0:
            if lora_base_norm in base_name_norm:
                match_score = 90
                match_reason = "input in filename"
            elif base_name_norm in lora_base_norm:
                match_score = 90
                match_reason = "filename in input"

        if debug_func and match_score > 0:
            debug_func(f"  Candidate: {rel_path} | score: {match_score} | {match_reason}", DEBUG_MODE)
            debug_func(f"    file_base_pattern: '{file_base_normalized}' vs input: '{input_base_normalized}'", DEBUG_MODE)

        if match_score > best_score:
            best_match = (rel_path, base_dir)
            best_score = match_score

    if debug_func:
        debug_func(f"  {target_variant.upper()} candidates checked: {candidates_checked}", DEBUG_MODE)
        debug_func(f"  Best score: {best_score}", DEBUG_MODE)

    return {
        'match': best_match if best_score >= 90 else None,
        'skipped_model_type': skipped_model_type,
        'search_mode': model_type_mode,
        'candidates_checked': candidates_checked,
        'best_score': best_score
    }


class illumoraeWan22I2VLoraLoaderByTextHighNode:
    """Loads Wan2.2 I2V HIGH LoRA models by fuzzy text string matching."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_type_mode": (["I2V_ONLY", "T2V_ONLY", "INCLUSIVE", "NONE"], {"default": "I2V_ONLY"}),
            },
            "optional": {
                "clip_strength_override": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "DEBUG_MODE": ("BOOLEAN", {"default": False}),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "lora_info")
    FUNCTION = "load_loras"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Loads Wan2.2 I2V HIGH LoRA models. HIGH typically affects model only (clip_strength=0 by default). Use clip_strength_override to change."

    # Default: HIGH LoRA does NOT affect CLIP
    DEFAULT_CLIP_STRENGTH = 0.0

    @staticmethod
    def debug_message(msg, DEBUG_MODE):
        if DEBUG_MODE:
            print(f"[illumoraeWan22I2VLoraLoaderByTextHighNode][DEBUG] {msg}")

    def parse_lora_syntax(self, text: str, DEBUG_MODE: bool = False) -> List[Tuple[str, float]]:
        lora_pattern = r"<lora:([^:<>]+):([0-9]*\.?[0-9]+)>"
        matches = re.findall(lora_pattern, text)
        parsed_loras = [(name.strip(), float(strength)) for name, strength in matches]
        self.debug_message(f"Parsed {len(parsed_loras)} LoRA tags from text: {parsed_loras}", DEBUG_MODE)
        return parsed_loras

    def apply_lora_to_model_and_clip(self, model, clip, lora_path: str, strength_model: float, strength_clip: float, DEBUG_MODE: bool = False):
        self.debug_message(f"Loading lora from: {lora_path} with model_strength={strength_model}, clip_strength={strength_clip}", DEBUG_MODE)
        lora_loader = nodes.LoraLoader()
        model_lora, clip_lora = lora_loader.load_lora(
            model=model,
            clip=clip,
            lora_name=lora_path,
            strength_model=strength_model,
            strength_clip=strength_clip,
        )
        return model_lora, clip_lora

    def load_loras(
        self,
        text: str,
        model,
        clip,
        model_type_mode: str = "I2V_ONLY",
        clip_strength_override: float = -1.0,
        DEBUG_MODE: bool = False,
        *args,
        **kwargs
    ) -> Tuple:
        parsed_loras = self.parse_lora_syntax(text, DEBUG_MODE)

        if len(parsed_loras) == 0:
            info = "No LoRA tags found in input text."
            return (model, clip, info)

        lora_dirs = folder_paths.get_folder_paths("loras")
        filenames_with_dirs = get_all_loras_recursive(lora_dirs)

        # Determine clip strength: -1 means use default (0 for HIGH)
        use_clip_strength = self.DEFAULT_CLIP_STRENGTH if clip_strength_override < 0 else clip_strength_override

        info_lines = [
            "=== WAN2.2 I2V HIGH LORA LOADING INFO ===",
            f"Model Type Mode: {model_type_mode}",
            f"Clip Strength: {use_clip_strength} (default: {self.DEFAULT_CLIP_STRENGTH}, override: {clip_strength_override})",
            f"Total LoRA tags parsed: {len(parsed_loras)}",
            "",
        ]

        current_model = model
        current_clip = clip
        applied_loras: Dict[str, float] = {}

        for lora_name, strength in parsed_loras:
            info_lines.append(f"Processing: <lora:{lora_name}:{strength}>")
            
            lora_base = strip_lora_extension(os.path.basename(lora_name)).lower()
            search_pattern = normalize_separators(remove_variant_keywords(lora_base))
            info_lines.append(f"  Search pattern: '{search_pattern}' (looking for HIGH)")
            
            result = find_matching_i2v_lora(
                lora_name, 
                filenames_with_dirs, 
                target_variant="high",
                model_type_mode=model_type_mode,
                debug_func=self.debug_message,
                DEBUG_MODE=DEBUG_MODE
            )
            
            match = result['match']
            skipped_model_type = result['skipped_model_type']
            
            if skipped_model_type:
                for skipped_path, skip_reason, file_markers in skipped_model_type:
                    info_lines.append(f"  [SKIP] {skipped_path}")
                    info_lines.append(f"         Reason: {skip_reason}")

            if not match:
                info_lines.append(f"  [FAIL] No I2V HIGH match found")
                info_lines.append(f"         Total files: {len(filenames_with_dirs)}, HIGH candidates: {result['candidates_checked']}, best score: {result['best_score']}")
                info_lines.append("")
                continue

            match_norm = match[0].replace("\\", "/").lower()
            info_lines.append(f"  HIGH: {match[0]}")

            if match_norm in applied_loras:
                prev_strength = applied_loras[match_norm]
                info_lines.append(f"  [SKIP] Already applied with strength {prev_strength} (dedup)")
                info_lines.append("")
                continue

            try:
                current_model, current_clip = self.apply_lora_to_model_and_clip(
                    current_model,
                    current_clip,
                    match[0],
                    strength,
                    DEBUG_MODE,
                )
                applied_loras[match_norm] = strength
                info_lines.append(f"  [OK] Applied I2V HIGH with strength {strength}")
            except Exception as e:
                info_lines.append(f"  [FAIL] Failed to apply I2V HIGH: {e}")

            info_lines.append("")

        info = "\n".join(info_lines)
        return (current_model, current_clip, info)


class illumoraeWan22I2VLoraLoaderByTextLowNode:
    """Loads Wan2.2 I2V LOW LoRA models by fuzzy text string matching."""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_type_mode": (["I2V_ONLY", "T2V_ONLY", "INCLUSIVE", "NONE"], {"default": "I2V_ONLY"}),
            },
            "optional": {
                "DEBUG_MODE": ("BOOLEAN", {"default": False}),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "lora_info")
    FUNCTION = "load_loras"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Loads Wan2.2 I2V LOW LoRA models by fuzzy text string matching from lora syntax tags."

    @staticmethod
    def debug_message(msg, DEBUG_MODE):
        if DEBUG_MODE:
            print(f"[illumoraeWan22I2VLoraLoaderByTextLowNode][DEBUG] {msg}")

    def parse_lora_syntax(self, text: str, DEBUG_MODE: bool = False) -> List[Tuple[str, float]]:
        lora_pattern = r"<lora:([^:<>]+):([0-9]*\.?[0-9]+)>"
        matches = re.findall(lora_pattern, text)
        parsed_loras = [(name.strip(), float(strength)) for name, strength in matches]
        self.debug_message(f"Parsed {len(parsed_loras)} LoRA tags from text: {parsed_loras}", DEBUG_MODE)
        return parsed_loras

    def apply_lora_to_model_and_clip(self, model, clip, lora_path: str, strength: float, DEBUG_MODE: bool = False):
        self.debug_message(f"Loading lora from: {lora_path} with strength {strength}", DEBUG_MODE)
        lora_loader = nodes.LoraLoader()
        model_lora, clip_lora = lora_loader.load_lora(
            model=model,
            clip=clip,
            lora_name=lora_path,
            strength_model=strength,
            strength_clip=strength,
        )
        return model_lora, clip_lora

    def load_loras(
        self,
        text: str,
        model,
        clip,
        model_type_mode: str = "I2V_ONLY",
        DEBUG_MODE: bool = False,
        *args,
        **kwargs
    ) -> Tuple:
        parsed_loras = self.parse_lora_syntax(text, DEBUG_MODE)

        if len(parsed_loras) == 0:
            info = "No LoRA tags found in input text."
            return (model, clip, info)

        lora_dirs = folder_paths.get_folder_paths("loras")
        filenames_with_dirs = get_all_loras_recursive(lora_dirs)

        info_lines = [
            "=== WAN2.2 I2V LOW LORA LOADING INFO ===",
            f"Model Type Mode: {model_type_mode}",
            f"Total LoRA tags parsed: {len(parsed_loras)}",
            "",
        ]

        current_model = model
        current_clip = clip
        applied_loras: Dict[str, float] = {}

        for lora_name, strength in parsed_loras:
            info_lines.append(f"Processing: <lora:{lora_name}:{strength}>")
            
            lora_base = strip_lora_extension(os.path.basename(lora_name)).lower()
            search_pattern = normalize_separators(remove_variant_keywords(lora_base))
            info_lines.append(f"  Search pattern: '{search_pattern}' (looking for LOW)")
            
            result = find_matching_i2v_lora(
                lora_name, 
                filenames_with_dirs, 
                target_variant="low",
                model_type_mode=model_type_mode,
                debug_func=self.debug_message,
                DEBUG_MODE=DEBUG_MODE
            )
            
            match = result['match']
            skipped_model_type = result['skipped_model_type']
            
            if skipped_model_type:
                for skipped_path, skip_reason, file_markers in skipped_model_type:
                    info_lines.append(f"  [SKIP] {skipped_path}")
                    info_lines.append(f"         Reason: {skip_reason}")

            if not match:
                info_lines.append(f"  [FAIL] No I2V LOW match found")
                info_lines.append(f"         Total files: {len(filenames_with_dirs)}, LOW candidates: {result['candidates_checked']}, best score: {result['best_score']}")
                info_lines.append("")
                continue

            match_norm = match[0].replace("\\", "/").lower()
            info_lines.append(f"  LOW: {match[0]}")

            if match_norm in applied_loras:
                prev_strength = applied_loras[match_norm]
                info_lines.append(f"  [SKIP] Already applied with strength {prev_strength} (dedup)")
                info_lines.append("")
                continue

            try:
                current_model, current_clip = self.apply_lora_to_model_and_clip(
                    current_model,
                    current_clip,
                    match[0],
                    strength,
                    DEBUG_MODE,
                )
                applied_loras[match_norm] = strength
                info_lines.append(f"  [OK] Applied I2V LOW with strength {strength}")
            except Exception as e:
                info_lines.append(f"  [FAIL] Failed to apply I2V LOW: {e}")

            info_lines.append("")

        info = "\n".join(info_lines)
        return (current_model, current_clip, info)


NODE_CLASS_MAPPINGS = {
    "illumoraeWan22I2VLoraLoaderByTextHighNode": illumoraeWan22I2VLoraLoaderByTextHighNode,
    "illumoraeWan22I2VLoraLoaderByTextLowNode": illumoraeWan22I2VLoraLoaderByTextLowNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeWan22I2VLoraLoaderByTextHighNode": "Wan2.2 I2V Lora Loader By Text (HIGH)",
    "illumoraeWan22I2VLoraLoaderByTextLowNode": "Wan2.2 I2V Lora Loader By Text (LOW)",
}
