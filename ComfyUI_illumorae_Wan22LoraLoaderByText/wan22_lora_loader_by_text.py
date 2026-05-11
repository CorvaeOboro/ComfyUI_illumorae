"""
illumorae Wan2.2 Lora Loader By Text - a ComfyUI custom_node

TITLE::Wan2.2 Lora Loader By Text
DESCRIPTIONSHORT::Loads Wan2.2 LoRA models (HIGH/LOW pairs) by fuzzy text string matching from lora syntax tags.
VERSION::20260124
IMAGE::comfyui_illumorae_wan22_lora_loader_by_text.png
GROUP::Lora

MATCHING RULES:

PHASE 1 - Primary Match (find the lora specified in the tag by ranked matching):
  - Exact path match (score 100)
  - Exact base filename match (score 100)  
  - Exact base pattern match after removing HIGH/LOW keywords (score 95)
  - Substring match where input contains filename or vice versa (score 90)
  - Minimum score of 90 required to accept a primary match

PHASE 2 - Pair Match (find the corresponding HIGH or LOW variant):
  - MUST be in the EXACT SAME directory as the primary match
  - MUST have the EXACT SAME base pattern (after removing HIGH/LOW keywords)
  - MUST be the opposite variant type (if primary is HIGH, pair must be LOW)

ALLOWED MATCHES:
  - "MyLora_High.safetensors" pairs with "MyLora_Low.safetensors" in same folder
  - "Style-T2V-HIGH-14B.safetensors" pairs with "Style-T2V-LOW-14B.safetensors" in same folder
  - If no valid pair found, only the primary is applied ( occasionally lora is only HIGH or only LOW)

NOT ALLOWED (will be rejected):
  - Pairs from DIFFERENT directories (e.g., T2V folder + I2V folder)
  - Pairs with DIFFERENT base names (e.g., "Apple" + "BerryV2")
  - Pairs with DIFFERENT model types (e.g., T2V + I2V, 14B + 1.3B)
  - Loose partial matches (e.g., matching on single tokens like "wan" or "lora")

MODEL TYPE MODE (toggle on the node):
  - T2V_ONLY (default): Accept files with T2V or no T2V/I2V marker. REJECT files with I2V only.
  - I2V_ONLY: Accept files with I2V or no T2V/I2V marker. REJECT files with T2V only.
  - INCLUSIVE: Accept files with BOTH T2V and I2V in name, or NEITHER. Reject files with only one.
  - NONE: No model type filtering at all. Accept any file regardless of T2V/I2V markers.
  
  Skipped files due to model type mode restrictions are shown in lora_info output for debugging.

HIGH/LOW KEYWORDS:
  - HIGH variants: highnoise, highres, highfreq, high_noise, high, hn (as whole words), _h_ or -h- or _h. (suffix patterns)
  - LOW variants: lownoise, lowres, lowfreq, low_noise, low, ln (as whole words), _l_ or -l- or _l. (suffix patterns)

DEDUPLICATION:
  - Tracks which LoRA files have been applied to each model (HIGH/LOW)
  - If a LoRA file is already applied, it will be SKIPPED (not applied twice)
  - This allows users to specify both HIGH and LOW versions explicitly with different strengths
  - Example: <lora:MyStyle_HIGH:0.8> and <lora:MyStyle_LOW:0.5> will apply each at their specified strength
  - The auto-pair feature won't create duplicates if the user already specified both variants

"""
import os
import re
import folder_paths
import nodes
from typing import Dict, List, Tuple, Optional

class illumoraeWan22LoraLoaderByTextNode:
    """
    A ComfyUI node that loads Wan2.2 LoRA models (HIGH/LOW pairs) by parsing text strings
    with lora syntax and fuzzy matching to find corresponding lora files.
    
    Wan2.2 Architecture:
    - Uses separate HIGH and LOW models as inputs
    - Each model processes independently with matched lora files
    
    Features:
    - Parses <lora:name:strength> syntax from input text
    - Fuzzy matching to find HIGH and LOW lora file pairs based on name similarity
    - Applies loras to separate HIGH and LOW models and CLIP
    - Optional mode restriction , for example T2V or I2V required in filename
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "clip": ("CLIP",),
                "model_type_mode": (["T2V_ONLY", "I2V_ONLY", "INCLUSIVE", "NONE"], {"default": "T2V_ONLY"}),
            },
            "optional": {
                "DEBUG_MODE": ("BOOLEAN", {"default": False}),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model_high", "model_low", "clip", "lora_info")
    FUNCTION = "load_loras"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Loads Wan2.2 LoRA models (HIGH/LOW pairs) by fuzzy text string matching from lora syntax tags."

    @staticmethod
    def debug_message(msg, DEBUG_MODE):
        if DEBUG_MODE:
            print(f"[illumoraeWan22LoraLoaderByTextNode][DEBUG] {msg}")

    @staticmethod
    def _get_all_loras_recursive(base_dirs):
        """
        Recursively collect all lora files under all base_dirs, returning (rel_path, base_dir) tuples.
        """
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

    def parse_lora_syntax(self, text: str, DEBUG_MODE: bool = False) -> List[Tuple[str, float]]:
        """
        Parse LoRA strings from the input text.
        Expected LoRA syntax format: <lora:name:strength>
        Returns a list of tuples (name, strength).
        """
        lora_pattern = r"<lora:([^:<>]+):([0-9]*\.?[0-9]+)>"
        matches = re.findall(lora_pattern, text)
        parsed_loras = [(name.strip(), float(strength)) for name, strength in matches]
        self.debug_message(f"Parsed {len(parsed_loras)} LoRA tags from text: {parsed_loras}", DEBUG_MODE)
        return parsed_loras

    @classmethod
    def find_matching_lora_pair(
        cls, 
        lora_name: str, 
        filenames_with_dirs: List[Tuple[str, str]], 
        model_type_mode: str = "T2V_ONLY",
        DEBUG_MODE: bool = False
    ) -> Dict:
        """
        Find matching HIGH and LOW lora files for the given lora name.
        STRICT matching: pairs must be in the SAME folder with nearly identical base names.
        
        Returns: dict with keys:
            - high_match: (rel_path, base_dir) or None
            - low_match: (rel_path, base_dir) or None
            - skipped_model_type: list of (file_path, reason) for files skipped due to model type
            - search_mode: the model_type_mode used
        """
        def norm(s):
            return s.replace("\\", "/").lower()
        
        def is_high_variant(text):
            """Check if text contains HIGH variant keywords using boundary-aware matching.
            A HIGH keyword is accepted when it is not part of a longer alphabetic run
            (letters before/after would indicate it's inside another word like 'thigh'),
            so digits, dots, separators, or string edges all count as boundaries.
            This allows patterns like '109high.LA0X' to match.
            """
            text_lower = text.lower()
            text_spaced = ' ' + text_lower.replace('_', ' ').replace('-', ' ') + ' '
            compound_keywords = ['highnoise', 'highres', 'highfreq', 'high noise']
            for kw in compound_keywords:
                if f' {kw} ' in text_spaced or kw in text_lower:
                    return True
            if re.search(r'(?<![a-z])high(?![a-z])', text_lower):
                return True
            if re.search(r'(?<![a-z])hn(?![a-z])', text_lower):
                return True
            if re.search(r'[\s_-]h[\s_.-]', text_lower) or re.search(r'[\s_-]h$', text_lower):
                return True
            return False
        
        def is_low_variant(text):
            """Check if text contains LOW variant keywords using boundary-aware matching.
            Non-letter characters (digits, dots, separators, edges) count as boundaries,
            so patterns like '109low.LA0X' match while false positives like 'flow',
            'below', 'yellow', 'lightx2v' are still rejected.
            """
            text_lower = text.lower()
            text_spaced = ' ' + text_lower.replace('_', ' ').replace('-', ' ') + ' '
            compound_keywords = ['lownoise', 'lowres', 'lowfreq', 'low noise']
            for kw in compound_keywords:
                if f' {kw} ' in text_spaced or kw in text_lower:
                    return True
            if re.search(r'(?<![a-z])low(?![a-z])', text_lower):
                return True
            if re.search(r'(?<![a-z])ln(?![a-z])', text_lower):
                return True
            if re.search(r'[\s_-]l[\s_.-]', text_lower) or re.search(r'[\s_-]l$', text_lower):
                return True
            return False

        def extract_model_type(text):
            """Extract model type markers like T2V, I2V, etc. Returns set of found markers."""
            text_upper = text.upper()
            markers = set()
            if 'T2V' in text_upper:
                markers.add('T2V')
            if 'I2V' in text_upper:
                markers.add('I2V')
            if '14B' in text_upper:
                markers.add('14B')
            if '1.3B' in text_upper or '1_3B' in text_upper:
                markers.add('1.3B')
            return markers
        
        def model_types_compatible(file_markers, mode):
            """
            Check if file model type markers are compatible with the selected mode.
            
            Modes:
            - T2V_ONLY: Accept files with T2V or no T2V/I2V marker. Reject files with I2V only.
            - I2V_ONLY: Accept files with I2V or no T2V/I2V marker. Reject files with T2V only.
            - INCLUSIVE: Accept files with BOTH T2V and I2V, or NEITHER. Reject files with only one.
            - NONE: No filtering, accept all files.
            
            Returns: (is_compatible, skip_reason) tuple
            """
            if mode == "NONE":
                return (True, None)
            
            t2v_in_file = 'T2V' in file_markers
            i2v_in_file = 'I2V' in file_markers
            
            if mode == "T2V_ONLY":
                if i2v_in_file and not t2v_in_file:
                    return (False, f"I2V-only file rejected in T2V_ONLY mode")
                return (True, None)
            elif mode == "I2V_ONLY":
                if t2v_in_file and not i2v_in_file:
                    return (False, f"T2V-only file rejected in I2V_ONLY mode")
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
            known_extensions = ['.safetensors', '.ckpt', '.pt', '.pth', '.bin']
            name_lower = name.lower()
            for ext in known_extensions:
                if name_lower.endswith(ext):
                    return name[:-len(ext)]
            return name
        
        lora_name_norm = norm(lora_name)
        lora_base_norm = norm(strip_lora_extension(os.path.basename(lora_name)))
        
        input_is_high = is_high_variant(lora_base_norm)
        input_is_low = is_low_variant(lora_base_norm)
        input_model_types = extract_model_type(lora_base_norm)
        
        def normalize_separators(text):
            """Normalize all separators (spaces, underscores, hyphens) to underscores for uniform comparison."""
            result = text.replace(" ", "_").replace("-", "_")
            while "__" in result:
                result = result.replace("__", "_")
            return result.strip("_")
        
        def remove_variant_keywords(text):
            """Remove HIGH/LOW variant keywords to get the base pattern."""
            result = text
            removal_patterns = [
                r'[_\s-]?highnoise[_\s-]?', r'[_\s-]?highres[_\s-]?', r'[_\s-]?highfreq[_\s-]?',
                r'[_\s-]?lownoise[_\s-]?', r'[_\s-]?lowres[_\s-]?', r'[_\s-]?lowfreq[_\s-]?',
                r'[_\s-]?high[_\s-]?noise[_\s-]?', r'[_\s-]?low[_\s-]?noise[_\s-]?',
                r'(?<![a-z])high(?![a-z])', r'(?<![a-z])low(?![a-z])',
                r'(?<![a-z])hn(?![a-z])', r'(?<![a-z])ln(?![a-z])',
                r'[_\s-]h(?=[_\s.-]|$)', r'[_\s-]l(?=[_\s.-]|$)',
            ]
            for pattern in removal_patterns:
                result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
            result = result.replace("_", " ").replace("-", " ")
            while "  " in result:
                result = result.replace("  ", " ")
            result = result.strip()
            return normalize_separators(result)
        
        def get_directory(path):
            """Get the directory portion of a path."""
            return norm(os.path.dirname(path))
        
        input_base_pattern = remove_variant_keywords(lora_base_norm)
        input_base_normalized = normalize_separators(input_base_pattern)
        
        cls.debug_message(f"\n{'='*60}", DEBUG_MODE)
        cls.debug_message(f"Searching for lora pair: '{lora_name}'", DEBUG_MODE)
        cls.debug_message(f"Normalized base: '{lora_base_norm}'", DEBUG_MODE)
        cls.debug_message(f"Input is HIGH: {input_is_high}, Input is LOW: {input_is_low}", DEBUG_MODE)
        cls.debug_message(f"Base pattern (no HIGH/LOW): '{input_base_normalized}'", DEBUG_MODE)
        cls.debug_message(f"Total files to search: {len(filenames_with_dirs)}", DEBUG_MODE)
        cls.debug_message(f"{'='*60}", DEBUG_MODE)
        
        primary_match = None
        primary_is_high = False
        primary_score = 0
        skipped_model_type = []
        
        for rel_path, base_dir in filenames_with_dirs:
            rel_path_norm = norm(rel_path)
            base_name_norm = norm(strip_lora_extension(os.path.basename(rel_path)))
            
            file_is_high = is_high_variant(base_name_norm)
            file_is_low = is_low_variant(base_name_norm)
            
            if not file_is_high and not file_is_low:
                continue
            
            if input_is_high and not file_is_high:
                continue
            if input_is_low and not file_is_low:
                continue
            
            file_model_types = extract_model_type(rel_path)
            is_compatible, skip_reason = model_types_compatible(file_model_types, model_type_mode)
            if not is_compatible:
                if lora_base_norm in base_name_norm or base_name_norm in lora_base_norm:
                    skipped_model_type.append((rel_path, skip_reason, file_model_types))
                cls.debug_message(f"  [!] Skipped (model type): {rel_path} | {skip_reason} | file markers: {file_model_types}", DEBUG_MODE)
                continue
            
            file_base_pattern = remove_variant_keywords(base_name_norm)
            file_base_normalized = normalize_separators(file_base_pattern)
            
            match_score = 0
            match_reason = ""
            
            if lora_name_norm == rel_path_norm:
                match_score = 100
                match_reason = "exact path match"
            elif lora_base_norm == base_name_norm:
                match_score = 100
                match_reason = "exact base name match"
            elif input_base_normalized == file_base_normalized and input_base_normalized:
                match_score = 95
                match_reason = "exact base pattern match"
            elif lora_base_norm in base_name_norm:
                match_score = 90
                match_reason = "input is substring of file"
            elif base_name_norm in lora_base_norm:
                match_score = 90
                match_reason = "file is substring of input"
            
            if match_score > primary_score:
                primary_match = (rel_path, base_dir)
                primary_is_high = file_is_high
                primary_score = match_score
                cls.debug_message(f"  [+] Primary candidate: {rel_path} | score: {match_score} | reason: {match_reason} | type: {'HIGH' if file_is_high else 'LOW'}", DEBUG_MODE)
        
        if not primary_match or primary_score < 90:
            cls.debug_message(f"\n[-] No strong primary match found for '{lora_name}' (best score: {primary_score})", DEBUG_MODE)
            return {
                'high_match': None,
                'low_match': None,
                'skipped_model_type': skipped_model_type,
                'search_mode': model_type_mode
            }
        
        cls.debug_message(f"\n[*] Primary match found: {primary_match[0]} ({'HIGH' if primary_is_high else 'LOW'})", DEBUG_MODE)
        
        primary_dir = get_directory(primary_match[0])
        primary_base_pattern = remove_variant_keywords(norm(strip_lora_extension(os.path.basename(primary_match[0]))))
        primary_base_normalized = normalize_separators(primary_base_pattern)
        
        cls.debug_message(f"  Looking for pair in same directory: {primary_dir}", DEBUG_MODE)
        cls.debug_message(f"  With base pattern: '{primary_base_normalized}'", DEBUG_MODE)
        
        pair_match = None
        
        for rel_path, base_dir in filenames_with_dirs:
            rel_path_norm = norm(rel_path)
            file_dir = get_directory(rel_path)
            
            if file_dir != primary_dir:
                continue
            
            if rel_path_norm == norm(primary_match[0]):
                continue
            
            base_name_norm = norm(strip_lora_extension(os.path.basename(rel_path)))
            
            file_is_high = is_high_variant(base_name_norm)
            file_is_low = is_low_variant(base_name_norm)
            
            if primary_is_high and not file_is_low:
                continue
            if not primary_is_high and not file_is_high:
                continue
            
            file_base_pattern = remove_variant_keywords(base_name_norm)
            file_base_normalized = normalize_separators(file_base_pattern)
            
            if file_base_normalized != primary_base_normalized:
                cls.debug_message(f"  [!] Rejected pair (different base): {rel_path} | base: '{file_base_normalized}' != '{primary_base_normalized}'", DEBUG_MODE)
                continue
            
            pair_match = (rel_path, base_dir)
            cls.debug_message(f"  [+] Found pair: {rel_path} | type: {'HIGH' if file_is_high else 'LOW'}", DEBUG_MODE)
            break
        
        high_match = None
        low_match = None
        
        if primary_is_high:
            high_match = primary_match
            low_match = pair_match
        else:
            low_match = primary_match
            high_match = pair_match
        
        if high_match:
            cls.debug_message(f"\n[*] SELECTED HIGH: {high_match[0]}", DEBUG_MODE)
        else:
            cls.debug_message(f"\n[-] No HIGH match found", DEBUG_MODE)
            
        if low_match:
            cls.debug_message(f"\n[*] SELECTED LOW: {low_match[0]}", DEBUG_MODE)
        else:
            cls.debug_message(f"\n[-] No LOW match found", DEBUG_MODE)
        
        return {
            'high_match': high_match,
            'low_match': low_match,
            'skipped_model_type': skipped_model_type,
            'search_mode': model_type_mode
        }

    def apply_lora_to_model_and_clip(self, model, clip, lora_path: str, strength: float, DEBUG_MODE: bool = False):
        """
        Apply a single lora to both model and CLIP using ComfyUI's official LoraLoader.
        """
        try:
            self.debug_message(f"Loading lora from: {lora_path} with strength {strength}", DEBUG_MODE)
            self.debug_message(f"Input model ID: {id(model)}", DEBUG_MODE)
            self.debug_message(f"Input clip ID: {id(clip)}", DEBUG_MODE)
            
            lora_loader = nodes.LoraLoader()
            model_lora, clip_lora = lora_loader.load_lora(
                model=model,
                clip=clip,
                lora_name=lora_path,
                strength_model=strength,
                strength_clip=strength
            )
            
            self.debug_message(f"Output model ID: {id(model_lora)}", DEBUG_MODE)
            self.debug_message(f"Output clip ID: {id(clip_lora)}", DEBUG_MODE)
            self.debug_message(f"Model changed: {id(model) != id(model_lora)}", DEBUG_MODE)
            self.debug_message(f"Clip changed: {id(clip) != id(clip_lora)}", DEBUG_MODE)
            
            if hasattr(model_lora, 'model') and hasattr(model_lora.model, 'diffusion_model'):
                self.debug_message(f"Model has diffusion_model attribute", DEBUG_MODE)
            
            self.debug_message(f"Successfully applied lora to model and CLIP: {lora_path}", DEBUG_MODE)
            return model_lora, clip_lora
            
        except Exception as e:
            self.debug_message(f"Error loading lora {lora_path}: {e}", True)
            import traceback
            self.debug_message(f"Traceback: {traceback.format_exc()}", True)
            raise

    def load_loras(
        self,
        text: str,
        model_high,
        model_low,
        clip,
        model_type_mode: str = "T2V_ONLY",
        DEBUG_MODE: bool = False,
        *args,
        **kwargs
    ) -> Tuple:
        """
        Main processing function that parses lora tags and loads corresponding HIGH/LOW lora pairs.
        
        Returns:
            - Modified MODEL HIGH
            - Modified MODEL LOW
            - Modified CLIP
            - Info string about status of matching and loaded loras
        """
        self.debug_message(f"\n{'#'*60}", DEBUG_MODE)
        self.debug_message(f"WAN2.2 LORA LOADER - PROCESSING START", DEBUG_MODE)
        self.debug_message(f"Model type mode: {model_type_mode}", DEBUG_MODE)
        self.debug_message(f"{'#'*60}", DEBUG_MODE)
        self.debug_message(f"Processing text for lora tags...", DEBUG_MODE)
        
        parsed_loras = self.parse_lora_syntax(text, DEBUG_MODE)
        
        if len(parsed_loras) == 0:
            info = "No LoRA tags found in input text."
            self.debug_message(info, DEBUG_MODE)
            return (model_high, model_low, clip, info)
        
        lora_dirs = folder_paths.get_folder_paths("loras")
        self.debug_message(f"\nLora search directories: {lora_dirs}", DEBUG_MODE)
        filenames_with_dirs = self._get_all_loras_recursive(lora_dirs)
        self.debug_message(f"Found {len(filenames_with_dirs)} lora files in search paths.", DEBUG_MODE)
        
        info_lines = [
            "=== WAN2.2 LORA LOADING INFO ===",
            f"Model Type Mode: {model_type_mode}",
            f"Total LoRA tags parsed: {len(parsed_loras)}",
            ""
        ]
        
        current_model_high = model_high
        current_model_low = model_low
        current_clip = clip
        
        applied_high_loras = {}
        applied_low_loras = {}
        
        for lora_name, strength in parsed_loras:
            info_lines.append(f"Processing: <lora:{lora_name}:{strength}>")
            
            result = self.find_matching_lora_pair(lora_name, filenames_with_dirs, model_type_mode, DEBUG_MODE)
            high_match = result['high_match']
            low_match = result['low_match']
            skipped_model_type = result['skipped_model_type']
            
            if skipped_model_type:
                for skipped_path, skip_reason, file_markers in skipped_model_type:
                    info_lines.append(f"  [SKIP] {skipped_path}")
                    info_lines.append(f"         Reason: {skip_reason}")
            
            if high_match:
                high_path_norm = high_match[0].replace("\\", "/").lower()
                if high_path_norm in applied_high_loras:
                    prev_strength = applied_high_loras[high_path_norm]
                    info_lines.append(f"  HIGH: {high_match[0]}")
                    info_lines.append(f"  [SKIP] Already applied HIGH with strength {prev_strength} (dedup)")
                    self.debug_message(f"Skipping duplicate HIGH lora: {high_match[0]} (already applied at {prev_strength})", DEBUG_MODE)
                else:
                    info_lines.append(f"  HIGH: {high_match[0]}")
                    try:
                        current_model_high, current_clip = self.apply_lora_to_model_and_clip(
                            current_model_high, 
                            current_clip, 
                            high_match[0], 
                            strength, 
                            DEBUG_MODE
                        )
                        applied_high_loras[high_path_norm] = strength
                        info_lines.append(f"  [OK] Applied HIGH to model_high with strength {strength}")
                    except Exception as e:
                        info_lines.append(f"  [FAIL] Failed to apply HIGH: {e}")
                        self.debug_message(f"ERROR applying HIGH lora: {e}", True)
            else:
                info_lines.append(f"  [FAIL] No HIGH match found")
            
            if low_match:
                low_path_norm = low_match[0].replace("\\", "/").lower()
                if low_path_norm in applied_low_loras:
                    prev_strength = applied_low_loras[low_path_norm]
                    info_lines.append(f"  LOW: {low_match[0]}")
                    info_lines.append(f"  [SKIP] Already applied LOW with strength {prev_strength} (dedup)")
                    self.debug_message(f"Skipping duplicate LOW lora: {low_match[0]} (already applied at {prev_strength})", DEBUG_MODE)
                else:
                    info_lines.append(f"  LOW: {low_match[0]}")
                    try:
                        current_model_low, current_clip = self.apply_lora_to_model_and_clip(
                            current_model_low, 
                            current_clip, 
                            low_match[0], 
                            strength, 
                            DEBUG_MODE
                        )
                        applied_low_loras[low_path_norm] = strength
                        info_lines.append(f"  [OK] Applied LOW to model_low with strength {strength}")
                    except Exception as e:
                        info_lines.append(f"  [FAIL] Failed to apply LOW: {e}")
                        self.debug_message(f"ERROR applying LOW lora: {e}", True)
            else:
                info_lines.append(f"  [FAIL] No LOW match found")
            
            info_lines.append("")
        
        info = "\n".join(info_lines)
        self.debug_message(f"\n{'#'*60}", DEBUG_MODE)
        self.debug_message(f"Completed loading {len(parsed_loras)} lora pairs", DEBUG_MODE)
        self.debug_message(f"{'#'*60}\n", DEBUG_MODE)
        
        return (current_model_high, current_model_low, current_clip, info)


NODE_CLASS_MAPPINGS = {
    'illumoraeWan22LoraLoaderByTextNode': illumoraeWan22LoraLoaderByTextNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeWan22LoraLoaderByTextNode': 'Wan2.2 Lora Loader By Text',
}
