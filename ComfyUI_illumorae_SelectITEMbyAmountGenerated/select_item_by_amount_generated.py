"""
TITLE::Select ITEM By Amount Generated
DESCRIPTIONSHORT::Selects an item from a project folder structure prioritizing low-coverage generation/ranking status, with optional prompt requirements and seeded iteration.
VERSION::20260127
GROUP::Select
IMAGE::comfyui_illumorae_select_item_by_amount_generated.png
"""
import re
from pathlib import Path

class illumoraeSelectITEMByAmountGeneratedNode:
    """
    SELECT ITEM BY AMOUNT GENERATED
    Within a project structure, prioritize items missing ranked images or with low generation coverage.
    
    Example PROJECT structure (CATEGORY > GROUP > ITEM ( with upscale ) ): 

      CATEGORY/                       # e.g., ART, UI, ENV
        GROUP/                        # e.g., ART_Food
          ITEM.psd                    # Base PSD for the item (original source)
          ITEM.bmp                    # (Optional) Base BMP preview or export
          ITEM_B.psd
          ITEM_B.bmp
          Upscale/
            ITEM.psd                  # Upscaled PSD for the same ITEM name
            ITEM.png                  # Upscaled PNG (preferred preview)
            ITEM/                     # Folder named after the ITEM
              prompt/
                prompt_sdxl.md        # prompt text for the sdxl model type 
              gen/
                01/                   # Best-ranked generated images for this upscaled ITEM
                  <ranked_images>.png
            ITEM_B.psd                # Upscaled PSD for the same ITEM name
            ITEM_B.png                # Upscaled PNG (preferred preview)
            ITEM_B/                   # Folder named after the ITEM
              gen/
                <unranked_images>.png
                01/                   # Best-ranked generated images for this upscaled ITEM
                  <ranked_good_images>.png
                02/                   # Second-best ranked generated images for this upscaled ITEM
                  <ranked_bad_images>.png

    Definitions and discovery:
    - ITEM is determined primarily from the Upscale folder contents:
      - If Upscale/ITEM.psd or Upscale/ITEM.png exists, ITEM is considered present in Upscale.
      - The corresponding base files are located in GROUP (e.g., GROUP/ITEM.psd, GROUP/ITEM.bmp).
    - Target image for editing is typically Upscale/ITEM.psd (the upscaled PSD).
    - Ranked generated outputs are stored under Upscale/ITEM/gen/01/*.png.
    - If gen/01 contains no files, the ITEM is considered unreviewed for rankings.

    Selection goals:
    - Items with fewer gen/01 images to surface assets needing attention.
    - Items matching a checkpoint-specific prompt (.md) for SDXL/sd1.5/flux/video
    - Items with low total generations in gen/ and ITEM Upscale
    - Items without any gen/01 ranked images
    - Down-rank items with large ranked histories in gen/02 over time

    PROJECT STRUCTURE EXAMPLE B (Simple - PNG in subfolder)
        base_dir/                       # example = POTION
            ITEM/                        # example = PotionA
                ITEM.png                    # target image
                prompt/
                    prompt_sdxl.md        # prompt text for the sdxl model type 
                gen/
                    01/                   # Best-ranked generated images for this upscaled ITEM
                        <ranked_images>.png

    PROJECT STRUCTURE EXAMPLE C (Simple - PNG in root)
        base_dir/                       # example = POTION
            ITEM.png                        # target image (in root)
            ITEM/                           # subfolder for item data
                prompt/
                    prompt_sdxl.md        # prompt text for the sdxl model type 
                gen/
                    01/                   # Best-ranked generated images for this upscaled ITEM
                        <ranked_images>.png


    """

    DESCRIPTION = "Selects an item from a project folder structure prioritizing low-coverage generation/ranking status, with optional prompt requirements and seeded iteration."

    @staticmethod
    def _debug_print(debug: bool, *args, **kwargs) -> None:
        if not debug:
            return
        print(*args, **kwargs)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'base_dir': ('STRING', {'default': str(Path(__file__).parent.parent)}),
                'use_nestedupscale_structure': ('BOOLEAN', {'default': False}),
                'png_in_item_root': ('BOOLEAN', {'default': False}),
                'require_prompt_sdxl': ('BOOLEAN', {'default': False}),
                'require_prompt_flux': ('BOOLEAN', {'default': False}),
                'require_prompt_sd15': ('BOOLEAN', {'default': False}),
                'require_prompt_video': ('BOOLEAN', {'default': False}),
                'max_images_target': ('INT', {'default': 5, 'min': 1, 'max': 1000, 'step': 1}),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            },
            'optional': {
                'debug': ('BOOLEAN', {'default': False}),
            }
        }

    RETURN_TYPES = ('STRING', 'STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('item_png_filepath', 'item_folder', 'item_name', 'item_names_list')
    FUNCTION = 'select'
    CATEGORY = 'illumorae'

    def select(self, base_dir, use_nestedupscale_structure=False, png_in_item_root=False, require_prompt_sdxl=False, 
               require_prompt_flux=False, require_prompt_sd15=False, require_prompt_video=False, 
               max_images_target=5, seed=0, debug=False):
        self._debug_print(debug, f"\n[SelectITEMbyAmountGenerated] === INPUT PARAMETERS ===")
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] base_dir: '{base_dir}'")
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] use_nestedupscale_structure: {use_nestedupscale_structure}")
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] png_in_item_root: {png_in_item_root}")
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] seed: {seed}")
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] debug: {debug}")
        
        root = Path(base_dir).expanduser()
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Expanded root path: '{root}'")
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Root exists: {root.exists()}")
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Root is_dir: {root.is_dir() if root.exists() else 'N/A'}")

        # Build prompt requirements dict
        prompt_requirements = {
            'sdxl': require_prompt_sdxl,
            'flux': require_prompt_flux,
            'sd15': require_prompt_sd15,
            'video': require_prompt_video,
        }

        # Discover and sort items based on structure type
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Structure type: {'nestedupscale' if use_nestedupscale_structure else 'simple'}")
        if use_nestedupscale_structure:
            items = self._discover_items_nestedupscale(
                root,
                include_categories=None,
                include_groups=None,
                prompt_requirements=prompt_requirements,
                debug=debug,
            )
        else:
            items = self._discover_items_simple(
                root,
                png_in_item_root=png_in_item_root,
                prompt_requirements=prompt_requirements,
                debug=debug,
            )
        
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Discovered {len(items)} items")
        if len(items) == 0:
            self._debug_print(debug, f"[SelectITEMbyAmountGenerated][WARNING] No items found! Check base_dir and structure settings.")
            return ('', '', '', '')
        
        # Build list of all item names for debugging/selection output
        all_item_names = sorted(set(item.get('name', '') for item in items if item.get('name')))
        item_names_list = '\n'.join(all_item_names)
        
        schedule = self._select_by_amount(items, max_images_target=max_images_target, debug=debug)

        # Use seed as index into schedule (ComfyUI increments seed with control_after_generate)
        top = None
        item_name = ''
        item_png_filepath = ''
        item_folder = ''
        
        if schedule:
            # Use seed modulo schedule length to select item
            idx = seed % len(schedule)
            top = schedule[idx]
            item_name = top.get('name', '')
            item_png_filepath = top.get('item_png', '')  # Get the PNG path from item data
            
            self._debug_print(debug, f"\n[SelectITEMbyAmountGenerated] === SELECTED ITEM ===")
            self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Schedule index: {idx}/{len(schedule)}")
            self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Item name: '{item_name}'")
            self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Item PNG from data: '{item_png_filepath}'")
            self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Item data: {top}")
            
            # Compute folder from available paths
            self._debug_print(debug, f"\n[SelectITEMbyAmountGenerated] === PATH CONSTRUCTION ===")
            try:
                if use_nestedupscale_structure:
                    # nestedupscale structure: CATEGORY/GROUP/Upscale/ITEM
                    self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Using nestedupscale structure")
                    base_psd = top.get('base_psd') or ''
                    item_psd = top.get('item_psd') or ''
                    existing_png = top.get('item_png') or ''
                    self._debug_print(debug, f"[SelectITEMbyAmountGenerated] base_psd: '{base_psd}'")
                    self._debug_print(debug, f"[SelectITEMbyAmountGenerated] item_psd: '{item_psd}'")
                    self._debug_print(debug, f"[SelectITEMbyAmountGenerated] existing_png: '{existing_png}'")
                    
                    group_dir = None
                    if base_psd:
                        group_dir = Path(base_psd).parent
                        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] group_dir from base_psd: '{group_dir}'")
                    elif item_psd:
                        group_dir = Path(item_psd).parent.parent
                        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] group_dir from item_psd: '{group_dir}'")
                    elif existing_png:
                        group_dir = Path(existing_png).parent.parent
                        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] group_dir from existing_png: '{group_dir}'")
                    
                    if group_dir:
                        item_folder = str(group_dir / 'Upscale')
                        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] item_folder: '{item_folder}'")
                    else:
                        self._debug_print(debug, f"[SelectITEMbyAmountGenerated][WARNING] Could not determine group_dir!")
                    
                    # Fill missing png path using folder + name
                    if not item_png_filepath and item_folder and item_name:
                        item_png_filepath = str(Path(item_folder) / f"{item_name}.png")
                        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Constructed PNG path: '{item_png_filepath}'")
                else:
                    # Simple structure: base_dir/ITEM or base_dir/ITEM.png
                    self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Using simple structure (png_in_item_root={png_in_item_root})")
                    if png_in_item_root:
                        # PNG in root: base_dir/ITEM.png
                        item_folder = str(root)
                        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] item_folder (root): '{item_folder}'")
                        if not item_png_filepath and item_name:
                            item_png_filepath = str(root / f"{item_name}.png")
                            self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Constructed PNG path: '{item_png_filepath}'")
                        elif not item_name:
                            self._debug_print(debug, f"[SelectITEMbyAmountGenerated][ERROR] item_name is empty!")
                    else:
                        # PNG in subfolder: base_dir/ITEM/ITEM.png
                        item_folder = str(root)
                        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] item_folder (root): '{item_folder}'")
                        if not item_png_filepath and item_name:
                            item_png_filepath = str(root / item_name / f"{item_name}.png")
                            self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Constructed PNG path: '{item_png_filepath}'")
                        elif not item_name:
                            self._debug_print(debug, f"[SelectITEMbyAmountGenerated][ERROR] item_name is empty!")
            except Exception as e:
                self._debug_print(debug, f"[SelectITEMbyAmountGenerated][ERROR] Building outputs: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
            
            # Additional debug output
            if debug:
                total_images = top.get('gen_count', 0) + top.get('gen01_count', 0) + top.get('gen02_count', 0)
                self._debug_print(debug, f"\n[SelectITEMbyAmountGenerated] === DEBUG INFO ===")
                self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Schedule: {len(schedule)} entries (target={max_images_target})")
                self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Total images: {total_images}")
                # Show next 3 items
                next_items = []
                for i in range(1, 4):
                    if idx + i < len(schedule):
                        next_items.append(schedule[idx + i].get('name', ''))
                if next_items:
                    self._debug_print(debug, f"[SelectITEMbyAmountGenerated] Next items: {', '.join(next_items)}")
        else:
            self._debug_print(debug, f"[SelectITEMbyAmountGenerated][WARNING] Schedule is empty!")
        
        # Final output validation
        self._debug_print(debug, f"\n[SelectITEMbyAmountGenerated] === FINAL OUTPUT ===")
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] item_png_filepath: '{item_png_filepath}'")
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] item_folder: '{item_folder}'")
        self._debug_print(debug, f"[SelectITEMbyAmountGenerated] item_name: '{item_name}'")
        
        if not item_png_filepath:
            self._debug_print(debug, f"[SelectITEMbyAmountGenerated][ERROR] item_png_filepath is EMPTY!")
        if not item_name:
            self._debug_print(debug, f"[SelectITEMbyAmountGenerated][ERROR] item_name is EMPTY!")
        
        return (item_png_filepath, item_folder, item_name, item_names_list)


    @staticmethod
    def _safe_str(p: Path | None) -> str:
        return str(p) if p else ""

    @staticmethod
    def _count_files(folder: Path, debug: bool = False) -> int:
        try:
            if not folder.exists() or not folder.is_dir():
                return 0
            return sum(1 for f in folder.iterdir() if f.is_file())
        except Exception as e:
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated][ERROR] Counting files in '{folder}': {e}")
            return 0
    
    @staticmethod
    def _build_gen_stats(gen_root: Path, gen01: Path, gen02: Path, prompt_folder: Path, debug: bool = False) -> dict:
        """Build common generation statistics dictionary for an item."""
        return {
            'gen_folder': illumoraeSelectITEMByAmountGeneratedNode._safe_str(gen_root if gen_root.exists() else None),
            'gen_count': int(illumoraeSelectITEMByAmountGeneratedNode._count_files(gen_root, debug=debug)) if gen_root.exists() else 0,
            'gen01_folder': illumoraeSelectITEMByAmountGeneratedNode._safe_str(gen01 if gen01.exists() else None),
            'gen01_count': int(illumoraeSelectITEMByAmountGeneratedNode._count_files(gen01, debug=debug)),
            'gen02_folder': illumoraeSelectITEMByAmountGeneratedNode._safe_str(gen02 if gen02.exists() else None),
            'gen02_count': int(illumoraeSelectITEMByAmountGeneratedNode._count_files(gen02, debug=debug)),
            'prompt_folder': illumoraeSelectITEMByAmountGeneratedNode._safe_str(prompt_folder if prompt_folder.exists() else None),
        }

    @staticmethod
    def _file_exists_and_has_text(file_path: Path, debug: bool = False) -> bool:
        try:
            if not file_path.exists() or not file_path.is_file():
                return False

            text = file_path.read_text(encoding='utf-8', errors='ignore')
            return bool(text.strip())
        except Exception as e:
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated][ERROR] Reading prompt file '{file_path}': {e}")
            return False
    
    @staticmethod
    def _check_prompt_requirements(prompt_folder: Path, prompt_requirements: dict, name: str, debug: bool) -> bool:
        """Check if item meets prompt requirements. Returns True if item should be included."""
        # If no requirements are enabled, include all items
        if not any(prompt_requirements.values()):
            if debug:
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]     No prompt requirements enabled - accepting all items")
            return True
        
        if debug:
            required_list = [k for k, v in prompt_requirements.items() if v]
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]     Required prompts: {', '.join(required_list)}")
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]     Prompt folder exists: {prompt_folder.exists()}")
        
        # Check which prompt files exist
        prompt_files = {
            'sdxl': prompt_folder / 'prompt_sdxl.md',
            'flux': prompt_folder / 'prompt_flux.md',
            'sd15': prompt_folder / 'prompt_sd15.md',
            'video': prompt_folder / 'prompt_video.md',
        }
        
        # Check if all required prompts exist
        missing_prompts = []
        for model_type, required in prompt_requirements.items():
            if required:
                prompt_file = prompt_files[model_type]
                file_exists = illumoraeSelectITEMByAmountGeneratedNode._file_exists_and_has_text(prompt_file, debug=debug)
                if debug:
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]       {model_type}: {prompt_file.name} exists: {file_exists}")
                if not file_exists:
                    missing_prompts.append(model_type)
        
        # If any required prompts are missing, skip this item
        if missing_prompts:
            if debug:
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]     [FAIL] Missing required prompts: {', '.join(missing_prompts)}")
            return False
        
        if debug:
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]     [PASS] All required prompts found")
        return True

    @staticmethod
    def _discover_items_simple(project_root: Path,
                               png_in_item_root: bool = False,
                               prompt_requirements: dict = None,
                               debug: bool = False):
        """
        Discover items in simple structure.
        
        Two variants supported:
        1. PNG in subfolder (png_in_item_root=False):
            base_dir/ITEM/ITEM.png with ITEM/prompt/ and ITEM/gen/
        
        2. PNG in root (png_in_item_root=True):
            base_dir/ITEM.png with ITEM/prompt/ and ITEM/gen/
        """
        items = []
        root = Path(project_root)
        if not root.exists():
            return items

        project_name = root.name

        structure_type = "PNG in root" if png_in_item_root else "PNG in subfolder"
        illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] === DISCOVERY (SIMPLE - {structure_type}) ===")
        illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Searching in: '{root}'")
        illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Project: '{project_name}'")

        if png_in_item_root:
            # Structure: base_dir/ITEM.png with ITEM/prompt/ and ITEM/gen/
            # Discover items by finding .png files in root
            png_files = sorted([p for p in root.glob('*.png') if not p.name.startswith('.')], key=lambda x: x.name.lower())
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Found {len(png_files)} PNG files in root")
            
            if len(png_files) == 0:
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated][WARNING] No PNG files found in root directory!")
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Expected structure: base_dir/ITEM.png with ITEM/prompt/ and ITEM/gen/")
                # List what IS in the directory
                all_files = list(root.iterdir())
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Directory contains {len(all_files)} items:")
                for item in all_files[:10]:  # Show first 10
                    item_type = "DIR" if item.is_dir() else "FILE"
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   [{item_type}] {item.name}")
                if len(all_files) > 10:
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   ... and {len(all_files) - 10} more items")
            
            for png_file in png_files:
                name = png_file.stem
                item_dir = root / name
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Checking PNG: '{png_file.name}' -> item name: '{name}'")
                
                if not item_dir.is_dir():
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   [SKIP] No corresponding folder '{name}' found")
                    continue  # Skip if no corresponding folder exists
                
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   [OK] Folder exists: '{item_dir}'")
                
                item_png = png_file
                gen_root = item_dir / 'gen'
                gen01 = gen_root / '01'
                gen02 = gen_root / '02'
                prompt_folder = item_dir / 'prompt'
                
                # Check folder structure
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   Checking subfolders:")
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]     prompt/: {prompt_folder.exists()}")
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]     gen/: {gen_root.exists()}")
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]     gen/01/: {gen01.exists()}")
                
                # Filter: require existing prompt files if requested
                if prompt_requirements is None:
                    prompt_requirements = {}
                
                # Check prompt requirements
                prompt_check = illumoraeSelectITEMByAmountGeneratedNode._check_prompt_requirements(prompt_folder, prompt_requirements, name, debug)
                if not prompt_check:
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   [SKIP] Failed prompt requirements check")
                    continue
                else:
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   [OK] Passed prompt requirements check")

                items.append({
                    'project': project_name,
                    'category': '',  # No category in simple structure
                    'group': '',     # No group in simple structure
                    'name': name,
                    'base_psd': '',  # No base PSD in simple structure
                    'item_psd': '',  # No item PSD in simple structure
                    'item_png': illumoraeSelectITEMByAmountGeneratedNode._safe_str(item_png if item_png.exists() else None),
                    **illumoraeSelectITEMByAmountGeneratedNode._build_gen_stats(gen_root, gen01, gen02, prompt_folder, debug=debug),
                })
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   [ADDED] Item '{name}': gen={items[-1]['gen_count']}, gen01={items[-1]['gen01_count']}, gen02={items[-1]['gen02_count']}")
        else:
            # Structure: base_dir/ITEM/ITEM.png with ITEM/prompt/ and ITEM/gen/
            # Iterate through immediate subdirectories as ITEMs
            item_dirs = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')], key=lambda x: x.name.lower())
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Found {len(item_dirs)} subdirectories")
            
            if len(item_dirs) == 0:
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated][WARNING] No subdirectories found!")
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Expected structure: base_dir/ITEM/ITEM.png with ITEM/prompt/ and ITEM/gen/")
                # List what IS in the directory
                all_files = list(root.iterdir())
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Directory contains {len(all_files)} items:")
                for item in all_files[:10]:  # Show first 10
                    item_type = "DIR" if item.is_dir() else "FILE"
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   [{item_type}] {item.name}")
                if len(all_files) > 10:
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   ... and {len(all_files) - 10} more items")
            
            for item_dir in item_dirs:
                name = item_dir.name
                item_png = item_dir / f"{name}.png"
                gen_root = item_dir / 'gen'
                gen01 = gen_root / '01'
                gen02 = gen_root / '02'
                prompt_folder = item_dir / 'prompt'
                
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Checking folder: '{name}'")
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   Expected PNG: '{item_png.name}' exists: {item_png.exists()}")
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   prompt/: {prompt_folder.exists()}")
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   gen/: {gen_root.exists()}")
                
                # Filter: require existing prompt files if requested
                if prompt_requirements is None:
                    prompt_requirements = {}
                
                prompt_check = illumoraeSelectITEMByAmountGeneratedNode._check_prompt_requirements(prompt_folder, prompt_requirements, name, debug)
                if not prompt_check:
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   [SKIP] Failed prompt requirements check")
                    continue
                else:
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   [OK] Passed prompt requirements check")

                items.append({
                    'project': project_name,
                    'category': '',  # No category in simple structure
                    'group': '',     # No group in simple structure
                    'name': name,
                    'base_psd': '',  # No base PSD in simple structure
                    'item_psd': '',  # No item PSD in simple structure
                    'item_png': illumoraeSelectITEMByAmountGeneratedNode._safe_str(item_png if item_png.exists() else None),
                    **illumoraeSelectITEMByAmountGeneratedNode._build_gen_stats(gen_root, gen01, gen02, prompt_folder, debug=debug),
                })
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated]   [ADDED] Item '{name}': gen={items[-1]['gen_count']}, gen01={items[-1]['gen01_count']}, gen02={items[-1]['gen02_count']}")
        
        illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] === DISCOVERY COMPLETE: {len(items)} items added ===")
        return items

    @staticmethod
    def _discover_items_nestedupscale(project_root: Path,
                                include_categories=None,
                                include_groups=None,
                                prompt_requirements: dict = None,
                                debug: bool = False):
        """
        Discover items in nestedupscale structure: CATEGORY/GROUP/Upscale/ITEM
        PROJECT STRUCTURE A (nestedupscale):
            CATEGORY/
                GROUP/
                    ITEM.psd
                    Upscale/
                        ITEM.psd
                        ITEM.png
                        ITEM/
                            prompt/
                                prompt_sdxl.md
                            gen/
                                01/
                                    <ranked_images>.png
        """
        items = []
        root = Path(project_root)
        if not root.exists():
            return items

        HEX_PREFIX_LOCAL = "0x"
        HEX_SUFFIX_RE_LOCAL = re.compile(r"0x[0-9A-Fa-f]+$")
        HEX_FILTER_ENABLED_LOCAL = False

        project_name = root.name

        if debug:
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Discovering items (nestedupscale structure) in project: '{project_name}' at {root}")

        for category in sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')], key=lambda x: x.name.lower()):
            if include_categories and category.name not in include_categories:
                continue
            if debug:
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"  [Category] {category.name}")
            for group in sorted([p for p in category.iterdir() if p.is_dir() and not p.name.startswith('.')], key=lambda x: x.name.lower()):
                if include_groups and group.name not in include_groups:
                    continue
                upscale = group / 'Upscale'
                names = set()
                if upscale.exists():
                    for p in list(upscale.glob('*.png')) + list(upscale.glob('*.psd')):
                        if HEX_FILTER_ENABLED_LOCAL and HEX_PREFIX_LOCAL not in p.name:
                            continue
                        names.add(p.stem)
                    for p in group.glob('*.psd'):
                        try:
                            if HEX_SUFFIX_RE_LOCAL.search(p.stem):
                                if HEX_FILTER_ENABLED_LOCAL and HEX_PREFIX_LOCAL not in p.name:
                                    continue
                                names.add(p.stem)
                        except Exception:
                            pass
                else:
                    for p in list(group.glob('*.psd')) + list(group.glob('*.bmp')):
                        if HEX_FILTER_ENABLED_LOCAL and HEX_PREFIX_LOCAL not in p.name:
                            continue
                        names.add(p.stem)
                if debug:
                    illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"    [Group] {group.name} -> candidates: {len(names)}")
                for name in sorted(names):
                    base_psd = group / f"{name}.psd"
                    item_psd = upscale / f"{name}.psd"
                    item_png = upscale / f"{name}.png"
                    gen_root = upscale / name / 'gen'
                    gen01 = gen_root / '01'
                    gen02 = gen_root / '02'
                    prompt_folder = upscale / name / 'prompt'
                    # Filter: require existing prompt files if requested
                    if prompt_requirements is None:
                        prompt_requirements = {}
                    if not illumoraeSelectITEMByAmountGeneratedNode._check_prompt_requirements(prompt_folder, prompt_requirements, name, debug):
                        continue

                    items.append({
                        'project': project_name,
                        'category': category.name,
                        'group': group.name,
                        'name': name,
                        'base_psd': illumoraeSelectITEMByAmountGeneratedNode._safe_str(base_psd if base_psd.exists() else None),
                        'item_psd': illumoraeSelectITEMByAmountGeneratedNode._safe_str(item_psd if item_psd.exists() else None),
                        'item_png': illumoraeSelectITEMByAmountGeneratedNode._safe_str(item_png if item_png.exists() else None),
                        **illumoraeSelectITEMByAmountGeneratedNode._build_gen_stats(gen_root, gen01, gen02, prompt_folder, debug=debug),
                    })
                    if debug:
                        illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"      [Item] {name}: gen01={items[-1]['gen01_count']}, gen02={items[-1]['gen02_count']}, prompt={'yes' if items[-1]['prompt_folder'] else 'no'}")
        return items

    @staticmethod
    def _select_by_amount(all_items, max_images_target=5, debug: bool = False):
        def sort_key(it):
            missing_item = 0 if (it.get('item_psd') or it.get('item_png')) else 1
            # Priority: gen_count (total unranked), gen01_count, missing_item, gen02_count
            return (
                it.get('gen_count', 0),  # Primary: total unranked generations (low gen/)
                it.get('gen01_count', 0),  # Secondary: ranked gen01 (low)
                -missing_item,  # Tertiary: missing item files (prioritize existing)
                it.get('gen02_count', 0),  # Quaternary: ranked gen02 (low)
                it.get('group', ''),
                it.get('name', ''),
            )
        sorted_items = sorted(all_items, key=sort_key)
        
        # Create balanced schedule: items with fewer images appear more frequently
        schedule = illumoraeSelectITEMByAmountGeneratedNode._create_balanced_schedule(sorted_items, max_images_target=max_images_target, debug=debug)
        
        if debug:
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Discovered {len(all_items)} items, created schedule with {len(schedule)} entries")
        return schedule
    
    @staticmethod
    def _create_balanced_schedule(items, max_images_target=5, debug: bool = False):
        """Create a balanced schedule where items with fewer images appear more frequently."""
        if not items:
            return []
        
        # Calculate total image count for each item (gen + gen01 + gen02)
        item_counts = []
        for item in items:
            total = item.get('gen_count', 0) + item.get('gen01_count', 0) + item.get('gen02_count', 0)
            item_counts.append((item, total))
        
        # Use max_images_target as the balancing target (or actual max if higher)
        actual_max = max(count for _, count in item_counts) if item_counts else 0
        max_count = max(max_images_target, actual_max)
        
        # Calculate repetitions for each item (items with fewer images get more repetitions)
        schedule_entries = []
        for item, count in item_counts:
            repetitions = max_count - count + 1  # +1 ensures even max_count items appear once
            schedule_entries.append((item, repetitions))
        
        if debug:
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Balanced Schedule Calculation:")
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"  Target image count: {max_images_target}")
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"  Actual max image count: {actual_max}")
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"  Using max count: {max_count}")
            for item, reps in schedule_entries[:5]:  # Show first 5
                total = item.get('gen_count', 0) + item.get('gen01_count', 0) + item.get('gen02_count', 0)
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"    {item.get('name')}: total_images={total}, repetitions={reps}")
        
        # Create interleaved schedule using round-robin with repetition counts
        schedule = []
        remaining = [(item, reps) for item, reps in schedule_entries]
        
        while remaining:
            # Add one entry from each item that still has repetitions left
            next_remaining = []
            for item, reps in remaining:
                schedule.append(item)
                if reps > 1:
                    next_remaining.append((item, reps - 1))
            remaining = next_remaining
        
        if debug:
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] Created schedule with {len(schedule)} entries")
            # Show first 10 entries
            illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"[SelectITEMbyAmountGenerated] First 10 schedule entries:")
            for i, item in enumerate(schedule[:10]):
                total = item.get('gen_count', 0) + item.get('gen01_count', 0) + item.get('gen02_count', 0)
                illumoraeSelectITEMByAmountGeneratedNode._debug_print(debug, f"    {i}: {item.get('name')} (total_images={total})")
        
        return schedule

NODE_CLASS_MAPPINGS = {
    'illumoraeSelectITEMByAmountGeneratedNode': illumoraeSelectITEMByAmountGeneratedNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeSelectITEMByAmountGeneratedNode': 'Select ITEM By Amount Generated',
}
