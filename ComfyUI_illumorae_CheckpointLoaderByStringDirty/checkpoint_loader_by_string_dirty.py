"""
illumorae Checkpoint Loader By String Dirty - a ComfyUI Custom Node

Loads a Diffusion checkpoint by matching a string input to available checkpoint files.
supporting full paths, relative paths, or filenames 

Inputs:
    ckpt_name: The name or path of the checkpoint to load (string).
    DEBUG_MODE: Enable debug output 

Outputs:
    model: The loaded model object.
    clip: The loaded CLIP object.
    vae: The loaded VAE object.
    ckpt_filename: The resolved checkpoint filename (string).

TITLE::Checkpoint Loader By String (Safe by Default)
DESCRIPTIONSHORT::Loads a checkpoint by fuzzy matching the text input finds available checkpoint files from partials
VERSION::20260113
GROUP::Checkpoint
"""
import os
import folder_paths
import nodes

class illumoraeCheckpointLoaderByStringDirty:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": ("STRING", {"default": ""}),
            },
            "optional": {
                "DEBUG_MODE": ("BOOLEAN", {"default": False}),
                "safe_mode": ("BOOLEAN", {"default": True}),
                "file_extensions": ("STRING", {"default": ".safetensors,.sft"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "ckpt_filename")
    FUNCTION = "load_checkpoint"
    CATEGORY = "illumorae"
    DESCRIPTION = "Loads a checkpoint by fuzzy matching the text input finds available checkpoint files from partials"

    @staticmethod
    def debug_message(msg, DEBUG_MODE):
        if DEBUG_MODE:
            print(f"[illumoraeCheckpointLoaderByStringDirty][DEBUG] {msg}")

    @staticmethod
    def _get_all_checkpoints_recursive_all_dirs(base_dirs, exts=(".safetensors", ".sft")):
        """
        Recursively collect all checkpoint files under all base_dirs, returning (rel_path, base_dir) tuples.
        exts: tuple of extensions (with leading dot)
        """
        files = []
        for base_dir in base_dirs:
            for root, dirs, filenames in os.walk(base_dir):
                for f in filenames:
                    if any(f.endswith(ext) for ext in exts):
                        rel_path = os.path.relpath(os.path.join(root, f), base_dir)
                        # Normalize to use forward slashes for matching
                        files.append((rel_path.replace("\\", "/"), base_dir))
        return files

    @classmethod
    def find_matching_filename(cls, input_string, filenames_with_dirs, DEBUG_MODE=False, preferred_exts=None):
        """
        Robustly search for a checkpoint file matching the input string, regardless of slashes, case, or path format.
        Tries all reasonable strategies (full path, filename, base name, partial match) in a case-insensitive way.
        filenames_with_dirs: list of (rel_path, base_dir)
        Returns: (rel_path, base_dir)
        """
        def norm(s):
            return s.replace("\\", "/").lower()

        def ext_rank(rel_path):
            if not preferred_exts:
                return 0
            ext = os.path.splitext(rel_path)[1].lower()
            try:
                return tuple(e.lower() for e in preferred_exts).index(ext)
            except ValueError:
                return len(preferred_exts)

        def pick_best(matches, match_type):
            if not matches:
                return None

            ranked = sorted(
                matches,
                key=lambda x: (
                    ext_rank(x[0]),
                    len(x[0]),
                    x[0].lower(),
                    x[1].lower() if isinstance(x[1], str) else str(x[1]).lower(),
                ),
            )
            best = ranked[0]

            if len(ranked) > 1:
                b0 = ranked[0]
                b1 = ranked[1]
                if (
                    ext_rank(b0[0]),
                    len(b0[0]),
                    b0[0].lower(),
                    b0[1].lower() if isinstance(b0[1], str) else str(b0[1]).lower(),
                ) == (
                    ext_rank(b1[0]),
                    len(b1[0]),
                    b1[0].lower(),
                    b1[1].lower() if isinstance(b1[1], str) else str(b1[1]).lower(),
                ):
                    raise ValueError(
                        f"Ambiguous checkpoint match for '{input_string}' ({match_type}). "
                        "Provide a more specific name/path."
                    )

            cls.debug_message(
                f"Selected checkpoint match ({match_type}): {best[0]} in {best[1]}",
                DEBUG_MODE,
            )
            return best

        input_string_norm = norm(input_string)
        input_filename_norm = norm(os.path.basename(input_string))
        input_base_norm, _ = os.path.splitext(input_filename_norm)

        cls.debug_message(f"Searching for checkpoint: input_string='{input_string}'", DEBUG_MODE)
        cls.debug_message(f"Available filenames: {[f for f, _ in filenames_with_dirs]}", DEBUG_MODE)
        cls.debug_message(f"Normalized input string: {input_string_norm}", DEBUG_MODE)
        cls.debug_message(f"Normalized input filename: {input_filename_norm}", DEBUG_MODE)
        cls.debug_message(f"Normalized input base: {input_base_norm}", DEBUG_MODE)

        # Normalize all filenames once
        normed_filenames = [
            (rel_path, base_dir, norm(rel_path), norm(os.path.basename(rel_path)), os.path.splitext(norm(os.path.basename(rel_path)))[0])
            for rel_path, base_dir in filenames_with_dirs
        ]

        # 1. Exact relative path match (case-insensitive)
        matches = [(rel_path, base_dir) for rel_path, base_dir, fn_norm, _, _ in normed_filenames if input_string_norm == fn_norm]
        picked = pick_best(matches, "exact relative path")
        if picked:
            return picked

        # 2. Exact filename match (case-insensitive)
        matches = [(rel_path, base_dir) for rel_path, base_dir, _, fn_base_norm, _ in normed_filenames if input_filename_norm == fn_base_norm]
        picked = pick_best(matches, "exact filename")
        if picked:
            return picked

        # 3. Base name match (case-insensitive)
        matches = [(rel_path, base_dir) for rel_path, base_dir, _, _, base_norm in normed_filenames if input_base_norm == base_norm]
        picked = pick_best(matches, "base name")
        if picked:
            return picked

        # 4. Partial filename match (case-insensitive)
        matches = [(rel_path, base_dir) for rel_path, base_dir, fn_norm, _, _ in normed_filenames if input_filename_norm and input_filename_norm in fn_norm]
        picked = pick_best(matches, "partial filename")
        if picked:
            return picked

        # 5. Partial path match (case-insensitive, e.g. input_string is a substring of the path)
        matches = [(rel_path, base_dir) for rel_path, base_dir, fn_norm, _, _ in normed_filenames if input_string_norm and input_string_norm in fn_norm]
        picked = pick_best(matches, "partial path")
        if picked:
            return picked

        cls.debug_message(f"No match found for '{input_string}'", DEBUG_MODE)
        raise FileNotFoundError(f"File '{input_string}' not found in checkpoint directories.")

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True, DEBUG_MODE=False, safe_mode=True, file_extensions=".safetensors,.sft"):
        self.debug_message(f"load_checkpoint called with ckpt_name='{ckpt_name}', output_vae={output_vae}, output_clip={output_clip}, DEBUG_MODE={DEBUG_MODE}, safe_mode={safe_mode}, file_extensions='{file_extensions}'", DEBUG_MODE)

        if safe_mode:
            return self.load_checkpoint_safe(ckpt_name, output_vae=output_vae, output_clip=output_clip, DEBUG_MODE=DEBUG_MODE)

        # Parse file_extensions string to tuple
        exts = tuple(ext.strip() if ext.strip().startswith(".") else "." + ext.strip() for ext in file_extensions.split(",") if ext.strip())
        self.debug_message(f"Using file extensions: {exts}", DEBUG_MODE)
        # Collect all checkpoint files from all registered directories
        checkpoints_dirs = folder_paths.get_folder_paths("checkpoints")
        filenames_with_dirs = self._get_all_checkpoints_recursive_all_dirs(checkpoints_dirs, exts=exts)
        self.debug_message(f"Found {len(filenames_with_dirs)} checkpoint files in all search paths (recursive).", DEBUG_MODE)
        rel_path, base_dir = self.find_matching_filename(ckpt_name, filenames_with_dirs, DEBUG_MODE, preferred_exts=exts)
        self.debug_message(f"Resolved checkpoint filename: {rel_path} in {base_dir}", DEBUG_MODE)
        loader = nodes.CheckpointLoaderSimple()
        model, clip, vae = loader.load_checkpoint(rel_path)
        self.debug_message(f"Checkpoint loaded: model={type(model)}, clip={type(clip)}, vae={type(vae)}", DEBUG_MODE)
        return model, clip, vae, rel_path

    def load_checkpoint_safe(self, ckpt_name, output_vae=True, output_clip=True, DEBUG_MODE=False):
        """
        Load a checkpoint ONLY if it is a valid safetensors file (safe mode).
        This avoids loading pickle-based files (ckpt) which could contain malware.
        """
        self.debug_message(f"load_checkpoint_safe called with ckpt_name='{ckpt_name}', output_vae={output_vae}, output_clip={output_clip}, DEBUG_MODE={DEBUG_MODE}", DEBUG_MODE)
        try:
            import safetensors
            from safetensors.torch import load_file as safetensors_load_file
        except ImportError:
            self.debug_message("[SAFE MODE] safetensors package not installed! Cannot verify file. Aborting.", True)
            raise ImportError("safetensors package is required for safe loading mode.")

        # Only allow safetensors-style extensions
        exts = (".safetensors", ".sft")
        self.debug_message(f"[SAFE MODE] Only accepting extensions: {exts}", DEBUG_MODE)
        checkpoints_dirs = folder_paths.get_folder_paths("checkpoints")
        filenames_with_dirs = self._get_all_checkpoints_recursive_all_dirs(checkpoints_dirs, exts=exts)
        self.debug_message(f"[SAFE MODE] Found {len(filenames_with_dirs)} safetensors files in all search paths (recursive).", DEBUG_MODE)
        rel_path, base_dir = self.find_matching_filename(ckpt_name, filenames_with_dirs, DEBUG_MODE, preferred_exts=exts)
        full_path = os.path.join(base_dir, rel_path)
        self.debug_message(f"[SAFE MODE] Resolved checkpoint filename: {rel_path} in {base_dir}", DEBUG_MODE)
        # Validate safetensors file
        try:
            # Will raise if file is corrupt or not a valid safetensors file
            safetensors_load_file(full_path, device="cpu")
            self.debug_message(f"[SAFE MODE] File '{full_path}' is a valid safetensors file.", DEBUG_MODE)
        except Exception as e:
            self.debug_message(f"[SAFE MODE] File '{full_path}' is NOT a valid safetensors file: {e}", True)
            raise ValueError(f"File '{full_path}' is not a valid safetensors file: {e}")
        loader = nodes.CheckpointLoaderSimple()
        model, clip, vae = loader.load_checkpoint(rel_path)
        self.debug_message(f"[SAFE MODE] Checkpoint loaded: model={type(model)}, clip={type(clip)}, vae={type(vae)}", DEBUG_MODE)
        return model, clip, vae, rel_path

NODE_CLASS_MAPPINGS = {
    'illumoraeCheckpointLoaderByStringDirty': illumoraeCheckpointLoaderByStringDirty,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeCheckpointLoaderByStringDirty': 'Checkpoint Loader By String Dirty',
}
