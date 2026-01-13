"""
illumorae Checkpoint Loader By String Dirty - a ComfyUI Custom Node

Loads a Stable Diffusion checkpoint by matching a string input to available checkpoint files.
supporting full paths, relative paths, or filenames 

Inputs:
    ckpt_name: The name or path of the checkpoint to load (string).
    DEBUG_MODE: Enable debug output 

Outputs:
    model: The loaded model object.
    clip: The loaded CLIP object.
    vae: The loaded VAE object.
    ckpt_filename: The resolved checkpoint filename (string).

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
                "file_extensions": ("STRING", {"default": ".ckpt,.safetensors,.sft"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "ckpt_filename")
    FUNCTION = "load_checkpoint"
    CATEGORY = "illumorae"
    DESCRIPTION = "Loads a Stable Diffusion checkpoint by matching a string input to available checkpoint files. supporting full paths, relative paths, or filenames"

    @staticmethod
    def debug_message(msg, DEBUG_MODE):
        if DEBUG_MODE:
            print(f"[illumoraeCheckpointLoaderByStringDirty][DEBUG] {msg}")

    @staticmethod
    def _get_all_checkpoints_recursive_all_dirs(base_dirs, exts=(".ckpt", ".safetensors", ".sft")):
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
    def find_matching_filename(cls, input_string, filenames_with_dirs, DEBUG_MODE=False):
        """
        Robustly search for a checkpoint file matching the input string, regardless of slashes, case, or path format.
        Tries all reasonable strategies (full path, filename, base name, partial match) in a case-insensitive way.
        filenames_with_dirs: list of (rel_path, base_dir)
        Returns: (rel_path, base_dir)
        """
        def norm(s):
            return s.replace("\\", "/").lower()

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
        for rel_path, base_dir, fn_norm, _, _ in normed_filenames:
            if input_string_norm == fn_norm:
                cls.debug_message(f"Exact relative path match found: {rel_path} in {base_dir}", DEBUG_MODE)
                return rel_path, base_dir

        # 2. Exact filename match (case-insensitive)
        for rel_path, base_dir, _, fn_base_norm, _ in normed_filenames:
            if input_filename_norm == fn_base_norm:
                cls.debug_message(f"Exact filename match found: {rel_path} in {base_dir}", DEBUG_MODE)
                return rel_path, base_dir

        # 3. Base name match (case-insensitive)
        for rel_path, base_dir, _, _, base_norm in normed_filenames:
            if input_base_norm == base_norm:
                cls.debug_message(f"Base name match found: {rel_path} in {base_dir}", DEBUG_MODE)
                return rel_path, base_dir

        # 4. Partial filename match (case-insensitive)
        for rel_path, base_dir, fn_norm, _, _ in normed_filenames:
            if input_filename_norm in fn_norm:
                cls.debug_message(f"Partial filename match found: {rel_path} in {base_dir}", DEBUG_MODE)
                return rel_path, base_dir

        # 5. Partial path match (case-insensitive, e.g. input_string is a substring of the path)
        for rel_path, base_dir, fn_norm, _, _ in normed_filenames:
            if input_string_norm in fn_norm:
                cls.debug_message(f"Partial path match found: {rel_path} in {base_dir}", DEBUG_MODE)
                return rel_path, base_dir

        cls.debug_message(f"No match found for '{input_string}'", DEBUG_MODE)
        raise FileNotFoundError(f"File '{input_string}' not found in checkpoint directories.")

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True, DEBUG_MODE=False, file_extensions=".ckpt,.safetensors,.sft"):
        self.debug_message(f"load_checkpoint called with ckpt_name='{ckpt_name}', output_vae={output_vae}, output_clip={output_clip}, DEBUG_MODE={DEBUG_MODE}, file_extensions='{file_extensions}'", DEBUG_MODE)
        # Parse file_extensions string to tuple
        exts = tuple(ext.strip() if ext.strip().startswith(".") else "." + ext.strip() for ext in file_extensions.split(",") if ext.strip())
        self.debug_message(f"Using file extensions: {exts}", DEBUG_MODE)
        # Collect all checkpoint files from all registered directories
        checkpoints_dirs = folder_paths.get_folder_paths("checkpoints")
        filenames_with_dirs = self._get_all_checkpoints_recursive_all_dirs(checkpoints_dirs, exts=exts)
        self.debug_message(f"Found {len(filenames_with_dirs)} checkpoint files in all search paths (recursive).", DEBUG_MODE)
        rel_path, base_dir = self.find_matching_filename(ckpt_name, filenames_with_dirs, DEBUG_MODE)
        self.debug_message(f"Resolved checkpoint filename: {rel_path} in {base_dir}", DEBUG_MODE)
        loader = nodes.CheckpointLoaderSimple()
        # Pass only the relative path to the loader (ComfyUI expects relative to any registered checkpoint dir)
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

        # Only allow .safetensors extension
        exts = (".safetensors",)
        self.debug_message(f"[SAFE MODE] Only accepting extensions: {exts}", DEBUG_MODE)
        checkpoints_dirs = folder_paths.get_folder_paths("checkpoints")
        filenames_with_dirs = self._get_all_checkpoints_recursive_all_dirs(checkpoints_dirs, exts=exts)
        self.debug_message(f"[SAFE MODE] Found {len(filenames_with_dirs)} safetensors files in all search paths (recursive).", DEBUG_MODE)
        rel_path, base_dir = self.find_matching_filename(ckpt_name, filenames_with_dirs, DEBUG_MODE)
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
