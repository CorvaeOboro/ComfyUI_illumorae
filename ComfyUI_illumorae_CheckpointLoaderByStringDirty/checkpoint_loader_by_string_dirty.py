"""
Checkpoint Loader By String Dirty - a ComfyUI Custom Node

Loads a Diffusion checkpoint by matching a partial string input to available checkpoint files.
supporting full paths, relative paths, or filenames 

Inputs:
    ckpt_name: The name or path of the checkpoint to load (string).
    DEBUG_MODE: Enable debug output
    safe_mode: only safetensors
    file_extensions: comma-separated list of extensions to search (default .safetensors,.sft)

Outputs:
    model: The loaded model object.
    clip: The loaded CLIP object.
    vae: The loaded VAE object.
    ckpt_filename: The resolved checkpoint filename (string).

TITLE::Checkpoint Loader By String Dirty
DESCRIPTIONSHORT::Loads a checkpoint by fuzzy matching the text input finds available checkpoint files from partials
GROUP::Checkpoint
GROUPORDER::2
LISTORDER::1
IMAGE::comfyui_illumorae_checkpoint_loader_by_string_dirty.png
STATUS::working
VERSION::20260815
"""
import json
import os
import folder_paths
import nodes


#region MODDEBUG
# Module-level debug printer shared by the class methods and the matcher.
def _debug_message(msg, debug_mode):
    """Print a debug line tagged with the node name when debug_mode is truthy."""
    if debug_mode:
        print(f"[illumoraeCheckpointLoaderByStringDirtyNode][DEBUG] {msg}")
#endregion


class illumoraeCheckpointLoaderByStringDirtyNode:

    #region CMETA
    # ComfyUI-facing metadata: input schema, return types, display fields.
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
    #endregion

    #region CDISCOVER
    # Filesystem discovery: walk registered checkpoint dirs and parse the
    # user-supplied extension string into a normalized tuple.
    @staticmethod
    def _get_all_checkpoints_recursive_all_dirs(base_dirs, exts=(".safetensors", ".sft")):
        """
        Recursively collect all checkpoint files under all base_dirs, returning (rel_path, base_dir) tuples.
        exts: tuple of extensions (with leading dot)
        """
        # Lowercase extensions once so the filter is case-insensitive on
        # platforms where filenames may carry uppercase extensions.
        exts_lower = tuple(e.lower() for e in exts)
        files = []
        for base_dir in base_dirs:
            for root, _, filenames in os.walk(base_dir):
                for f in filenames:
                    if any(f.lower().endswith(ext) for ext in exts_lower):
                        rel_path = os.path.relpath(os.path.join(root, f), base_dir)
                        # Normalize to use forward slashes for matching
                        files.append((rel_path.replace("\\", "/"), base_dir))
        return files

    @staticmethod
    def _parse_extensions(file_extensions):
        """Parse a comma-separated extension string into a normalized tuple.

        Each entry gets a leading dot and is lowercased. Empty entries are
        dropped.
        """
        return tuple(
            (ext.strip() if ext.strip().startswith(".") else "." + ext.strip()).lower()
            for ext in file_extensions.split(",")
            if ext.strip()
        )
    #endregion

    #region CVALIDATE
    # Safetensors header-only validation: confirms a file is a well-formed
    # safetensors archive without loading any tensor data.
    @staticmethod
    def _validate_safetensors_header(full_path):
        """Validate that ``full_path`` is a well-formed safetensors file by
        reading only its header (8-byte little-endian length + JSON blob),
        without loading any tensor data. Raises ``ValueError`` if the file is
        not a valid safetensors archive.
        """
        class _Invalid(Exception):
            """Sentinel raised inside the try block for validation failures."""

        try:
            with open(full_path, "rb") as fh:
                raw_len = fh.read(8)
                if len(raw_len) < 8:
                    raise _Invalid("file too small to contain a safetensors header")
                header_len = int.from_bytes(raw_len, "little")
                # Sanity cap: a non-safetensors file (e.g. a pickle) may
                # produce an absurd header length from its leading bytes.
                # Real safetensors headers are JSON and stay well under this.
                if header_len > 256 * 1024 * 1024:
                    raise _Invalid(f"declared header length {header_len} exceeds sane maximum")
                header_bytes = fh.read(header_len)
                if len(header_bytes) < header_len:
                    raise _Invalid("file truncated before end of declared header")
                json.loads(header_bytes)
        except (OSError, json.JSONDecodeError, _Invalid) as e:
            raise ValueError(f"File '{full_path}' is not a valid safetensors file: {e}") from e
    #endregion

    #region CMATCH
    # Core fuzzy matcher: resolves a free-text input to a (rel_path, base_dir)
    # tuple via ordered case-insensitive strategies, with extension preference
    # ranking and ambiguity detection.
    @staticmethod
    def find_matching_filename(input_string, filenames_with_dirs, DEBUG_MODE=False, preferred_exts=None):
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

            # Meaningful ranking criteria only: extension preference, then
            # path length (shorter = more specific). The path string is used
            # solely as a deterministic final tiebreaker and is excluded from
            # the ambiguity comparison so that two distinct files with equal
            # rank actually raise instead of being silently ordered.
            def rank_key(x):
                return (ext_rank(x[0]), len(x[0]))

            def tiebreak_key(x):
                return (
                    x[0].lower(),
                    x[1].lower() if isinstance(x[1], str) else str(x[1]).lower(),
                )

            ranked = sorted(matches, key=lambda x: (rank_key(x), tiebreak_key(x)))
            best = ranked[0]

            if len(ranked) > 1 and rank_key(ranked[0]) == rank_key(ranked[1]):
                raise ValueError(
                    f"Ambiguous checkpoint match for '{input_string}' ({match_type}). "
                    "Provide a more specific name/path."
                )

            _debug_message(
                f"Selected checkpoint match ({match_type}): {best[0]} in {best[1]}",
                DEBUG_MODE,
            )
            return best

        input_string_norm = norm(input_string)
        input_filename_norm = norm(os.path.basename(input_string))
        input_base_norm, _ = os.path.splitext(input_filename_norm)

        # Empty / whitespace-only input cannot match anything; raise early
        # with a clear message rather than falling through every strategy.
        if not input_string_norm.strip():
            raise ValueError(
                "ckpt_name is empty; provide a checkpoint name or path."
            )

        _debug_message(f"Searching for checkpoint: input_string='{input_string}'", DEBUG_MODE)
        _debug_message(f"Available filenames: {[f for f, _ in filenames_with_dirs]}", DEBUG_MODE)
        _debug_message(f"Normalized input string: {input_string_norm}", DEBUG_MODE)
        _debug_message(f"Normalized input filename: {input_filename_norm}", DEBUG_MODE)
        _debug_message(f"Normalized input base: {input_base_norm}", DEBUG_MODE)

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

        # 4. Partial filename match (case-insensitive) - match the input
        # basename as a substring of each file's *basename* only, not the
        # full path, so a short input does not match unrelated nested files.
        matches = [(rel_path, base_dir) for rel_path, base_dir, _, fn_base_norm, _ in normed_filenames if input_filename_norm and input_filename_norm in fn_base_norm]
        picked = pick_best(matches, "partial filename")
        if picked:
            return picked

        # 5. Partial path match (case-insensitive, e.g. input_string is a substring of the path)
        matches = [(rel_path, base_dir) for rel_path, base_dir, fn_norm, _, _ in normed_filenames if input_string_norm and input_string_norm in fn_norm]
        picked = pick_best(matches, "partial path")
        if picked:
            return picked

        _debug_message(f"No match found for '{input_string}'", DEBUG_MODE)
        raise FileNotFoundError(f"File '{input_string}' not found in checkpoint directories.")
    #endregion

    #region CLOAD
    # Entry points invoked by ComfyUI. load_checkpoint dispatches to safe or
    # dirty mode; load_checkpoint_safe validates the safetensors header before
    # delegating to the core CheckpointLoaderSimple.
    def load_checkpoint(self, ckpt_name, DEBUG_MODE=False, safe_mode=True, file_extensions=".safetensors,.sft"):
        _debug_message(f"load_checkpoint called with ckpt_name='{ckpt_name}', DEBUG_MODE={DEBUG_MODE}, safe_mode={safe_mode}, file_extensions='{file_extensions}'", DEBUG_MODE)

        if safe_mode:
            return self.load_checkpoint_safe(ckpt_name, DEBUG_MODE=DEBUG_MODE, file_extensions=file_extensions)

        exts = self._parse_extensions(file_extensions)
        _debug_message(f"Using file extensions: {exts}", DEBUG_MODE)
        # Collect all checkpoint files from all registered directories
        checkpoints_dirs = folder_paths.get_folder_paths("checkpoints")
        filenames_with_dirs = self._get_all_checkpoints_recursive_all_dirs(checkpoints_dirs, exts=exts)
        _debug_message(f"Found {len(filenames_with_dirs)} checkpoint files in all search paths (recursive).", DEBUG_MODE)
        rel_path, base_dir = self.find_matching_filename(ckpt_name, filenames_with_dirs, DEBUG_MODE, preferred_exts=exts)
        _debug_message(f"Resolved checkpoint filename: {rel_path} in {base_dir}", DEBUG_MODE)
        loader = nodes.CheckpointLoaderSimple()
        model, clip, vae = loader.load_checkpoint(rel_path)
        _debug_message(f"Checkpoint loaded: model={type(model)}, clip={type(clip)}, vae={type(vae)}", DEBUG_MODE)
        return model, clip, vae, rel_path

    def load_checkpoint_safe(self, ckpt_name, DEBUG_MODE=False, file_extensions=".safetensors,.sft"):
        """
        Load a checkpoint ONLY if it is a valid safetensors file (safe mode).
        This avoids loading pickle-based files (ckpt) which could contain malware.

        ``file_extensions`` is intersected with the safetensors-only set so the
        input remains observable while safety is enforced: only
        ``.safetensors`` / ``.sft`` entries are kept.
        """
        _debug_message(f"load_checkpoint_safe called with ckpt_name='{ckpt_name}', DEBUG_MODE={DEBUG_MODE}, file_extensions='{file_extensions}'", DEBUG_MODE)

        # Restrict to safetensors-style extensions. Intersect the user-provided
        # extensions with the safe set so the input still has observable effect.
        safe_exts = (".safetensors", ".sft")
        requested = self._parse_extensions(file_extensions)
        exts = tuple(e for e in requested if e in safe_exts) or safe_exts
        _debug_message(f"[SAFE MODE] Only accepting extensions: {exts}", DEBUG_MODE)
        checkpoints_dirs = folder_paths.get_folder_paths("checkpoints")
        filenames_with_dirs = self._get_all_checkpoints_recursive_all_dirs(checkpoints_dirs, exts=exts)
        _debug_message(f"[SAFE MODE] Found {len(filenames_with_dirs)} safetensors files in all search paths (recursive).", DEBUG_MODE)
        rel_path, base_dir = self.find_matching_filename(ckpt_name, filenames_with_dirs, DEBUG_MODE, preferred_exts=exts)
        _debug_message(f"[SAFE MODE] Resolved checkpoint filename: {rel_path} in {base_dir}", DEBUG_MODE)

        # Resolve the exact path the loader will use, so the validated file and
        # the loaded file are the same object. folder_paths.get_full_path is
        # what nodes.CheckpointLoaderSimple calls internally; falling back to
        # the discovered (base_dir, rel_path) covers unregistered directories.
        resolved_full_path = folder_paths.get_full_path("checkpoints", rel_path)
        if resolved_full_path is None:
            resolved_full_path = os.path.join(base_dir, rel_path)
        _debug_message(f"[SAFE MODE] Validating resolved path: {resolved_full_path}", DEBUG_MODE)

        # Header-only validation: confirms the file is a well-formed
        # safetensors archive without loading any tensor data.
        self._validate_safetensors_header(resolved_full_path)
        _debug_message(f"[SAFE MODE] File '{resolved_full_path}' is a valid safetensors file.", DEBUG_MODE)

        loader = nodes.CheckpointLoaderSimple()
        model, clip, vae = loader.load_checkpoint(rel_path)
        _debug_message(f"[SAFE MODE] Checkpoint loaded: model={type(model)}, clip={type(clip)}, vae={type(vae)}", DEBUG_MODE)
        return model, clip, vae, rel_path
    #endregion


#region REGISTRY
# ComfyUI node registration mappings.
NODE_CLASS_MAPPINGS = {
    'illumoraeCheckpointLoaderByStringDirtyNode': illumoraeCheckpointLoaderByStringDirtyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeCheckpointLoaderByStringDirtyNode': 'Checkpoint Loader By String Dirty',
}
#endregion
