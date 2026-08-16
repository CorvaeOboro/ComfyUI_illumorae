"""
TITLE::Load Text File Graceful
DESCRIPTIONSHORT::Loads a text file from a path and returns its contents without crashing if the file is missing.
VERSION::20260816
IMAGE::comfyui_illumorae_load_text_graceful.png
GROUP::Load
GROUPORDER::3
LISTORDER::5
STATUS::working
"""
#region IMPORTS
import os
#endregion


class illumoraeLoadTextFileGracefulNode:
    """
    A ComfyUI node that loads text from a file.

    Lines whose first non-whitespace character is '#' are treated as comments
    and skipped; inline trailing '#' is preserved. The remaining lines are
    joined with newlines into a single output string. Original line content
    (including indentation) is preserved; only the line terminator is removed.

    If the file path is invalid or the file cannot be read, an empty string
    and a status describing the failure are returned (no crash). Non-UTF-8
    files are decoded lossily with errors='replace' so a usable string is
    still produced.

    Returns (text, status) where status is a short human-readable string.
    """

    #region CACHE
    # On-disk file signature folded into the IS_CHANGED cache key so editing
    # the text file refreshes the output even when file_path is stable.

    @classmethod
    def IS_CHANGED(cls, file_path="", **kwargs):
        # Re-run when file_path changes or when the file's on-disk state
        # changes (mtime or size). A missing/unreadable file yields a
        # distinct key so a later-created file is picked up.
        try:
            st = os.stat(file_path)
            return f"{file_path}|{st.st_mtime_ns}|{st.st_size}"
        except OSError:
            return f"{file_path}|<missing>"

    #endregion

    #region UI
    # ComfyUI interface declarations: input schema, return types, and node
    # metadata. Read by ComfyUI at registration time; not executed per run.

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "debug_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")  # text, status
    RETURN_NAMES = ("text", "status")
    FUNCTION = "load_file"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Loads a text file from a path and returns its contents without crashing if the file is missing."

    #endregion

    #region CORE
    # Main entry point and FUNCTION target. Validates the path, reads the
    # file (UTF-8 with a lossy fallback), strips comment lines, and returns
    # (text, status).

    def load_file(self, file_path="", debug_mode=False):
        """
        Load a text file and return its non-comment contents.

        1. If file_path does not exist or is not a file, return ("", status).
        2. Skip any line whose first non-whitespace character is '#'.
        3. Preserve original line text (indentation kept); only the line
           terminator is removed.
        4. Non-UTF-8 files are decoded lossily with errors='replace'.
        5. Return (text, status) where text is the non-comment lines joined
           with newlines and status is a short human-readable string.
        """
        # isfile implies existence, so a separate exists() check is redundant.
        if not os.path.isfile(file_path):
            print(f"[LoadTextFileGraceful] Warning: File not found: {file_path}")
            return "", f"File not found: {file_path}"

        # Read file contents. UTF-8 first; fall back to a lossy decode so a
        # non-UTF-8 text file still yields a usable string instead of an error.
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
            self._dprint(debug_mode, f"[LoadTextFileGraceful][DEBUG] Non-UTF-8 file, used lossy decode: {file_path}")
        except OSError as e:
            print(f"[LoadTextFileGraceful] Error reading file {file_path}: {e}")
            return "", f"Error reading file: {file_path} ({e})"

        # Split into lines and classify: comment vs. content. A line is a
        # comment when its first non-whitespace character is '#'; inline
        # trailing '#' is preserved. Original line text is kept (only the
        # line terminator is removed by splitlines) so indentation survives.
        lines = []
        comments = []
        for idx, line in enumerate(raw_text.splitlines(), 1):
            lstripped = line.lstrip()
            if lstripped.startswith('#'):
                comments.append(lstripped.strip())
                self._dprint(debug_mode, f"[LoadTextFileGraceful][DEBUG] Skipping comment line {idx}: {lstripped.strip()}")
            else:
                lines.append(line)
                self._dprint(debug_mode, f"[LoadTextFileGraceful][DEBUG] Loaded line {idx}: {line}")

        self._dprint(
            debug_mode,
            f"[LoadTextFileGraceful][DEBUG] Finished loading. "
            f"{len(lines)} content lines, {len(comments)} comment lines.",
        )
        if debug_mode and comments:
            print("[LoadTextFileGraceful][DEBUG] Comments found in file:")
            for comment in comments:
                print(f"    {comment}")

        # Join non-comment lines into a single string.
        text_output = "\n".join(lines)

        # Status distinguishes a file that loaded with content from one that
        # was empty after comment stripping, so the two are not confused with
        # each other or with the error/missing empty-string returns.
        if lines:
            status = f"Loaded: {file_path} ({len(lines)} lines)"
        else:
            status = f"Loaded (empty after comment strip): {file_path}"

        return text_output, status

    #endregion

    #region UTIL
    # Debug print helper: only emits output when debug_mode is True.

    def _dprint(self, debug_mode, *args, **kwargs):
        if debug_mode:
            print(*args, **kwargs)

    #endregion


#region MAPPING
# ComfyUI registration: node class + display name mappings exported via
# __init__.py so ComfyUI can discover and instantiate the node.
NODE_CLASS_MAPPINGS = {
    "illumoraeLoadTextFileGracefulNode": illumoraeLoadTextFileGracefulNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeLoadTextFileGracefulNode": "Load Text File Graceful",
}
#endregion
