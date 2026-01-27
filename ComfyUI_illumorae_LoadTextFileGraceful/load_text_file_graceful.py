"""
TITLE::Load Text File Graceful
DESCRIPTIONSHORT::Loads a text file from a path and returns its contents without crashing if the file is missing.
VERSION::20260127
IMAGE::comfyui_illumorae_load_text_graceful.png
GROUP::Load
"""
import os

class illumoraeLoadTextFileGracefulNode:
    """
    A ComfyUI node that loads text from a file. 
    Non-comment lines (lines that do not start with '#') are:
      - joined into a single output string, and 
      - also stored in a dictionary under the user-specified key.

    If the file path is invalid or the file cannot be read, an empty 
    string and empty list are returned gracefully (no crash).
    """

    def __init__(self):
        """
        Node initialization can hold default values or references 
        to external modules if necessary.
        """
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "debug_mode": ("BOOLEAN", {"default": False}),
            },
            "hidden": {},
        }

    # ComfyUI expects certain class variables that define how the node behaves
    RETURN_TYPES = ("STRING", "DICT", "LABEL")  # Add LABEL output for status
    RETURN_NAMES = ("text", "lines_dict", "status")
    FUNCTION = "load_file"            # The method in this class that is called
    CATEGORY = "illumorae"                 # Category name for where this node appears
    OUTPUT_NODE = False               # Whether this node can terminate a workflow
    DESCRIPTION = "Loads a text file from a path and returns its contents without crashing if the file is missing."

    def load_file(self, 
                  file_path="",
                  debug_mode=False,
                  *args, 
                  **kwargs):
        """
        Main logic to load a text file:

        1. The dictionary key is always the base filename (no extension).
        2. If file_path does not exist, return an empty string and empty dictionary.
        3. Skip any line that starts with '#'.
        4. Return a single string (joined with newlines), a dict {filename: [list_of_lines]}, and a status label.
        """
        # Check for file existence
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            print(f"[LoadTextFileGraceful] Warning: File not found: {file_path}")
            status = "File not found"
            return "", status

        # Attempt to read file lines, skipping comments
        lines = []
        comments = []
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        comments.append(stripped)
                        if debug_mode:
                            print(f"[LoadTextFileGraceful][DEBUG] Skipping comment line {idx}: {stripped}")
                    else:
                        lines.append(stripped)
                        if debug_mode:
                            print(f"[LoadTextFileGraceful][DEBUG] Loaded line {idx}: {stripped}")
        except Exception as e:
            print(f"[LoadTextFileGraceful] Error reading file {file_path}: {e}")
            status = f"Error reading file: {os.path.basename(file_path)}"
            return "", status

        if debug_mode:
            print(f"[LoadTextFileGraceful][DEBUG] Finished loading. {len(lines)} content lines, {len(comments)} comment lines.")
            if comments:
                print("[LoadTextFileGraceful][DEBUG] Comments found in file:")
                for comment in comments:
                    print(f"    {comment}")

        # Join non-comment lines into a single string
        text_output = "\n".join(lines)

        # Status: show loaded filename
        status = f"Loaded: {os.path.basename(file_path)}"

        # Return the text,  and status label
        return text_output, status

NODE_CLASS_MAPPINGS = {
    "illumoraeLoadTextFileGracefulNode": illumoraeLoadTextFileGracefulNode, # ComfyUI needs to know which classes to load when scanning your .py file
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeLoadTextFileGracefulNode": "Load Text File Graceful", # Provide a human-readable display name for your node
}
