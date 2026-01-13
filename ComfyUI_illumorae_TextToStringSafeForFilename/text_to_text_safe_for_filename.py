"""
TITLE::Text To Filename Safe Text
DESCRIPTIONSHORT::Converts text into a filename-safe string (Windows-safe) with normalization, character filtering, and length truncation.
VERSION::20260113
IMAGE::comfyui_illumorae_text_to_string_filename_safe.png
GROUP::Text
"""
import re
import string
import unicodedata

class illumoraeTextToTextSafeForFilename:
    """
    This node converts a text input into a filename-safe variant by:
      - Replacing or removing invalid characters
      - Handling Windows-reserved filenames
      - Optionally normalizing Unicode
      - Collapsing underscores
      - Truncating length if needed
    """

DIRECT_REPLACEMENTS = {
    " ": "_",
    "\n": "_",
    "\r": "_",
    "(": "_",
    ")": "_",
    ":": "_",
    "<": "_",
    ">": "_",  
    "\\": "_",
    "/": "_",
    "*": "_",
    "?": "_",
    "\"": "_",
    "|": "_",
    "-": "_",
}

# ---- BAD FILENAME CHARACTERS & CATEGORIES ----
BAD_FILENAME_CHAR_SET = set([
    '\\', '/', ':', '*', '?', '"', '<', '>', '|', '@', '~', '`', '^', '=', '+', '[', ']', '{', '}', '(', ')', ';', ',',
    '…', '–', '—', '•', '“', '”', '‘', '’', '′', '″', '‽', '†', '‡', '∑', '∯', '∫', '∉', '∈', '∋', '∂', '∇', '∩', '∪', '∧', '∨', '∃', '∄', '∅', '∏', '−', '∓', '∔', '∗', '∘', '√', '∞', '∟', '∠', '∡', '∢', '∣', '∤', '∥', '∦', '∧', '∨', '∩', '∪', '∫', '∬', '∭', '∮', '∯', '∰', '∱', '∲', '∳', '∴', '∵', '∶', '∷', '∸', '∹', '∺', '∻', '∼', '∽', '∾', '≁', '≃', '≄', '≅', '≆', '≇', '≈', '≉', '≊', '≋', '≌', '≍', '≎', '≏', '≐', '≑', '≒', '≓', '≔', '≕', '≖', '≗', '≘', '≙', '≚', '≛', '≜', '≝', '≞', '≟', '≠', '≡', '≢', '≣', '≤', '≥', '≦', '≧', '≨', '≩', '≪', '≫', '≬', '≭', '≮', '≯', '≰', '≱', '≲', '≳', '≴', '≵', '≶', '≷', '≸', '≹', '≺', '≻', '≼', '≽', '≾', '≿', '⊀', '⊁', '⊂', '⊃', '⊄', '⊅', '⊆', '⊇', '⊈', '⊉', '⊊', '⊋', '⊌', '⊍', '⊎', '⊏', '⊐', '⊑', '⊒', '⊓', '⊔', '⊕', '⊖', '⊗', '⊘', '⊙', '⊚', '⊛', '⊜', '⊝', '⊞', '⊟', '⊠', '⊡', '⊢', '⊣', '⊤', '⊥', '⊦', '⊧', '⊨', '⊩', '⊪', '⊫', '⊬', '⊭', '⊮', '⊯', '⊰', '⊱', '⊲', '⊳', '⊴', '⊵', '⊶', '⊷', '⊸', '⊹', '⊺', '⊻', '⊼', '⊽', '⊾', '⊿', '⋀', '⋁', '⋂', '⋃', '⋄', '⋅', '⋆', '⋇', '⋈', '⋉', '⋊', '⋋', '⋌', '⋍', '⋎', '⋏', '⋐', '⋑', '⋒', '⋓', '⋔', '⋕', '⋖', '⋗', '⋘', '⋙', '⋚', '⋛', '⋜', '⋝', '⋞', '⋟', '⋠', '⋡', '⋢', '⋣', '⋤', '⋥', '⋦', '⋧', '⋨', '⋩', '⋪', '⋫', '⋬', '⋭', '⋮', '⋯', '⋰', '⋱', '⋲', '⋳', '⋴', '⋵', '⋶', '⋷', '⋸', '⋹', '⋺', '⋻', '⋼', '⋽', '⋾', '⋿', '⌀', '⌁', '⌂', '⌃', '⌄', '⌅', '⌆', '⌇', '⌈', '⌉', '⌊', '⌋', '⌌', '⌍', '⌎', '⌏', '⌐', '⌑', '⌒', '⌓', '⌔', '⌕', '⌖', '⌗', '⌘', '⌙', '⌚', '⌛', '⌜', '⌝', '⌞', '⌟', '⌠', '⌡', '⌢', '⌣', '⌤', '⌥', '⌦', '⌧', '⌨', '〈', '〉', '⌫', '⌬', '⌭', '⌮', '⌯', '⌰', '⌱', '⌲', '⌳', '⌴', '⌵', '⌶', '⌷', '⌸', '⌹', '⌺', '⌻', '⌼', '⌽', '⌾', '⌿', '⍀', '⍁', '⍂', '⍃', '⍄', '⍅', '⍆', '⍇', '⍈', '⍉', '⍊', '⍋', '⍌', '⍍', '⍎', '⍏', '⍐', '⍑', '⍒', '⍓', '⍔', '⍕', '⍖', '⍗', '⍘', '⍙', '⍚', '⍛', '⍜', '⍝', '⍞', '⍟', '⍠', '⍡', '⍢', '⍣', '⍤', '⍥', '⍦', '⍧', '⍨', '⍩', '⍪', '⍫', '⍬', '⍭', '⍮', '⍯', '⍰', '⍱', '⍲', '⍳', '⍴', '⍵', '⍶', '⍷', '⍸', '⍹', '⍺', '⍻', '⍼', '⍽', '⍾', '⍿', '⎀', '⎁', '⎂', '⎃', '⎄', '⎅', '⎆', '⎇', '⎈', '⎉', '⎊', '⎋', '⎌', '⎍', '⎎', '⎏', '⎐', '⎑', '⎒', '⎓', '⎔', '⎕', '⎖', '⎗', '⎘', '⎙', '⎚', '⎛', '⎜', '⎝', '⎞', '⎟', '⎠', '⎡', '⎢', '⎣', '⎤', '⎥', '⎦', '⎧', '⎨', '⎩', '⎪', '⎫', '⎬', '⎭', '⎮', '⎯', '⎰', '⎱', '⎲', '⎳', '⎴', '⎵', '⎶', '⎷', '⎸', '⎹', '⎺', '⎻', '⎼', '⎽', '⎾', '⎿', '⏀', '⏁', '⏂', '⏃', '⏄', '⏅', '⏆', '⏇', '⏈', '⏉', '⏊', '⏋', '⏌', '⏍', '⏎', '⏏', '⏐', '⏑', '⏒', '⏓', '⏔', '⏕', '⏖', '⏗', '⏘', '⏙', '⏚', '⏛', '⏜', '⏝', '⏞', '⏟', '⏠', '⏡', '⏢', '⏣', '⏤', '⏥', '⏦', '⏧', '⏨', '⏩', '⏪', '⏫', '⏬', '⏭', '⏮', '⏯', '⏰', '⏱', '⏲', '⏳', '⏴', '⏵', '⏶', '⏷', '⏸', '⏹', '⏺', '⏻', '⏼', '⏽', '⏾', '⏿'
])
BAD_UNICODE_CATEGORIES = ("C", "M", "S")


class illumoraeTextToTextSafeForFilename:
    DIRECT_REPLACEMENTS = {
        " ": "_",
        "\n": "_",
        "\r": "_",
        "(": "_",
        ")": "_",
        ":": "_",
        "<": "_",
        ">": "_",  
        "\\": "_",
        "/": "_",
        "*": "_",
        "?": "_",
        "\"": "_",
        "|": "_",
        "-": "_",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_in": ("STRING",),
                "max_length": ("INT", {"default": 150, "min": 1, "max": 250}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_string"
    CATEGORY = "illumorae"
    DESCRIPTION = "Converts a text input into a filename-safe variant."

    # ---- CONFIGURATIONS ----
    # Characters to remove outright , no comma , no apostrophe
    remove_characters = [
        ",",
        "'",
    ]

    # Reserved Windows filenames
    reserved_names = {
        "con", "prn", "aux", "nul",
        "com1", "com2", "com3", "com4", "com5",
        "com6", "com7", "com8", "com9",
        "lpt1", "lpt2", "lpt3", "lpt4", "lpt5",
        "lpt6", "lpt7", "lpt8", "lpt9",
    }

    def process_string(self, string_in, max_length=150):

        # Normalize Unicode to decompose accents
        string_in = unicodedata.normalize("NFKD", string_in)
        
        # Step A: Direct replacements
        for original_char, replacement_char in self.DIRECT_REPLACEMENTS.items():
            string_in = string_in.replace(original_char, replacement_char)

        # Step B: Remove specifically designated characters
        for char_to_remove in self.remove_characters:
            string_in = string_in.replace(char_to_remove, "")

        # Step C: Remove any leftover control characters or punctuation that we do not want. We'll keep underscore though.
        # 1) Remove control chars (ASCII 0-31 and 127)
        control_chars = ''.join(chr(c) for c in range(32)) + chr(127)
        for cc in control_chars:
            string_in = string_in.replace(cc, "")

        # Step D: Remove or replace problematic Unicode/symbol characters
        string_in = self.remove_bad_filename_chars(string_in)

        # 2) Remove all punctuation except underscore
        punctuation_without_underscore = "".join(
            ch for ch in string.punctuation if ch != "_"
        )
        string_in = re.sub(f"[{re.escape(punctuation_without_underscore)}]+", "", string_in)

        # Step D: Collapse multiple underscores
        string_in = re.sub(r"_{2,}", "_", string_in)

        # Step E: Remove trailing/leading periods/spaces/underscores if needed
        # Windows doesn't handle trailing spaces or periods well:
        string_in = string_in.strip(" .")

        # Step F: If the entire cleaned string is one of the reserved Windows filenames, suffix  to avoid conflicts.
        if string_in.lower() in self.reserved_names:
            string_in = string_in + "_0"

        # Step G: Truncate to max length
        if len(string_in) > max_length:
            string_in = string_in[:max_length]

        # If after everything it's empty, give it a default name
        if not string_in:
            string_in = "error_empty_string"

        return (string_in,)

    def is_bad_char(self,c):
        if c == "_":
            return False
        cat = unicodedata.category(c)
        if c in BAD_FILENAME_CHAR_SET:
            return True
        if cat.startswith(BAD_UNICODE_CATEGORIES):
            return True
        if ord(c) > 127:
            return True
        return False

    def remove_bad_filename_chars(self,s):
        return ''.join(c for c in s if not self.is_bad_char(c))

# Register the node
NODE_CLASS_MAPPINGS = {
    "illumoraeTextToTextSafeForFilename": illumoraeTextToTextSafeForFilename,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeTextToTextSafeForFilename": "Text To Filename Safe Text",
}
