"""
TITLE::Text Multiline FindColorized
DESCRIPTIONSHORT::Editable multiline text input with search/find highlighting and syntax-colored LoRA bracket tags.
VERSION::20260815
IMAGE::comfyui_illumorae_text_multiline_findcolorized.png
GROUP::Text
GROUPORDER::5
LISTORDER::51
STATUS::working
"""


class illumoraeTextMultilineFindColorizedNode:
    """
    A ComfyUI node that provides a multiline text input with search/find
    and colorized syntax highlighting, rendered directly in the node UI.

    Features:
    - Editable multiline text with live syntax highlighting.
    - Search / find bar that highlights matched text in orange.
    - Angle-bracket tags (e.g. <lora:example:0.2>) colored light blue.
    - Outputs plain STRING for downstream connections.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "illumorae"
    DESCRIPTION = (
        "Editable multiline text input with search/find highlighting "
        "and syntax-colored LoRA bracket tags."
    )

    def process(self, text: str) -> tuple:
        """Pass through the text value unchanged."""
        return (text,)


# ----------------------------------------------------------------------
# ComfyUI registration
# ----------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "illumoraeTextMultilineFindColorizedNode": illumoraeTextMultilineFindColorizedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeTextMultilineFindColorizedNode": "Text Multiline FindColorized",
}
