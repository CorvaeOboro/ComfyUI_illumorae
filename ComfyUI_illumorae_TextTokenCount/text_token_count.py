"""
TITLE::Text Token Count
DESCRIPTIONSHORT::Counts tokens for a string using a CLIP tokenizer if available, otherwise estimates.
VERSION::20260814
IMAGE::comfyui_illumorae_text_token_count.png
GROUP::Text
GROUPORDER::5
LISTORDER::100
STATUS::working
"""

class illumoraeTextTokenCountNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "threshold": ("INT", {"default": 700, "min": 0, "max": 99999}),
            },
            "optional": {"clip": ("CLIP", )}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("token_count",)
    FUNCTION = "count_tokens"
    CATEGORY = "illumorae"
    OUTPUT_NODE = True
    DESCRIPTION = "Counts tokens for a string using a CLIP tokenizer if available, otherwise estimates."

    def count_tokens(self, text, threshold, clip=None):
        """
        Count tokens using CLIP's tokenizer if possible, otherwise estimate.
        Args:
            text (str): The string to count tokens for.
            threshold (int): Threshold value; display turns threshold_color when exceeded.
            clip (optional): CLIP model object with a tokenizer.
        Returns:
            tuple: (token_count_as_string,)
        """
        token_count = None
        if clip is not None:
            try:
                # Try common CLIP tokenizer patterns
                if hasattr(clip, 'tokenizer') and hasattr(clip.tokenizer, 'tokenize'):
                    tokens = clip.tokenizer.tokenize(text)
                    token_count = len(tokens[0]) if hasattr(tokens, '__getitem__') else len(tokens)
                elif hasattr(clip, 'tokenize'):
                    tokens = clip.tokenize(text)
                    token_count = len(tokens[0]) if hasattr(tokens, '__getitem__') else len(tokens)
            except Exception as e:
                token_count = None
        if token_count is None:
            # Fallback: Simple LLM-like token estimation
            word_tokens = len(text.split())
            char_tokens = max(0, int(len(text) / 4))
            token_count = word_tokens + char_tokens
        token_count_str = str(token_count)
        # Return both the output tuple (for downstream nodes) and a `ui`
        # payload so the JS onExecuted handler can update the display widget.
        return {"ui": {"token_count": [token_count_str]}, "result": (token_count_str,)}


NODE_CLASS_MAPPINGS = {
    'illumoraeTextTokenCountNode': illumoraeTextTokenCountNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeTextTokenCountNode': 'Text Token Count',
}
