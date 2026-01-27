"""
TITLE::Text Token Count
DESCRIPTIONSHORT::Counts tokens for a string using a CLIP tokenizer if available, otherwise estimates.
VERSION::20260127
IMAGE::comfyui_illumorae_text_token_count.png
GROUP::Text
"""

class illumoraeTextTokenCountNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"text": ("STRING", {"default": ""})},
            "optional": {"clip": ("CLIP", )}
        }

    RETURN_TYPES = ("STRING", "LABEL")
    RETURN_NAMES = ("token_count", "status")
    FUNCTION = "count_tokens"
    CATEGORY = "illumorae"
    DESCRIPTION = "Counts tokens for a string using a CLIP tokenizer if available, otherwise estimates."

    def count_tokens(self, text, clip=None):
        """
        Count tokens using CLIP's tokenizer if possible, otherwise estimate.
        Args:
            text (str): The string to count tokens for.
            clip (optional): CLIP model object with a tokenizer.
        Returns:
            tuple: (token_count_as_string, status_label)
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
        status = f"Tokens: {token_count}"
        return str(token_count), status


NODE_CLASS_MAPPINGS = {
    'illumoraeTextTokenCountNode': illumoraeTextTokenCountNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeTextTokenCountNode': 'Text Token Count',
}
