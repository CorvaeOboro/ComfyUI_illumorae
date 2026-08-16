"""
TITLE::Text Show Multiline Color
DESCRIPTIONSHORT::Displays multiline text with syntax highlighting in a persistent, selectable widget. Gracefully handles missing or broken inputs.
VERSION::20260811
IMAGE::comfyui_illumorae_text_show_multiline_color.png
GROUP::Text
GROUPORDER::5
LISTORDER::81
STATUS::working
"""
import re
from html import escape


class illumoraeTextShowMultilineColorNode:
    """
    A ComfyUI node that displays text in a persistent, colorized multiline widget.

    Features:
    - Outputs plain STRING for downstream connections.
    - Renders syntax-highlighted HTML in the node's UI area (IDE-like coloring).
    - Text remains selectable and copyable (not an image).
    - Stores output persistently across workflow reloads (unlike blanking preview nodes).
    - Gracefully handles broken/missing inputs (None, dict, list, bypassed upstream nodes).

    Syntax highlighting covers:
    - Parentheses with depth-based coloring
    - Square brackets and curly braces
    - Numbers (integers, decimals, negatives)
    - <lora:...> and <embedding:...> tags
    - Single/double quoted strings
    - Comments (# and //)
    - Commas as separators
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "enable_highlighting": ("BOOLEAN", {"default": True}),
                "theme": (["dark", "light"], {"default": "dark"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = "illumorae"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Displays multiline text with syntax highlighting in a persistent, "
        "selectable widget. Gracefully handles missing or broken inputs."
    )

    # ------------------------------------------------------------------
    # Color palettes (Catppuccin Mocha / Latte inspired)
    # ------------------------------------------------------------------
    PALETTES = {
        "dark": {
            "bg": "#1e1e2e",
            "text": "#cdd6f4",
            "paren_1": "#f38ba8",   # pink
            "paren_2": "#fab387",   # peach
            "paren_3": "#a6e3a1",   # green
            "bracket": "#89b4fa",   # blue
            "brace": "#f9e2af",     # yellow
            "number": "#f5c2e7",    # mauve
            "lora_tag": "#94e2d5",  # teal
            "comment": "#6c7086",   # overlay0
            "comma": "#7f849c",     # overlay1
            "quote": "#b4befe",     # lavender
            "error": "#ff0000",
        },
        "light": {
            "bg": "#eff1f5",
            "text": "#4c4f69",
            "paren_1": "#d20f39",   # red
            "paren_2": "#fe640b",   # peach
            "paren_3": "#40a02b",   # green
            "bracket": "#1e66f5",   # blue
            "brace": "#df8e1d",     # yellow
            "number": "#8839ef",    # mauve
            "lora_tag": "#179299",  # teal
            "comment": "#8c8fa1",   # overlay1
            "comma": "#6c6f85",     # overlay0
            "quote": "#7287fd",     # lavender
            "error": "#d20f39",
        },
    }

    # ------------------------------------------------------------------
    # Input sanitisation
    # ------------------------------------------------------------------
    def _sanitize_input(self, text) -> str:
        """
        Convert broken / unexpected inputs into a displayable string.

        Handles:
        - None  (bypassed or disconnected upstream node)
        - dict  (upstream node returning a dict on a secondary output)
        - list / tuple
        - Anything else via str()
        """
        if text is None:
            return ""
        if isinstance(text, str):
            return text
        if isinstance(text, dict):
            # Many ComfyUI nodes emit dicts with a 'text' or 'string' key
            for key in ("text", "string", "prompt", "content", "result"):
                if key in text and isinstance(text[key], str):
                    return text[key]
            # Fallback: pretty-print the dict so the user sees what arrived
            try:
                import json
                return json.dumps(text, indent=2, ensure_ascii=False)
            except Exception:
                return str(text)
        if isinstance(text, (list, tuple)):
            if len(text) == 1 and isinstance(text[0], str):
                return text[0]
            return "\n".join(str(item) for item in text)
        return str(text)

    def _build_error_text(self, raw_input, theme: str) -> str:
        """Build a friendly HTML block when the input is None / truly missing."""
        p = self.PALETTES.get(theme, self.PALETTES["dark"])
        return (
            f'<pre style="margin:0;padding:12px;background:{p["bg"]};'
            f'color:{p["comment"]};font-family:monospace;font-size:13px;'
            f'line-height:1.5;white-space:pre-wrap;word-break:break-word;'
            f'border-radius:6px;">'
            f'[No input received - upstream node may be bypassed, '
            f'disconnected, or returned None.]\n\n'
            f'Raw value type: {type(raw_input).__name__}'
            f'</pre>'
        )

    def _build_plain_html(self, text: str, theme: str) -> str:
        """Build a plain (non-highlighted) HTML block."""
        p = self.PALETTES.get(theme, self.PALETTES["dark"])
        return (
            f'<pre style="margin:0;padding:12px;background:{p["bg"]};'
            f'color:{p["text"]};font-family:monospace;font-size:13px;'
            f'line-height:1.5;white-space:pre-wrap;word-break:break-word;'
            f'border-radius:6px;">'
            f'{escape(text)}'
            f'</pre>'
        )

    # ------------------------------------------------------------------
    # Syntax highlighting
    # ------------------------------------------------------------------
    def _highlight_text(self, text: str, theme: str) -> str:
        """Convert raw text into syntax-highlighted HTML."""
        p = self.PALETTES.get(theme, self.PALETTES["dark"])
        parts = [
            f'<pre style="margin:0;padding:12px;background:{p["bg"]};'
            f'color:{p["text"]};font-family:monospace;font-size:13px;'
            f'line-height:1.5;white-space:pre-wrap;word-break:break-word;'
            f'border-radius:6px;">'
        ]

        paren_stack = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]

            # ----------------------------------------------------------
            # 1. <lora:...>  and  <embedding:...> tags
            # ----------------------------------------------------------
            if ch == '<':
                m = re.match(r'<lora:[^>]*>', text[i:])
                if m:
                    parts.append(
                        f'<span style="color:{p["lora_tag"]};">'
                        f'{escape(m.group())}</span>'
                    )
                    i += len(m.group())
                    continue
                m = re.match(r'<(?:embedding|embed):[^>]*>', text[i:])
                if m:
                    parts.append(
                        f'<span style="color:{p["lora_tag"]};">'
                        f'{escape(m.group())}</span>'
                    )
                    i += len(m.group())
                    continue

            # ----------------------------------------------------------
            # 2. Comments (# ...  or  // ...)
            # ----------------------------------------------------------
            if ch == '#' or (ch == '/' and i + 1 < n and text[i + 1] == '/'):
                end = text.find('\n', i)
                if end == -1:
                    end = n
                comment_text = text[i:end]
                parts.append(
                    f'<span style="color:{p["comment"]};">'
                    f'{escape(comment_text)}</span>'
                )
                i = end
                continue

            # ----------------------------------------------------------
            # 3. Numbers (-?
            # ----------------------------------------------------------
            m = re.match(r'-?\d+(?:\.\d+)?', text[i:])
            if m:
                num = m.group()
                # Guard against lone "-" that regex can sometimes leave behind
                if num != '-' and num != '.':
                    parts.append(
                        f'<span style="color:{p["number"]};">'
                        f'{escape(num)}</span>'
                    )
                    i += len(num)
                    continue

            # ----------------------------------------------------------
            # 4. Quoted strings
            # ----------------------------------------------------------
            if ch == '"':
                end = text.find('"', i + 1)
                if end == -1:
                    end = n
                else:
                    end += 1
                s = text[i:end]
                parts.append(
                    f'<span style="color:{p["quote"]};">'
                    f'{escape(s)}</span>'
                )
                i = end
                continue
            if ch == "'":
                end = text.find("'", i + 1)
                if end == -1:
                    end = n
                else:
                    end += 1
                s = text[i:end]
                parts.append(
                    f'<span style="color:{p["quote"]};">'
                    f'{escape(s)}</span>'
                )
                i = end
                continue

            # ----------------------------------------------------------
            # 5. Parentheses with depth colouring
            # ----------------------------------------------------------
            if ch == '(':
                paren_stack.append(i)
                depth = len(paren_stack)
                key = f"paren_{min(depth, 3)}"
                parts.append(
                    f'<span style="color:{p[key]};">(</span>'
                )
                i += 1
                continue
            if ch == ')':
                depth = len(paren_stack)
                if depth > 0:
                    paren_stack.pop()
                key = f"paren_{min(depth, 3)}" if depth > 0 else "error"
                parts.append(
                    f'<span style="color:{p.get(key, p["error"])};">)</span>'
                )
                i += 1
                continue

            # ----------------------------------------------------------
            # 6. Square brackets
            # ----------------------------------------------------------
            if ch == '[':
                parts.append(
                    f'<span style="color:{p["bracket"]};">[</span>'
                )
                i += 1
                continue
            if ch == ']':
                parts.append(
                    f'<span style="color:{p["bracket"]};">]</span>'
                )
                i += 1
                continue

            # ----------------------------------------------------------
            # 7. Curly braces
            # ----------------------------------------------------------
            if ch == '{':
                parts.append(
                    f'<span style="color:{p["brace"]};">{{</span>'
                )
                i += 1
                continue
            if ch == '}':
                parts.append(
                    f'<span style="color:{p["brace"]};">}}</span>'
                )
                i += 1
                continue

            # ----------------------------------------------------------
            # 8. Commas
            # ----------------------------------------------------------
            if ch == ',':
                parts.append(
                    f'<span style="color:{p["comma"]};">,</span>'
                )
                i += 1
                continue

            # ----------------------------------------------------------
            # 9. Newlines
            # ----------------------------------------------------------
            if ch == '\n':
                parts.append('\n')
                i += 1
                continue

            # ----------------------------------------------------------
            # 10. Plain text (run of non-special chars)
            # ----------------------------------------------------------
            # Gather a contiguous run of "plain" characters to reduce
            # the number of HTML spans we emit.
            j = i
            while j < n and text[j] not in self._SPECIAL_CHARS:
                j += 1
            plain = text[i:j]
            if plain:
                parts.append(escape(plain))
                i = j
                continue

            # Fallback for any truly isolated special char
            parts.append(escape(ch))
            i += 1

        parts.append('</pre>')
        return ''.join(parts)

    # Characters that trigger a special token path above.
    _SPECIAL_CHARS = frozenset("<>()[]{}\'\",#/\n-")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process(self, text, enable_highlighting=True, theme="dark"):
        """
        Sanitise input, build highlighted HTML, and return both the
        plain text (for downstream nodes) and the HTML (for the UI).
        """
        raw_input = text
        sanitized = self._sanitize_input(text)

        if raw_input is None and not sanitized:
            html = self._build_error_text(raw_input, theme)
            return {"ui": {"text": [html]}, "result": ("",)}

        if enable_highlighting:
            html = self._highlight_text(sanitized, theme)
        else:
            html = self._build_plain_html(sanitized, theme)

        return {"ui": {"text": [html]}, "result": (sanitized,)}


# ----------------------------------------------------------------------
# ComfyUI registration
# ----------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "illumoraeTextShowMultilineColorNode": illumoraeTextShowMultilineColorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeTextShowMultilineColorNode": "Text Show Multiline Color",
}
