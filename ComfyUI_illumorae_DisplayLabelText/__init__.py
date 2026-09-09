"""
illumoraeDisplayLabelText - a frontend-only ComfyUI custom node package.

This package ships a single virtual node, "Display Label Text (illumorae)", that
exists entirely in the browser. There is no Python node class and nothing is
executed on the ComfyUI backend. The node is registered by the JavaScript
extension under ``web/`` via ``LiteGraph.registerNodeType``.

ComfyUI loads the extension because ``WEB_DIRECTORY`` is set below; the empty
``NODE_CLASS_MAPPINGS`` signals that no server-side node needs to be wired up.
"""

# Frontend-only: no Python node classes are registered. The node is brought to
# life by web/display_label_text.js, which ComfyUI auto-loads from WEB_DIRECTORY.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
