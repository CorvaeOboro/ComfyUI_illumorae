"""
illumoraeDisplayLabelText - a frontend-only ComfyUI virtual node
----------------------------------------------------------------
A purely visual node that drops a floating text label anywhere on the graph.
The node has no Python backend: it is registered entirely in the browser by
``web/display_label_text.js`` and never runs on the ComfyUI server. This module exists so
the illumorae docs generator can pick up the node's frontmatter and document
its properties; it defines no node class and no ``NODE_CLASS_MAPPINGS``.

The displayed text is the node Title. The following properties are adjustable
from the node's properties panel (double-click the node to open it):

    fontSize        (number)  Font size in pixels. Default 12.
    fontFamily      (string)  CSS font family. Default "Arial".
    fontColor       (string)  Hex color string, e.g. "#ffffff". Default "#ffffff".
    textAlign       (combo)   "left" | "center" | "right". Default "left".
    backgroundColor (string)  Hex color string, or "transparent" / "" for none.
                              Default "transparent".
    padding         (number)  Pixels of padding around the text. Default 0.
    borderRadius    (number)  Corner radius of the background. Default 0.
    angle           (number)  Rotation in degrees around the label center. Default 0.

The Title supports the literal sequence "\\n" to insert a newline when drawing.
Multiline text can also be typed in the properties panel (ComfyUI allows
shift+enter there). Use ComfyUI's native right-click "pin" option to make the
label stick to the workflow and let clicks pass through; right-click again to
unpin. Color values accept a 7th/8th hex digit (or 5th in shorthand) for
translucency, e.g. "#FFFFFF88" is semi-transparent white.

Inspired by the rgthree-comfy "Label" frontend-only node, reimplemented
standalone with no dependency on the rgthree nodepack.

TITLE::Display Label Text
DESCRIPTIONSHORT::Frontend-only floating text label for the graph. No backend execution; styled via the properties panel.
VERSION::20260906
GROUP::Display
GROUPORDER::13
LISTORDER::1
STATUS::working
"""
#region NOTES - frontend-only package, no Python node class registered
# This module intentionally defines no ``illumorae*`` node class and no
# ``NODE_CLASS_MAPPINGS``. The node is a virtual LGraphNode registered by
# ``web/label.js`` through ``LiteGraph.registerNodeType``. ComfyUI loads that
# script via ``WEB_DIRECTORY`` declared in ``__init__.py``.
#
# The illumorae docs generator scans this docstring for the frontmatter fields
# above (TITLE, DESCRIPTIONSHORT, VERSION, GROUP, etc.) and renders a node
# page even though there are no Python inputs/outputs to parse.
#endregion
