# region DOC
"""
illumorae Workflow Generator
----------------------------
Scans every ComfyUI_illumorae_* node package in this repository and procedurally
generates a minimal ComfyUI workflow that demonstrates the node, placed inside
the node's own folder under a ``workflows/`` subfolder.

For each node two files are written:

  <node_folder>/workflows/<class_name>.json        - graph/UI format (loadable
                                                      via the ComfyUI "Load"
                                                      button; nodes + links +
                                                      groups arrays, matching
                                                      the ComfyUI_illumorae_
                                                      nodes_example_*.json
                                                      shape)
  <node_folder>/workflows/<class_name>_api.json    - API format
                                                      ({node_id: {class_type,
                                                      inputs}}) used by the
                                                      queue/API

Each generated workflow is intentionally minimal: the demonstrated node, a few
ComfyUI-core "provider" nodes that feed its required connection inputs, a few
"sink" nodes that display its outputs (PreviewImage / PreviewAny / VAEDecode),
and one or two Note nodes explaining what the demo shows. A group bounding box
frames the demo.

Per-group strategies pick sensible context (e.g. Text/Lora groups feed the
primary string input from a PrimitiveStringMultiline so the node is shown in a
prompt pipeline; Image groups feed an EmptyImage; Save groups feed an image
into the saver). Per-node overrides supply demo widget values. Nodes may also
guide their own demo through extra Obsidian-style frontmatter fields in their
module docstring:

  WORKFLOWNOTE:: extra text appended to the usage Note node
  DEMO_<INPUTNAME>:: override the widget value for that input
                     (e.g. DEMO_CKPT_NAME:: sd_xl_base)

Refresh policy:

  default               generate a workflow only if it is MISSING
                        (existing workflows are left untouched)
  --refresh <a> <b> ... force-regenerate the listed nodes (match by folder
                        suffix, folder name, TITLE, or class name) plus their
                        API files
  --all                 force-regenerate every node's workflows
  --dry-run             report what would be written, write nothing
  --list                list discovered nodes and exit without writing

Usage:
  python tools/illumorae_workflow_generator.py
  python tools/illumorae_workflow_generator.py --refresh TextTokenCount ImageCLAHE
  python tools/illumorae_workflow_generator.py --all
  python tools/illumorae_workflow_generator.py --dry-run
  python tools/illumorae_workflow_generator.py --list
  tools\\run_workflow_generator.bat --all

Depends only on the Python standard library (ast, json, argparse, os, re,
dataclasses). Reuses the pure parsing helpers from
illumorae_docs_html_generator (parse_dataview_fields, _find_main_py_file).

VERSION::20260814
"""

# endregion

# region IMPORTS
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Reuse the stable, pure helpers from the docs generator (same tools/ folder).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from illumorae_docs_html_generator import (  # noqa: E402
    parse_dataview_fields,
    _read_text_file,
)

# endregion

# region CONST
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
NODE_FOLDER_PREFIX = "ComfyUI_illumorae_"

# Frontmatter field keys (mirrors the docs generator).
FIELD_TITLE = "TITLE"
FIELD_DESC_SHORT = "DESCRIPTIONSHORT"
FIELD_GROUP = "GROUP"
FIELD_STATUS = "STATUS"
FIELD_WORKFLOW_NOTE = "WORKFLOWNOTE"
DEMO_FIELD_PREFIX = "DEMO_"

# ComfyUI core node names used as providers / sinks. These are documented in
# tools/notes/illumorae_workflow_generator.md.
CORE_EMPTY_IMAGE = "EmptyImage"
CORE_CREATE_SHAPE_MASK = "CreateShapeMask"
CORE_EMPTY_LATENT = "EmptyLatentImage"
CORE_PRIMITIVE_STR_ML = "PrimitiveStringMultiline"
CORE_CKPT_LOADER = "CheckpointLoaderSimple"
CORE_CLIP_TEXT_ENCODE = "CLIPTextEncode"
CORE_VAE_DECODE = "VAEDecode"
CORE_PREVIEW_IMAGE = "PreviewImage"
CORE_PREVIEW_ANY = "PreviewAny"
CORE_MASK_TO_IMAGE = "MaskToImage"
CORE_NOTE = "Note"

# Scalar types that are widgets by default (not wired unless forceInput).
WIDGET_SCALAR_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}

# Colors (match the example ALL workflow).
COLOR_NODE = "#222"
COLOR_NODE_BG = "#000"
COLOR_ILLUMORAE = "#332922"
COLOR_ILLUMORAE_BG = "#593930"
COLOR_NOTE = "#432"
COLOR_NOTE_BG = "#653"
COLOR_GROUP = "#8A8"

# Layout constants (ComfyUI canvas coords; x right, y down).
DEMO_POS = [620, 360]
DEMO_SIZE = [320, 260]
PROVIDER_X = 120
SINK_X = 1080
NOTE_TITLE_POS = [620, 60]
NOTE_USAGE_POS = [620, 690]
STACK_STEP_Y = 220
NODE_DEFAULT_H = 120
# endregion


# region MODELS
@dataclass
class InputSpec:
    name: str
    section: str  # "required" | "optional" | "hidden"
    type: str  # resolved type token ("STRING","IMAGE",...); "ENUM" for combos
    is_enum: bool = False
    options: List[Any] = field(default_factory=list)
    default: Any = None
    force_input: bool = False
    multiline: bool = False

    @property
    def is_widget(self) -> bool:
        """True when this input is a widget (value in widgets_values), not a link."""
        if self.force_input:
            return False
        if self.is_enum:
            return True
        return self.type in WIDGET_SCALAR_TYPES


@dataclass
class NodeInfo:
    folder_name: str
    folder_suffix: str
    folder_path: str
    main_py_path: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    class_name: Optional[str] = None
    category: Optional[str] = None
    function_name: Optional[str] = None
    description: Optional[str] = None
    inputs: List[InputSpec] = field(default_factory=list)
    return_types: List[str] = field(default_factory=list)
    return_names: List[str] = field(default_factory=list)
    output_node: bool = False

    @property
    def title(self) -> str:
        v = self.fields.get(FIELD_TITLE)
        if isinstance(v, list):
            v = v[0] if v else None
        return str(v).strip() if v else self.folder_suffix

    @property
    def description_short(self) -> str:
        v = self.fields.get(FIELD_DESC_SHORT)
        if isinstance(v, list):
            v = v[0] if v else None
        return str(v).strip() if v else (self.description or "").strip()

    @property
    def group(self) -> str:
        v = self.fields.get(FIELD_GROUP)
        if isinstance(v, list):
            v = v[0] if v else None
        return str(v).strip() if v else "Other"

    @property
    def status(self) -> str:
        v = self.fields.get(FIELD_STATUS)
        if isinstance(v, list):
            v = v[0] if v else None
        return str(v).strip() if v else ""

    @property
    def workflow_note(self) -> str:
        v = self.fields.get(FIELD_WORKFLOW_NOTE)
        if isinstance(v, list):
            return "\n".join(str(x) for x in v)
        return str(v).strip() if v else ""

    @property
    def demo_overrides(self) -> Dict[str, Any]:
        """DEMO_<INPUT>:: frontmatter fields -> {input_name: value}."""
        out: Dict[str, Any] = {}
        plen = len(DEMO_FIELD_PREFIX)
        for key, val in self.fields.items():
            if not key.startswith(DEMO_FIELD_PREFIX):
                continue
            iname = key[plen:].lower()
            out[iname] = val
        return out

    @property
    def required_inputs(self) -> List[InputSpec]:
        return [i for i in self.inputs if i.section == "required"]

    @property
    def optional_inputs(self) -> List[InputSpec]:
        return [i for i in self.inputs if i.section == "optional"]


# endregion


# region PARSE
def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        return -v if isinstance(v, (int, float)) else v
    if isinstance(node, ast.List):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal(e) for e in node.elts)
    if isinstance(node, ast.Name):
        return node.id
    return None


def _parse_options(d: ast.Dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k_node, v_node in zip(d.keys, d.values):
        key = _const_str(k_node)
        if not key:
            continue
        out[key] = _literal(v_node)
    return out


def _literal_const(node: ast.AST) -> Any:
    """Extract a scalar literal (str/int/float/bool/None) from an AST node, or None."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        return -v if isinstance(v, (int, float)) else v
    return None


def _resolve_constant_tuple(node: ast.AST, constants: Dict[str, Tuple[Any, ...]]) -> Optional[Tuple[Any, ...]]:
    """Resolve an AST expression to a tuple of literal values, or None.

    Handles literal tuples/lists of scalar constants (strings, ints, ...),
    ``list(EXPR)`` / ``tuple(EXPR)`` calls, Name references to already-known
    module constants, and tuple concatenation (``("a",) + KNOWN``) so derived
    option lists like ``PATTERN_TYPES_UI = ("random",) + PATTERN_TYPES_CONCRETE``
    resolve. Non-scalar elements (nested calls, names not in ``constants``)
    cause the whole expression to return None so the caller can fall back.
    """
    if isinstance(node, (ast.Tuple, ast.List)):
        vals: List[Any] = []
        for e in node.elts:
            v = _literal_const(e)
            if v is None:
                return None
            vals.append(v)
        return tuple(vals)
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolve_constant_tuple(node.left, constants)
        right = _resolve_constant_tuple(node.right, constants)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("list", "tuple"):
        if len(node.args) == 1:
            return _resolve_constant_tuple(node.args[0], constants)
    return None


def _collect_module_constants(tree: ast.Module) -> Dict[str, Tuple[Any, ...]]:
    """Collect module-level assignments resolvable to tuples of literal values.

    Used to resolve combo-enum option lists that are defined as module
    constants (e.g. ``PATTERN_TYPES_UI = ("random",) + PATTERN_TYPES_CONCRETE``)
    and then referenced inside ``INPUT_TYPES`` via ``list(PATTERN_TYPES_UI)``.
    Assignments are processed in source order so later definitions can build
    on earlier ones.
    """
    constants: Dict[str, Tuple[Any, ...]] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        name = stmt.targets[0].id
        val = _resolve_constant_tuple(stmt.value, constants)
        if val is not None:
            constants[name] = val
    return constants


def _parse_input_spec(node: ast.AST, constants: Optional[Dict[str, Tuple[Any, ...]]] = None) -> InputSpec:
    """Parse a single INPUT_TYPES entry into a rich InputSpec (enum + forceInput aware).

    ``constants`` is the module-level literal-tuple constant map (from
    ``_collect_module_constants``); it lets us resolve combo enums written as
    ``list(PATTERN_TYPES_UI)`` or bare ``PATTERN_TYPES_UI`` references, not just
    inline ``["a", "b"]`` literals. Integer/string/mixed combos are all
    recognised; unresolvable combos (e.g. ``list(self.methods.keys())``) are
    still marked as enums so they become widgets rather than connection slots.
    """
    constants = constants or {}
    ptype = "ANY"
    is_enum = False
    options: List[Any] = []
    meta: Dict[str, Any] = {}

    def _try_enum(first: ast.AST) -> bool:
        """If ``first`` is a combo-enum expression, populate options and return True."""
        nonlocal is_enum, options
        # Inline literal list/tuple of scalars: ["a","b"] / (2,3,4) / ("a","b")
        if isinstance(first, (ast.List, ast.Tuple)):
            resolved = _resolve_constant_tuple(first, constants)
            if resolved is not None:
                is_enum = True
                options = list(resolved)
                return True
        # list(EXPR) / tuple(EXPR) - common for dict-keys / derived constants.
        if isinstance(first, ast.Call) and isinstance(first.func, ast.Name) and first.func.id in ("list", "tuple"):
            resolved = _resolve_constant_tuple(first, constants)
            is_enum = True
            if resolved is not None:
                options = list(resolved)
            return True
        # Bare Name referencing a known module constant tuple.
        if isinstance(first, ast.Name) and first.id in constants:
            is_enum = True
            options = list(constants[first.id])
            return True
        return False

    if isinstance(node, ast.Tuple) and node.elts:
        first = node.elts[0]
        if _try_enum(first):
            ptype = "ENUM"
        elif isinstance(first, ast.Constant) and isinstance(first.value, str):
            ptype = first.value
        elif isinstance(first, ast.Name):
            ptype = first.id
        if len(node.elts) >= 2 and isinstance(node.elts[1], ast.Dict):
            meta = _parse_options(node.elts[1])
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        ptype = node.value
    elif isinstance(node, ast.Name):
        ptype = node.id
    elif isinstance(node, (ast.List, ast.Tuple)):
        resolved = _resolve_constant_tuple(node, constants)
        is_enum = True
        ptype = "ENUM"
        if resolved is not None:
            options = list(resolved)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("list", "tuple"):
        resolved = _resolve_constant_tuple(node, constants)
        is_enum = True
        ptype = "ENUM"
        if resolved is not None:
            options = list(resolved)
    return InputSpec(
        name="",  # filled by caller
        section="",  # filled by caller
        type=ptype,
        is_enum=is_enum,
        options=options,
        default=meta.get("default"),
        force_input=bool(meta.get("forceInput", False)),
        multiline=bool(meta.get("multiline", False)),
    )


def _parse_return_tuple(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Tuple):
        return [str(_literal(e)) for e in node.elts if _literal(e) is not None]
    val = _literal(node)
    return [str(val)] if val is not None else []


def _parse_node_classes(
    src: str,
) -> Tuple[
    Dict[str, Any],
    List[Tuple[str, Optional[str], Optional[str], Optional[str], List[InputSpec], List[str], List[str], bool]],
]:
    """Parse a source file and return (module_frontmatter, [per-class tuples...]).

    Each per-class tuple is
    (class_name, category, function, description, inputs, return_types, return_names, output_node).
    All classes in a file share the file's module-docstring frontmatter.
    """
    tree = ast.parse(src)
    module_doc = ast.get_docstring(tree, clean=False) or ""
    fields = parse_dataview_fields(module_doc)
    # Collect module-level string-tuple constants so combo enums written as
    # list(PATTERN_TYPES_UI) / bare PATTERN_TYPES_UI can be resolved.
    constants = _collect_module_constants(tree)
    classes: List[
        Tuple[str, Optional[str], Optional[str], Optional[str], List[InputSpec], List[str], List[str], bool]
    ] = []

    for stmt in tree.body:
        if not isinstance(stmt, ast.ClassDef) or not stmt.name.startswith("illumorae"):
            continue
        has_input_types = any(
            isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == "INPUT_TYPES" for s in stmt.body
        )
        if not has_input_types:
            continue
        category = function_name = description = None
        return_types: List[str] = []
        return_names: List[str] = []
        output_node = False
        for body_stmt in stmt.body:
            if not isinstance(body_stmt, ast.Assign):
                continue
            for t in body_stmt.targets:
                if not isinstance(t, ast.Name):
                    continue
                if t.id == "RETURN_TYPES":
                    return_types = _parse_return_tuple(body_stmt.value)
                elif t.id == "RETURN_NAMES":
                    return_names = _parse_return_tuple(body_stmt.value)
                else:
                    val = _const_str(body_stmt.value)
                    if val is None:
                        continue
                    if t.id == "CATEGORY":
                        category = val
                    elif t.id == "FUNCTION":
                        function_name = val
                    elif t.id == "DESCRIPTION":
                        description = val
            for t in body_stmt.targets:
                if isinstance(t, ast.Name) and t.id == "OUTPUT_NODE":
                    output_node = bool(_literal(body_stmt.value))
        inputs = _parse_input_types(stmt, constants)
        classes.append(
            (stmt.name, category, function_name, description, inputs, return_types, return_names, output_node)
        )
    return fields, classes


def _parse_input_types(class_node: ast.ClassDef, constants: Optional[Dict[str, Tuple[str, ...]]] = None) -> List[InputSpec]:
    fn: Optional[ast.FunctionDef] = None
    for stmt in class_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "INPUT_TYPES":
            fn = stmt
            break
    if fn is None:
        return []
    ret_dict: Optional[ast.Dict] = None
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
            ret_dict = stmt.value
            break
    if ret_dict is None:
        return []

    section_map: Dict[str, ast.Dict] = {}
    for k_node, v_node in zip(ret_dict.keys, ret_dict.values):
        key = _const_str(k_node)
        if key and isinstance(v_node, ast.Dict):
            section_map[key] = v_node

    params: List[InputSpec] = []
    for section in ("required", "optional", "hidden"):
        sec_dict = section_map.get(section)
        if not sec_dict:
            continue
        for k_node, v_node in zip(sec_dict.keys, sec_dict.values):
            name = _const_str(k_node)
            if not name:
                continue
            spec = _parse_input_spec(v_node, constants)
            spec.name = name
            spec.section = section
            params.append(spec)
    return params


def _node_py_files(folder_path: str) -> List[str]:
    """All non-__init__ .py files in a package that may define node classes."""
    out = []
    for f in sorted(os.listdir(folder_path)):
        if not f.endswith(".py") or f == "__init__.py":
            continue
        full = os.path.join(folder_path, f)
        if os.path.isfile(full):
            out.append(full)
    return out


def scan_nodes(project_root: str) -> List[NodeInfo]:
    """Scan every node package and every illumorae node class within it.

    A package may contain several .py files and each file may define several
    node classes; one NodeInfo is produced per class so each gets its own
    demo workflow. All classes in a file share that file's module-docstring
    frontmatter (TITLE / GROUP / ...).
    """
    nodes: List[NodeInfo] = []
    for name in sorted(os.listdir(project_root)):
        folder_path = os.path.join(project_root, name)
        if name.startswith("backup") or name.startswith("."):
            continue
        if not os.path.isdir(folder_path) or not name.startswith(NODE_FOLDER_PREFIX):
            continue
        if not os.path.isfile(os.path.join(folder_path, "__init__.py")):
            continue
        prefix_len = len(NODE_FOLDER_PREFIX)
        suffix = name[prefix_len:]
        for py_path in _node_py_files(folder_path):
            try:
                src = _read_text_file(py_path)
            except OSError:
                continue
            fields, classes = _parse_node_classes(src)
            if not classes:
                continue
            for (
                cls_name,
                category,
                function_name,
                description,
                inputs,
                return_types,
                return_names,
                output_node,
            ) in classes:
                info = NodeInfo(
                    folder_name=name,
                    folder_suffix=suffix,
                    folder_path=folder_path,
                    main_py_path=py_path,
                    fields=fields,
                    class_name=cls_name,
                    category=category,
                    function_name=function_name,
                    description=description,
                    inputs=inputs,
                    return_types=return_types,
                    return_names=return_names,
                    output_node=output_node,
                )
                nodes.append(info)
    return nodes


# endregion


# region BUILDER
class WorkflowBuilder:
    """Builds a ComfyUI graph-format workflow and a parallel API-format dict."""

    def __init__(self):
        self.nodes: List[dict] = []
        self.links: List[list] = []
        self.groups: List[dict] = []
        self._next_node_id = 1
        self._next_link_id = 1
        self._next_order = 0
        self._next_group_id = 1
        # node_id -> {"class_type", "widget_inputs", "connected_inputs"}
        self._api_meta: Dict[int, dict] = {}

    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _new_link_id(self) -> int:
        lid = self._next_link_id
        self._next_link_id += 1
        return lid

    def _get(self, nid: int) -> dict:
        for n in self.nodes:
            if n["id"] == nid:
                return n
        raise KeyError(nid)

    def add_node(
        self,
        node_type: str,
        pos: List[float],
        size: List[float],
        inputs: List[dict],
        outputs: List[dict],
        widgets_values: Optional[list] = None,
        properties: Optional[dict] = None,
        color: str = COLOR_NODE,
        bgcolor: str = COLOR_NODE_BG,
        api_meta: Optional[dict] = None,
    ) -> int:
        nid = self._new_node_id()
        order = self._next_order
        self._next_order += 1
        node = {
            "id": nid,
            "type": node_type,
            "pos": pos,
            "size": size,
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": [dict(i) for i in inputs],
            "outputs": [dict(o) for o in outputs],
            "properties": properties or {"widget_ue_connectable": {}},
            "widgets_values": list(widgets_values) if widgets_values else [],
        }
        if color:
            node["color"] = color
        if bgcolor:
            node["bgcolor"] = bgcolor
        self.nodes.append(node)
        self._api_meta[nid] = api_meta or {
            "class_type": node_type,
            "widget_inputs": {},
            "connected_inputs": {},
        }
        return nid

    def connect(self, from_id: int, from_slot: int, to_id: int, to_slot: int, type_str: str) -> int:
        lid = self._new_link_id()
        self.links.append([lid, from_id, from_slot, to_id, to_slot, type_str])
        from_node = self._get(from_id)
        to_node = self._get(to_id)
        out = from_node["outputs"][from_slot]
        out.setdefault("links", [])
        if lid not in out["links"]:
            out["links"].append(lid)
        out.setdefault("slot_index", from_slot)
        to_node["inputs"][to_slot]["link"] = lid
        # API format: connected input -> [from_node_id_str, output_index]
        self._api_meta[to_id]["connected_inputs"][to_node["inputs"][to_slot]["name"]] = [str(from_id), from_slot]
        self._api_meta[to_id]["widget_inputs"].pop(to_node["inputs"][to_slot]["name"], None)
        return lid

    def add_group(self, title: str, bounding: List[float], color: str = COLOR_GROUP) -> None:
        gid = self._next_group_id
        self._next_group_id += 1
        self.groups.append(
            {
                "id": gid,
                "title": title,
                "bounding": bounding,
                "color": color,
                "font_size": 24,
                "flags": {},
            }
        )

    def bounding_box(self, pad: float = 40.0) -> List[float]:
        if not self.nodes:
            return [0, 0, 400, 300]
        xs, ys, xe, ye = [], [], [], []
        for n in self.nodes:
            x, y = n["pos"]
            w, h = n["size"]
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)
        return [min(xs) - pad, min(ys) - pad, max(xe) - min(xs) + 2 * pad, max(ye) - min(ys) + 2 * pad]

    # ---- node factories -------------------------------------------------
    def make_io(self, inputs: List[Tuple[str, str]], outputs: List[Tuple[str, str]]) -> Tuple[List[dict], List[dict]]:
        in_dicts = [{"name": n, "type": t, "link": None} for n, t in inputs]
        out_dicts = [{"name": n, "type": t, "links": [], "slot_index": i} for i, (n, t) in enumerate(outputs)]
        return in_dicts, out_dicts

    def add_note(self, text: str, pos: List[float], size: Optional[List[float]] = None) -> int:
        ins, outs = self.make_io([], [])
        nid = self.add_node(
            CORE_NOTE,
            pos,
            size or [320, 150],
            ins,
            outs,
            widgets_values=[text],
            properties={"widget_ue_connectable": {}},
            color=COLOR_NOTE,
            bgcolor=COLOR_NOTE_BG,
            api_meta={"class_type": CORE_NOTE, "widget_inputs": {"text": text}, "connected_inputs": {}},
        )
        return nid

    def add_primitive_string(self, text: str, pos: List[float]) -> int:
        ins, outs = self.make_io([], [("STRING", "STRING")])
        return self.add_node(
            CORE_PRIMITIVE_STR_ML,
            pos,
            [300, 130],
            ins,
            outs,
            widgets_values=[text, [False, True]],
            properties={
                "cnr_id": "comfy-core",
                "Node name for S&R": CORE_PRIMITIVE_STR_ML,
                "widget_ue_connectable": {},
            },
            api_meta={"class_type": CORE_PRIMITIVE_STR_ML, "widget_inputs": {"string": text}, "connected_inputs": {}},
        )

    def add_empty_image(self, pos: List[float], width: int = 512, height: int = 512) -> int:
        ins, outs = self.make_io([], [("IMAGE", "IMAGE")])
        return self.add_node(
            CORE_EMPTY_IMAGE,
            pos,
            [315, 130],
            ins,
            outs,
            widgets_values=[width, height, 0, 1],
            properties={"cnr_id": "comfy-core", "Node name for S&R": CORE_EMPTY_IMAGE, "widget_ue_connectable": {}},
            api_meta={
                "class_type": CORE_EMPTY_IMAGE,
                "widget_inputs": {"width": width, "height": height, "color": 0, "batch_size": 1},
                "connected_inputs": {},
            },
        )

    def add_empty_latent(self, pos: List[float], width: int = 512, height: int = 512) -> int:
        ins, outs = self.make_io([], [("LATENT", "LATENT")])
        return self.add_node(
            CORE_EMPTY_LATENT,
            pos,
            [315, 106],
            ins,
            outs,
            widgets_values=[width, height, 1],
            properties={"cnr_id": "comfy-core", "Node name for S&R": CORE_EMPTY_LATENT, "widget_ue_connectable": {}},
            api_meta={
                "class_type": CORE_EMPTY_LATENT,
                "widget_inputs": {"width": width, "height": height, "batch_size": 1},
                "connected_inputs": {},
            },
        )

    def add_shape_mask(self, pos: List[float]) -> int:
        ins, outs = self.make_io([], [("IMAGE", "IMAGE"), ("MASK", "MASK")])
        return self.add_node(
            CORE_CREATE_SHAPE_MASK,
            pos,
            [315, 170],
            ins,
            outs,
            widgets_values=[512, 0, "circle", 512, 512],
            properties={
                "cnr_id": "comfy-core",
                "Node name for S&R": CORE_CREATE_SHAPE_MASK,
                "widget_ue_connectable": {},
            },
            api_meta={
                "class_type": CORE_CREATE_SHAPE_MASK,
                "widget_inputs": {
                    "side_length": 512,
                    "side_mask": 0,
                    "shape": "circle",
                    "image_width": 512,
                    "image_height": 512,
                },
                "connected_inputs": {},
            },
        )

    def add_checkpoint_loader(self, pos: List[float], ckpt: str = "sd_xl_base_1.0.safetensors") -> int:
        ins, outs = self.make_io([], [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")])
        return self.add_node(
            CORE_CKPT_LOADER,
            pos,
            [320, 100],
            ins,
            outs,
            widgets_values=[ckpt],
            properties={"cnr_id": "comfy-core", "Node name for S&R": CORE_CKPT_LOADER, "widget_ue_connectable": {}},
            api_meta={"class_type": CORE_CKPT_LOADER, "widget_inputs": {"ckpt_name": ckpt}, "connected_inputs": {}},
        )

    def add_clip_text_encode(self, pos: List[float], clip_id: int, text: str) -> int:
        ins, outs = self.make_io([("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")])
        nid = self.add_node(
            CORE_CLIP_TEXT_ENCODE,
            pos,
            [320, 120],
            ins,
            outs,
            widgets_values=[text, [False, True]],
            properties={
                "cnr_id": "comfy-core",
                "Node name for S&R": CORE_CLIP_TEXT_ENCODE,
                "widget_ue_connectable": {},
            },
            api_meta={"class_type": CORE_CLIP_TEXT_ENCODE, "widget_inputs": {"text": text}, "connected_inputs": {}},
        )
        self.connect(clip_id, 1, nid, 0, "CLIP")
        return nid

    def add_preview_image(self, pos: List[float]) -> int:
        ins, outs = self.make_io([("images", "IMAGE")], [])
        return self.add_node(
            CORE_PREVIEW_IMAGE,
            pos,
            [210, 250],
            ins,
            outs,
            widgets_values=[],
            properties={"cnr_id": "comfy-core", "Node name for S&R": CORE_PREVIEW_IMAGE, "widget_ue_connectable": {}},
            api_meta={"class_type": CORE_PREVIEW_IMAGE, "widget_inputs": {}, "connected_inputs": {}},
        )

    def add_preview_any(self, pos: List[float]) -> int:
        ins, outs = self.make_io([("source", "*")], [])
        return self.add_node(
            CORE_PREVIEW_ANY,
            pos,
            [240, 120],
            ins,
            outs,
            widgets_values=[],
            properties={"cnr_id": "comfy-core", "Node name for S&R": CORE_PREVIEW_ANY, "widget_ue_connectable": {}},
            api_meta={"class_type": CORE_PREVIEW_ANY, "widget_inputs": {}, "connected_inputs": {}},
        )

    def add_vae_decode(self, pos: List[float], vae_id: int) -> int:
        ins, outs = self.make_io([("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")])
        nid = self.add_node(
            CORE_VAE_DECODE,
            pos,
            [220, 90],
            ins,
            outs,
            widgets_values=[],
            properties={"cnr_id": "comfy-core", "Node name for S&R": CORE_VAE_DECODE, "widget_ue_connectable": {}},
            api_meta={"class_type": CORE_VAE_DECODE, "widget_inputs": {}, "connected_inputs": {}},
        )
        self.connect(vae_id, 2, nid, 1, "VAE")
        return nid

    def add_mask_to_image(self, pos: List[float]) -> int:
        ins, outs = self.make_io([("mask", "MASK")], [("IMAGE", "IMAGE")])
        return self.add_node(
            CORE_MASK_TO_IMAGE,
            pos,
            [220, 90],
            ins,
            outs,
            widgets_values=[],
            properties={"cnr_id": "comfy-core", "Node name for S&R": CORE_MASK_TO_IMAGE, "widget_ue_connectable": {}},
            api_meta={"class_type": CORE_MASK_TO_IMAGE, "widget_inputs": {}, "connected_inputs": {}},
        )

    # ---- output ---------------------------------------------------------
    def to_graph(self) -> dict:
        return {
            "id": "illumorae-workflow",
            "revision": 0,
            "last_node_id": self._next_node_id - 1,
            "last_link_id": self._next_link_id - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {"ds": {"scale": 0.8, "offset": [0, 0]}, "ue_links": [], "links_added_by_ue": []},
            "version": 0.4,
        }

    def to_api(self) -> dict:
        # Note / Reroute are UI-only graph elements with no registered
        # class_type, so they cannot appear in an executable API-format prompt.
        _SKIP_API = {CORE_NOTE, "Reroute"}
        api: Dict[str, dict] = {}
        for nid, meta in self._api_meta.items():
            if meta["class_type"] in _SKIP_API:
                continue
            entry = {"class_type": meta["class_type"], "inputs": {}}
            # connected inputs override widget inputs of the same name
            entry["inputs"].update(meta["widget_inputs"])
            entry["inputs"].update(meta["connected_inputs"])
            api[str(nid)] = entry
        return api


# endregion


# region PROVIDERS
class ProviderFactory:
    """Lazily creates and caches ComfyUI-core provider nodes for a given type."""

    def __init__(self, builder: WorkflowBuilder):
        self.b = builder
        self._cache: Dict[str, int] = {}  # type -> node_id
        self._ckpt_id: Optional[int] = None
        self._provider_count = 0

    def _next_provider_pos(self) -> List[float]:
        y = 360 + self._provider_count * STACK_STEP_Y
        self._provider_count += 1
        return [PROVIDER_X, y]

    def _next_sink_pos(self) -> List[float]:
        # tracked separately via a counter on the builder side
        return [SINK_X, 360]

    def get(self, type_str: str) -> Optional[Tuple[int, int]]:
        """Return (node_id, output_slot) for a provider of type_str, or None."""
        type_str = type_str.upper()
        if type_str in self._cache:
            return self._cache[type_str], 0
        pos = self._next_provider_pos()
        if type_str == "IMAGE":
            nid = self.b.add_empty_image(pos)
            self._cache["IMAGE"] = nid
            return nid, 0
        if type_str == "LATENT":
            nid = self.b.add_empty_latent(pos)
            self._cache["LATENT"] = nid
            return nid, 0
        if type_str == "STRING":
            nid = self.b.add_primitive_string("sample prompt text", pos)
            self._cache["STRING"] = nid
            return nid, 0
        if type_str == "MASK":
            nid = self.b.add_shape_mask(pos)
            self._cache["MASK"] = nid
            return nid, 1
        if type_str in ("MODEL", "CLIP", "VAE"):
            ckpt = self._ensure_checkpoint()
            slot = {"MODEL": 0, "CLIP": 1, "VAE": 2}[type_str]
            return ckpt, slot
        if type_str == "CONDITIONING":
            ckpt = self._ensure_checkpoint()
            nid = self.b.add_clip_text_encode(pos, ckpt, "positive prompt text")
            self._cache["CONDITIONING"] = nid
            return nid, 0
        return None

    def get_string_with_text(self, text: str) -> Tuple[int, int]:
        """A dedicated STRING provider carrying the given demo text (not cached)."""
        pos = self._next_provider_pos()
        nid = self.b.add_primitive_string(text, pos)
        return nid, 0

    def _ensure_checkpoint(self) -> int:
        if self._ckpt_id is None:
            self._ckpt_id = self.b.add_checkpoint_loader(self._next_provider_pos())
        return self._ckpt_id


# endregion


# region SINKS
def add_sink_for(
    builder: WorkflowBuilder, type_str: str, sink_index: int, ckpt_id: Optional[int]
) -> Optional[Tuple[int, int, str]]:
    """Create a sink that displays the given output type.

    Returns (sink_node_id, sink_input_slot, sink_input_type) so the caller can
    connect the demo node's output to it, or None when no sink applies.
    """
    t = type_str.upper()
    pos = [SINK_X, 360 + sink_index * STACK_STEP_Y]
    if t == "IMAGE":
        nid = builder.add_preview_image(pos)
        return nid, 0, "IMAGE"
    if t == "MASK":
        m2i = builder.add_mask_to_image(pos)
        prev = builder.add_preview_image([pos[0] + 240, pos[1]])
        builder.connect(m2i, 0, prev, 0, "IMAGE")
        return m2i, 0, "MASK"
    if t == "LATENT":
        if ckpt_id is None:
            return None
        vae = builder.add_vae_decode(pos, ckpt_id)
        prev = builder.add_preview_image([pos[0] + 260, pos[1]])
        builder.connect(vae, 0, prev, 0, "IMAGE")
        return vae, 0, "LATENT"
    # STRING, MODEL, CLIP, VAE, CONDITIONING, * -> PreviewAny
    nid = builder.add_preview_any(pos)
    return nid, 0, "*"


# endregion

# region STRATEGY
# Group-specific configuration. ``promote_text`` converts the primary STRING
# required input into a connected PrimitiveStringMultiline so the node is shown
# inside a prompt pipeline rather than with a bare widget.
GROUP_CONFIG: Dict[str, dict] = {
    "Text": {
        "promote_text": True,
        "sample_prompt": "( berry : 1.2) , carrot , ( radish :1.3)\nplants , <lora:berry:0.6>\n<lora:carrot:0.3>",
        "note": "Text-group node shown in a prompt pipeline. The primary text input is fed from a PrimitiveStringMultiline so the node can be chained before CLIPTextEncode / KSampler.",
    },
    "Lora": {
        "promote_text": True,
        "sample_prompt": "a portrait of a wizard , <lora:style:0.8> , <lora:detail:0.4>",
        "note": "LoRA-group node shown parsing <lora:name:strength> tags. Feed it prompt text containing LoRA tags; the modified text / model can then drive a sampler.",
    },
    "Image": {
        "promote_text": False,
        "note": "Image-group node fed by an EmptyImage provider. Output is previewed with PreviewImage (or MaskToImage -> PreviewImage for MASK outputs).",
    },
    "Load": {
        "promote_text": False,
        "note": "Load-group node. Folder / path inputs are left as widgets with their declared defaults; point them at a real folder on your machine before running. IMAGE outputs preview directly; STRING outputs (file name / path) show in PreviewAny.",
    },
    "Save": {
        "promote_text": False,
        "note": "Save-group node. An EmptyImage feeds the IMAGE input so the saver can run standalone; set folderpath_input to a writable folder before queueing.",
    },
    "Checkpoint": {
        "promote_text": False,
        "note": "Checkpoint-group node. ckpt_name is a partial-match string widget; MODEL / CLIP / VAE outputs are shown via PreviewAny and would normally feed a KSampler.",
    },
    "Select": {
        "promote_text": False,
        "note": "Select-group node. Path / project inputs are widgets; the selected item string is shown via PreviewAny.",
    },
    "VLM": {
        "promote_text": False,
        "note": "VLM-group node. An EmptyImage feeds the image input; model loading inputs are widgets. Outputs are shown via PreviewAny / PreviewImage.",
    },
    "SAM3D": {
        "promote_text": False,
        "note": "SAM3D-group node. Mesh / scene inputs are specialized and left unconnected here; connect a SAM3D pipeline upstream before running. This workflow shows the node's widgets and output slots.",
    },
    "Other": {
        "promote_text": False,
        "note": "Node shown with generic providers for its connection inputs and PreviewAny sinks for its outputs.",
    },
}

# Per-node widget overrides keyed by class name. Values map input name -> value.
NODE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "illumoraeCheckpointLoaderByStringDirtyNode": {"ckpt_name": "sd_xl_base"},
    "illumoraeCheckpointRandomSelectorNode": {"ckpt_search_string": "sd_xl"},
    "illumoraeLoadImageRandomVariantNode": {"folder": "C:/input", "base_filename": "image"},
    "illumoraeLoadImageWFilePathOutNode": {"image_filepath": "C:/input/image.png"},
    "illumoraeLoadRandomFileFromPathByPrefixNode": {"folder": "C:/input", "prefix": "image"},
    "illumoraeLoadTextFileGracefulNode": {"path": "C:/input/note.txt"},
    "illumoraeSaveImageExtendedFolderPathNode": {"folderpath_input": "C:/output", "filename_prefix": "demo"},
    "illumoraeSaveAnimatedWEBPFolderPathNode": {"folderpath_input": "C:/output", "filename_prefix": "demo"},
    "illumoraeSelectITEMByAmountGeneratedNode": {"root_folder": "C:/items"},
    "illumoraeTextPromptFromImagesFolderNode": {"folder": "C:/input"},
}


def _widget_default(spec: InputSpec) -> Any:
    if spec.is_enum:
        if spec.default is not None:
            return spec.default
        return spec.options[0] if spec.options else ""
    if spec.default is not None:
        return spec.default
    if spec.type == "INT":
        return 0
    if spec.type == "FLOAT":
        return 0.0
    if spec.type == "BOOLEAN":
        return False
    return ""


def _resolve_widget_value(spec: InputSpec, info: NodeInfo) -> Any:
    overrides = dict(NODE_OVERRIDES.get(info.class_name, {}))
    overrides.update(info.demo_overrides)
    if spec.name in overrides:
        return overrides[spec.name]
    return _widget_default(spec)


def _primary_text_input(info: NodeInfo) -> Optional[InputSpec]:
    """Pick the STRING required input to promote for Text/Lora groups."""
    candidates = [i for i in info.required_inputs if i.type == "STRING" and not i.force_input]
    if not candidates:
        return None
    # Prefer one literally named "text" / "prompt".
    for pref in ("text", "prompt", "lora_text", "input_text"):
        for c in candidates:
            if c.name.lower() == pref:
                return c
    return candidates[0]


def build_workflow(info: NodeInfo) -> Tuple[dict, dict]:
    """Build (graph_format, api_format) workflows demonstrating ``info``."""
    b = WorkflowBuilder()
    factory = ProviderFactory(b)

    group_cfg = GROUP_CONFIG.get(info.group, GROUP_CONFIG["Other"])
    promote_text = bool(group_cfg.get("promote_text", False))
    sample_prompt = str(group_cfg.get("sample_prompt", "sample prompt text"))

    # Decide which inputs become graph connection slots. ComfyUI's graph format
    # only lists connection inputs (and STRING widgets that are actually wired)
    # in a node's ``inputs`` array; pure widgets (INT/FLOAT/BOOLEAN/combo) live
    # only in widgets_values. Build the slot list in declaration order and keep
    # a decl_idx -> graph_slot map so connections target the right slot.
    primary_text = _primary_text_input(info) if promote_text else None
    promoted_text_name: Optional[str] = primary_text.name if primary_text is not None else None

    graph_slots: List[Tuple[int, str, str, bool]] = []  # (decl_idx, name, type, is_string_widget)
    decl_to_slot: Dict[int, int] = {}
    for decl_idx, spec in enumerate(info.inputs):
        if spec.section == "hidden":
            continue
        if not spec.is_widget:
            slot = len(graph_slots)
            graph_slots.append((decl_idx, spec.name, spec.type, False))
            decl_to_slot[decl_idx] = slot
        elif spec.type == "STRING" and spec.name == promoted_text_name:
            slot = len(graph_slots)
            graph_slots.append((decl_idx, spec.name, "STRING", True))
            decl_to_slot[decl_idx] = slot

    in_dicts = [
        {"name": n, "type": t, "link": None, **({"widget": {"name": n}} if is_sw else {})}
        for (_, n, t, is_sw) in graph_slots
    ]
    demo_outputs = [
        (
            (
                info.return_names[k]
                if k < len(info.return_names) and info.return_names[k]
                else info.return_types[k] if k < len(info.return_types) else f"OUT{k}"
            ),
            info.return_types[k] if k < len(info.return_types) else "ANY",
        )
        for k in range(len(info.return_types))
    ]
    _, out_dicts = b.make_io([], demo_outputs)
    demo_id = b.add_node(
        info.class_name,
        list(DEMO_POS),
        list(DEMO_SIZE),
        in_dicts,
        out_dicts,
        widgets_values=[],
        properties={
            "cnr_id": "comfyui_illumorae",
            "Node name for S&R": info.class_name,
            "aux_id": "CorvaeOboro/ComfyUI_illumorae",
            "widget_ue_connectable": {},
        },
        color=COLOR_ILLUMORAE,
        bgcolor=COLOR_ILLUMORAE_BG,
        api_meta={"class_type": info.class_name, "widget_inputs": {}, "connected_inputs": {}},
    )

    # Track a shared checkpoint id (created on demand by the factory) for sinks.
    ckpt_id_holder: Dict[str, Optional[int]] = {"id": None}

    def ensure_ckpt_for_sink() -> Optional[int]:
        if ckpt_id_holder["id"] is None:
            # Trigger creation through the factory without linking anywhere.
            got = factory.get("MODEL")
            if got is not None:
                ckpt_id_holder["id"] = got[0]
        return ckpt_id_holder["id"]

    # Wire inputs. Widget inputs (except the promoted STRING) record their value
    # for the API format and for widgets_values; connection inputs get a provider.
    for decl_idx, spec in enumerate(info.inputs):
        if spec.section == "hidden":
            continue
        if spec.is_widget and spec.name != promoted_text_name:
            b._api_meta[demo_id]["widget_inputs"][spec.name] = _resolve_widget_value(spec, info)
            continue
        if not spec.is_widget:
            got = factory.get(spec.type)
            if got is None:
                continue  # no provider; leave the slot unconnected
            b.connect(got[0], got[1], demo_id, decl_to_slot[decl_idx], spec.type)
            if spec.type in ("MODEL", "CLIP", "VAE"):
                ckpt_id_holder["id"] = got[0]
            continue
        # promoted STRING widget -> connect a PrimitiveStringMultiline provider
        got = factory.get_string_with_text(sample_prompt)
        b.connect(got[0], got[1], demo_id, decl_to_slot[decl_idx], "STRING")

    # Build widgets_values for the demo node in declaration order (required then
    # optional, widget-only), skipping the promoted/connected STRING input.
    # ComfyUI auto-inserts a `control_after_generate` combo widget (a virtual
    # subwidget) immediately after any widget named "seed" or "noise_seed".
    # Although excluded from the prompt (serialize:false), its value IS part of
    # the saved graph's widgets_values, so it must be emitted here in position
    # or every later widget value shifts by one and combos fail to validate.
    widget_values: List[Any] = []
    for spec in info.inputs:
        if spec.section == "hidden":
            continue
        if not spec.is_widget:
            continue
        if spec.name == promoted_text_name:
            continue
        widget_values.append(_resolve_widget_value(spec, info))
        if spec.name in ("seed", "noise_seed"):
            widget_values.append("fixed")
    b._get(demo_id)["widgets_values"] = widget_values

    # Wire outputs to sinks.
    for out_slot, out_type in enumerate(info.return_types):
        if out_slot >= len(b._get(demo_id)["outputs"]):
            break
        ckpt = ensure_ckpt_for_sink() if out_type.upper() == "LATENT" else None
        sink = add_sink_for(b, out_type, out_slot, ckpt)
        if sink is None:
            continue
        sink_id, sink_slot, sink_type = sink
        b.connect(demo_id, out_slot, sink_id, sink_slot, sink_type)

    # Note nodes.
    title_note = f"{info.title}\n[{info.class_name}]\nGroup: {info.group}" + (
        f"  Status: {info.status}" if info.status else ""
    )
    b.add_note(title_note, list(NOTE_TITLE_POS), [360, 110])

    usage_lines = []
    if info.description_short:
        usage_lines.append(info.description_short)
    usage_lines.append("")
    req = ", ".join(f"{i.name}:{i.type}" for i in info.required_inputs) or "(none)"
    outs = (
        ", ".join(
            f"{info.return_names[k]}:{info.return_types[k]}" if k < len(info.return_names) else info.return_types[k]
            for k in range(len(info.return_types))
        )
        or "(none - output/saver node)"
    )
    usage_lines.append(f"Inputs (required): {req}")
    usage_lines.append(f"Outputs: {outs}")
    usage_lines.append("")
    usage_lines.append(group_cfg.get("note", ""))
    if info.workflow_note:
        usage_lines.append("")
        usage_lines.append(info.workflow_note)
    b.add_note("\n".join(usage_lines).strip(), list(NOTE_USAGE_POS), [420, 200])

    # Group bounding box around everything.
    b.add_group(f"{info.group}: {info.title}", b.bounding_box())

    return b.to_graph(), b.to_api()


# endregion


# region WRITE
def _workflow_paths(info: NodeInfo) -> Tuple[str, str]:
    wf_dir = os.path.join(info.folder_path, "workflows")
    base = f"{info.class_name}.json"
    api = f"{info.class_name}_api.json"
    return os.path.join(wf_dir, base), os.path.join(wf_dir, api)


def _write_json(path: str, data: Any, dry: bool) -> None:
    if dry:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _matches(info: NodeInfo, token: str) -> bool:
    tok = token.lower()
    return (
        info.folder_name.lower() == tok
        or info.folder_suffix.lower() == tok
        or info.title.lower() == tok
        or (info.class_name or "").lower() == tok
    )


def generate(project_root: str, refresh: List[str], all_: bool, dry: bool) -> List[Tuple[str, str, str]]:
    """Generate workflows. Returns list of (folder_suffix, action, path) tuples."""
    nodes = scan_nodes(project_root)
    results: List[Tuple[str, str, str]] = []
    for info in nodes:
        graph_path, api_path = _workflow_paths(info)
        graph_exists = os.path.isfile(graph_path)

        if all_:
            action = "regenerate"
        elif refresh and any(_matches(info, t) for t in refresh):
            action = "refresh"
        elif not graph_exists:
            action = "generate"
        else:
            action = "skip"

        if action == "skip":
            results.append((info.folder_suffix, "skip", graph_path))
            continue

        graph, api = build_workflow(info)
        _write_json(graph_path, graph, dry)
        _write_json(api_path, api, dry)
        results.append((info.folder_suffix, action, graph_path))
    return results


# endregion


# region CLI
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Procedurally generate minimal ComfyUI demo workflows for each illumorae node.",
    )
    p.add_argument(
        "project_root",
        nargs="?",
        default=REPO_ROOT,
        help="Repository root containing ComfyUI_illumorae_* folders (default: this repo).",
    )
    p.add_argument(
        "--refresh",
        nargs="+",
        metavar="NAME",
        default=[],
        help="Force-regenerate the listed nodes (folder suffix / name / TITLE / class name).",
    )
    p.add_argument("--all", action="store_true", dest="all_", help="Force-regenerate every node's workflows.")
    p.add_argument("--dry-run", action="store_true", help="Report what would be written without writing files.")
    p.add_argument("--list", action="store_true", help="List discovered nodes and exit without writing.")
    args = p.parse_args(argv)

    nodes = scan_nodes(args.project_root)

    if args.list:
        print(f"Discovered {len(nodes)} illumorae nodes in {args.project_root}:\n")
        for info in nodes:
            gp, ap = _workflow_paths(info)
            exists = "ok" if os.path.isfile(gp) else "missing"
            print(f"  [{info.group:>8}] {info.folder_suffix:<45} {info.class_name}  ({exists})")
        return 0

    results = generate(args.project_root, args.refresh, args.all_, args.dry_run)
    written = 0
    skipped = 0
    for suffix, action, path in results:
        if action == "skip":
            skipped += 1
            continue
        written += 1
        rel = os.path.relpath(path, args.project_root)
        print(f"  [{action:>10}] {suffix:<45} -> {rel}")
    print(f"\n{written} workflow(s) {'would be ' if args.dry_run else ''}written, {skipped} skipped.")
    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# endregion
