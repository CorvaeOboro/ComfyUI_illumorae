#region DOC
"""
illumorae Docs HTML Generator
-----------------------------
Scans ComfyUI_illumorae_* node packages in this repository and generates a
static HTML documentation site under docs/ , styled after the reference pages
in tools/ref/html (iframe sidebar + per-page content, dark theme).

For each node package the generator:
  - finds the main .py file (excluding __init__.py)
  - parses the module docstring for Obsidian-style DataView fields (KEY:: value)
    e.g. TITLE:: , DESCRIPTIONSHORT:: , VERSION:: , IMAGE:: , GROUP:: ,
    GROUPORDER:: , LISTORDER:: , STATUS::
  - AST-parses INPUT_TYPES (required/optional) into a parameter table
  - AST-parses RETURN_TYPES / RETURN_NAMES into an outputs table
  - reads CATEGORY , FUNCTION , DESCRIPTION class attributes
  - looks up the IMAGE:: screenshot under docs/ and embeds it when found

Output layout (docs/nodes/ subfolder, images referenced as ../<img>.png):
  docs/index.html            overview page (sidebar + welcome content)
  docs/illumorae_menu.htm    sidebar menu, grouped by GROUP::
  docs/nodes/<NodeName>.html one page per node, iframe sidebar -> ../illumorae_menu.htm

Refresh policy:
  default               generate only MISSING node pages (skip existing),
                         always regenerate menu + index so the sidebar stays current
  --refresh <a> <b> ... force-regenerate the listed node pages (match by folder
                         suffix, folder name, or TITLE) and refresh menu + index
  --all                 force-regenerate every node page and refresh menu + index
  --menu-only           only regenerate the sidebar menu + index, leave node pages
  --dry-run             report what would be written, write nothing

Usage:
  python tools/illumorae_docs_html_generator.py
  python tools/illumorae_docs_html_generator.py --refresh ImageCLAHE TextTokenCount
  python tools/illumorae_docs_html_generator.py --all
  python tools/illumorae_docs_html_generator.py --menu-only
  python tools/illumorae_docs_html_generator.py --dry-run

VERSION::20260814
"""
#endregion

#region IMPORTS
from __future__ import annotations

import argparse
import ast
import html
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
#endregion


#region CONST
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
NODES_DIR = os.path.join(DOCS_DIR, "nodes")
MENU_FILE = os.path.join(DOCS_DIR, "illumorae_menu.htm")
INDEX_FILE = os.path.join(DOCS_DIR, "index.html")

NODE_FOLDER_PREFIX = "ComfyUI_illumorae_"

# Obsidian-style DataView field keys we care about (uppercased).
FIELD_TITLE = "TITLE"
FIELD_DESC_SHORT = "DESCRIPTIONSHORT"
FIELD_VERSION = "VERSION"
FIELD_IMAGE = "IMAGE"
FIELD_GROUP = "GROUP"
FIELD_GROUP_ORDER = "GROUPORDER"
FIELD_LIST_ORDER = "LISTORDER"
FIELD_STATUS = "STATUS"

# CSS pulled from tools/ref/html/Mods_ItemMergeBestAffix.html (dark theme).
PAGE_CSS = """:root { --bg:#0d0d0d; --surface:#141414; --border:#2a2a2a; --text:#e0e0e0; --muted:#999; --accent:#4a7fb5; --link:#64b5f6; --code:#1e1e1e; }
* { box-sizing:border-box; }
html,body { margin:0; height:100%; font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }
.page { display:flex; height:100vh; }
.sidebar { width:14%; min-width:200px; border-right:1px solid var(--border); background:var(--bg); }
.sidebar iframe { width:100%; height:100%; border:none; }
.content { flex:1; overflow-y:auto; }
header { background:linear-gradient(90deg,#0a1520 0%,#0d0d0d 100%); border-bottom:2px solid var(--accent); padding:1.2rem 1rem; text-align:center; }
header h1 { margin:0; font-size:2rem; color:#fff; }
header p { margin:0.5rem 0 0; color:var(--muted); }
.container { max-width:1000px; margin:0 auto; padding:2rem 1.5rem; }
h2 { color:#fff; border-bottom:1px solid var(--border); padding-bottom:0.5rem; margin-top:2.5rem; font-size:1.4rem; }
h3 { color:#fff; margin-top:1.5rem; font-size:1.1rem; }
a { color:var(--link); text-decoration:none; }
a:hover { text-decoration:underline; }
ul,ol { padding-left:1.4rem; }
li { margin-bottom:0.4rem; }
table { width:100%; border-collapse:collapse; margin:1rem 0; font-size:0.95rem; }
th,td { border:1px solid var(--border); padding:0.5rem 0.7rem; text-align:left; vertical-align:top; }
th { background:var(--surface); color:#fff; }
td { background:#111; }
code,pre { background:var(--code); border:1px solid var(--border); border-radius:4px; font-family:Consolas,Monaco,'Courier New',monospace; font-size:0.9rem; }
code { padding:0.15rem 0.35rem; }
pre { padding:0.8rem; overflow-x:auto; }
.note { border-left:3px solid var(--accent); background:var(--surface); padding:0.8rem 1rem; margin:1rem 0; border-radius:0 6px 6px 0; }
.meta-grid { display:grid; grid-template-columns:160px 1fr; gap:0.4rem 1rem; margin:1rem 0; font-size:0.95rem; }
.meta-grid dt { color:var(--muted); }
.meta-grid dd { margin:0; }
.badge { display:inline-block; padding:0.1rem 0.5rem; border-radius:10px; font-size:0.75rem; border:1px solid var(--border); background:var(--surface); color:var(--muted); }
.badge.status-working { color:#b5d65a; border-color:#5a8a3a; }
.badge.status-broken { color:#ff6b6b; border-color:#a75a5a; }
.badge.status-experimental { color:#d6b55a; border-color:#a79a5a; }
img.node-shot { width:100%; border:1px solid var(--border); border-radius:4px; margin:1rem 0; }
.img-missing { border:1px dashed var(--border); background:var(--surface); color:var(--muted); padding:2rem; text-align:center; border-radius:4px; margin:1rem 0; }
footer { text-align:center; padding:1.2rem 1rem; color:var(--muted); font-size:0.9rem; border-top:1px solid var(--border); margin-top:3rem; }"""

MENU_CSS = """body { background-color:#0d0d0d; font-family:Verdana,Arial,Helvetica,sans-serif; color:#999; margin:0; padding:0; }
a:link, a:visited { color:#4a7fb5; text-decoration:none; display:block; padding:2px 0; font-weight:normal; }
a:hover { color:#7ab5e8; }
a:active { color:#4a7fb5; }
.menu-section { margin-top:0.6rem; }
.menu-section h3 { color:#ffffff; font-size:0.85rem; text-transform:uppercase; margin:0 0 0.2rem 0; padding-bottom:0.15rem; border-bottom:1px solid #2a2a2a; }
.menu-section a { font-size:0.85rem; padding:2px 0; }
.external { margin-top:1rem; padding-top:0.6rem; border-top:1px solid #2a2a2a; }"""
#endregion


#region MODELS
@dataclass
class InputParam:
    name: str
    type: str
    section: str  # "required" | "optional"
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeDoc:
    folder_name: str
    folder_suffix: str
    folder_path: str
    main_py_file: Optional[str] = None
    main_py_path: Optional[str] = None

    fields: Dict[str, Any] = field(default_factory=dict)

    category: Optional[str] = None
    function_name: Optional[str] = None
    description: Optional[str] = None
    class_name: Optional[str] = None

    inputs: List[InputParam] = field(default_factory=list)
    return_types: List[str] = field(default_factory=list)
    return_names: List[str] = field(default_factory=list)

    image_filename: Optional[str] = None
    image_exists: bool = False

    # derived
    page_filename: str = ""
    page_rel_from_docs: str = ""  # path relative to docs/ (for menu links)

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
        return str(v).strip() if v else ""

    @property
    def version(self) -> str:
        v = self.fields.get(FIELD_VERSION)
        if isinstance(v, list):
            v = v[0] if v else None
        return str(v).strip() if v else ""

    @property
    def group(self) -> str:
        v = self.fields.get(FIELD_GROUP)
        if isinstance(v, list):
            v = v[0] if v else None
        return str(v).strip() if v else "Other"

    @property
    def group_order(self) -> int:
        v = self.fields.get(FIELD_GROUP_ORDER)
        if isinstance(v, list):
            v = v[0] if v else None
        try:
            return int(v) if v is not None else 99
        except (TypeError, ValueError):
            return 99

    @property
    def list_order(self) -> int:
        v = self.fields.get(FIELD_LIST_ORDER)
        if isinstance(v, list):
            v = v[0] if v else None
        try:
            return int(v) if v is not None else 999
        except (TypeError, ValueError):
            return 999

    @property
    def status(self) -> str:
        v = self.fields.get(FIELD_STATUS)
        if isinstance(v, list):
            v = v[0] if v else None
        return str(v).strip() if v else ""

    @property
    def image_declared(self) -> Optional[str]:
        v = self.fields.get(FIELD_IMAGE)
        if isinstance(v, list):
            v = v[0] if v else None
        return str(v).strip() if v else None
#endregion


#region PARSE
def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_value(raw: str) -> Any:
    v = raw.strip()
    if not v:
        return ""
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if re.fullmatch(r"-?\d+", v):
        try:
            return int(v)
        except ValueError:
            return v
    if re.fullmatch(r"-?\d*\.\d+", v):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def parse_dataview_fields(text: str) -> Dict[str, Any]:
    """Parse Obsidian-style KEY:: value lines from a text block (module docstring)."""
    fields: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_\-/]+)\s*::\s*(.*?)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip().upper()
        val = _parse_value(m.group(2))
        if not key:
            continue
        if key in fields:
            existing = fields[key]
            if isinstance(existing, list):
                existing.append(val)
            else:
                fields[key] = [existing, val]
        else:
            fields[key] = val
    return fields


def _find_main_py_file(folder_path: str, folder_suffix: str) -> Tuple[Optional[str], Optional[str]]:
    py_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".py") and f != "__init__.py" and os.path.isfile(os.path.join(folder_path, f))
    ]
    if not py_files:
        return None, None
    if len(py_files) == 1:
        f = py_files[0]
        return f, os.path.join(folder_path, f)

    # Prefer a file that defines NODE_CLASS_MAPPINGS.
    with_mappings: List[str] = []
    for f in py_files:
        try:
            if "NODE_CLASS_MAPPINGS" in _read_text_file(os.path.join(folder_path, f)):
                with_mappings.append(f)
        except OSError:
            pass
    if len(with_mappings) == 1:
        f = with_mappings[0]
        return f, os.path.join(folder_path, f)

    # Snake-case match against the folder suffix.
    expected = re.sub(r"(?<!^)(?=[A-Z])", "_", folder_suffix).lower() + ".py"
    for f in py_files:
        if f.lower() == expected.lower():
            return f, os.path.join(folder_path, f)

    if with_mappings:
        with_mappings.sort()
        f = with_mappings[0]
        return f, os.path.join(folder_path, f)

    py_files.sort()
    f = py_files[0]
    return f, os.path.join(folder_path, f)


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _parse_input_types(class_node: ast.ClassDef) -> List[InputParam]:
    """Parse the INPUT_TYPES classmethod return dict into InputParam rows."""
    params: List[InputParam] = []
    input_types_fn: Optional[ast.FunctionDef] = None
    for stmt in class_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "INPUT_TYPES":
            input_types_fn = stmt
            break
    if input_types_fn is None:
        return params

    # Find the returned dict literal.
    ret_dict: Optional[ast.Dict] = None
    for stmt in ast.walk(input_types_fn):
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
            ret_dict = stmt.value
            break
    if ret_dict is None:
        return params

    section_map: Dict[str, ast.Dict] = {}
    for k_node, v_node in zip(ret_dict.keys, ret_dict.values):
        key = _const_str(k_node)
        if key and isinstance(v_node, ast.Dict):
            section_map[key] = v_node

    for section in ("required", "optional"):
        sec_dict = section_map.get(section)
        if not sec_dict:
            continue
        for k_node, v_node in zip(sec_dict.keys, sec_dict.values):
            param_name = _const_str(k_node)
            if not param_name:
                continue
            ptype, meta = _parse_input_spec(v_node)
            params.append(
                InputParam(
                    name=param_name,
                    type=ptype,
                    section=section,
                    default=meta.get("default"),
                    min=meta.get("min"),
                    max=meta.get("max"),
                    step=meta.get("step"),
                    extra={k: v for k, v in meta.items() if k not in {"default", "min", "max", "step"}},
                )
            )
    return params


def _parse_input_spec(node: ast.AST) -> Tuple[str, Dict[str, Any]]:
    """An input spec is either ("TYPE",) or ("TYPE", { ...options... })."""
    ptype = "ANY"
    meta: Dict[str, Any] = {}
    if isinstance(node, ast.Tuple) and node.elts:
        first = node.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            ptype = first.value
        elif isinstance(first, ast.Name):
            ptype = first.id
        if len(node.elts) >= 2 and isinstance(node.elts[1], ast.Dict):
            meta = _parse_options_dict(node.elts[1])
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        ptype = node.value
    elif isinstance(node, ast.Name):
        ptype = node.id
    return ptype, meta


def _parse_options_dict(d: ast.Dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k_node, v_node in zip(d.keys, d.values):
        key = _const_str(k_node)
        if not key:
            continue
        out[key] = _literal_value(v_node)
    return out


def _literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        return -v if isinstance(v, (int, float)) else v
    if isinstance(node, ast.List):
        return [_literal_value(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(e) for e in node.elts)
    if isinstance(node, ast.Name):
        return node.id
    return None


def _parse_return_tuple(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Tuple):
        return [_literal_value(e) if not isinstance(_literal_value(e), str) else str(_literal_value(e)) for e in node.elts]
    val = _literal_value(node)
    return [str(val)] if val is not None else []


def _parse_node_source(source_text: str) -> Tuple[
    Dict[str, Any], Optional[str], Optional[str], Optional[str], Optional[str],
    List[InputParam], List[str], List[str],
]:
    """Return (frontmatter_fields, category, function, description, class_name, inputs, return_types, return_names)."""
    tree = ast.parse(source_text)
    module_doc = ast.get_docstring(tree, clean=False) or ""
    frontmatter = parse_dataview_fields(module_doc)

    category: Optional[str] = None
    function_name: Optional[str] = None
    description: Optional[str] = None
    class_name: Optional[str] = None
    inputs: List[InputParam] = []
    return_types: List[str] = []
    return_names: List[str] = []

    # First illumorae* class that defines INPUT_TYPES is the node class.
    for stmt in tree.body:
        if not isinstance(stmt, ast.ClassDef):
            continue
        if not stmt.name.startswith("illumorae"):
            continue
        has_input_types = any(
            isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == "INPUT_TYPES"
            for s in stmt.body
        )
        if not has_input_types:
            continue
        class_name = stmt.name
        for body_stmt in stmt.body:
            if not isinstance(body_stmt, ast.Assign):
                continue
            for t in body_stmt.targets:
                if not isinstance(t, ast.Name):
                    continue
                # Tuple-valued attributes (RETURN_TYPES / RETURN_NAMES) are not
                # string constants, so handle them before the _const_str guard.
                if t.id == "RETURN_TYPES":
                    return_types = _parse_return_tuple(body_stmt.value)
                    continue
                if t.id == "RETURN_NAMES":
                    return_names = _parse_return_tuple(body_stmt.value)
                    continue
                val = _const_str(body_stmt.value)
                if val is None:
                    continue
                if t.id == "CATEGORY":
                    category = val
                elif t.id == "FUNCTION":
                    function_name = val
                elif t.id == "DESCRIPTION":
                    description = val
        inputs = _parse_input_types(stmt)
        break

    return frontmatter, category, function_name, description, class_name, inputs, return_types, return_names
#endregion


#region SCAN
def scan_nodes(project_root: str) -> List[NodeDoc]:
    nodes: List[NodeDoc] = []
    for name in sorted(os.listdir(project_root)):
        folder_path = os.path.join(project_root, name)
        if name.startswith("backup") or name.startswith("."):
            continue
        if not os.path.isdir(folder_path):
            continue
        if not name.startswith(NODE_FOLDER_PREFIX):
            continue
        if not os.path.isfile(os.path.join(folder_path, "__init__.py")):
            continue

        suffix = name[len(NODE_FOLDER_PREFIX):]
        node = NodeDoc(folder_name=name, folder_suffix=suffix, folder_path=folder_path)
        node.main_py_file, node.main_py_path = _find_main_py_file(folder_path, suffix)

        if node.main_py_path and os.path.isfile(node.main_py_path):
            src = _read_text_file(node.main_py_path)
            (
                node.fields,
                node.category,
                node.function_name,
                node.description,
                node.class_name,
                node.inputs,
                node.return_types,
                node.return_names,
            ) = _parse_node_source(src)

        # Image lookup under docs/.
        declared = node.image_declared
        if declared:
            candidate = os.path.join(DOCS_DIR, declared)
            if os.path.isfile(candidate):
                node.image_filename = declared
                node.image_exists = True
            else:
                node.image_filename = declared
                node.image_exists = False
        else:
            # Fallback: look for a docs image named after the folder suffix (snake-ish).
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", suffix).lower()
            guesses = [
                f"comfyui_illumorae_{snake}.png",
                f"comfyui_illumorae_{snake}.jpg",
            ]
            for g in guesses:
                if os.path.isfile(os.path.join(DOCS_DIR, g)):
                    node.image_filename = g
                    node.image_exists = True
                    break

        node.page_filename = _page_filename_for(node)
        node.page_rel_from_docs = f"nodes/{node.page_filename}"
        nodes.append(node)

    return nodes


def _page_filename_for(node: NodeDoc) -> str:
    # Use the folder suffix (PascalCase) as the page slug; sanitize for filesystem.
    slug = re.sub(r"[^A-Za-z0-9_\-]", "_", node.folder_suffix)
    return f"{slug}.html"
#endregion


#region HTML
def _esc(text: Any) -> str:
    return html.escape(str(text) if text is not None else "")


def _format_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return _esc(v)


def _status_badge(status: str) -> str:
    s = status.strip().lower()
    if not s:
        return ""
    cls = "badge"
    if s.startswith("work"):
        cls = "badge status-working"
    elif s.startswith("break") or s.startswith("bad") or s.startswith("fail"):
        cls = "badge status-broken"
    elif s.startswith("exp") or s.startswith("wip") or s.startswith("draft"):
        cls = "badge status-experimental"
    return f'<span class="{cls}">{_esc(status)}</span>'


def render_node_page(node: NodeDoc) -> str:
    title = node.title
    short = node.description_short

    meta_rows: List[str] = []
    meta_rows.append(f"<dt>Node ID</dt><dd><code>{_esc(node.class_name or node.folder_suffix)}</code></dd>")
    if node.category:
        meta_rows.append(f"<dt>Category</dt><dd><code>{_esc(node.category)}</code></dd>")
    if node.group:
        meta_rows.append(f"<dt>Group</dt><dd>{_esc(node.group)}</dd>")
    if node.version:
        meta_rows.append(f"<dt>Version</dt><dd><code>{_esc(node.version)}</code></dd>")
    if node.function_name:
        meta_rows.append(f"<dt>Function</dt><dd><code>{_esc(node.function_name)}</code></dd>")
    if node.status:
        meta_rows.append(f"<dt>Status</dt><dd>{_status_badge(node.status)}</dd>")
    meta_rows.append(f"<dt>Package</dt><dd><code>{_esc(node.folder_name)}</code></dd>")
    if node.main_py_file:
        meta_rows.append(f"<dt>Source</dt><dd><code>{_esc(node.main_py_file)}</code></dd>")

    # Image block.
    if node.image_filename and node.image_exists:
        img_block = (
            f'<img class="node-shot" src="../{_esc(node.image_filename)}" '
            f'alt="{_esc(title)}">'
        )
    elif node.image_filename:
        img_block = (
            f'<div class="img-missing">Screenshot declared as '
            f"<code>{_esc(node.image_filename)}</code> but not found under docs/.</div>"
        )
    else:
        img_block = '<div class="img-missing">No screenshot declared (IMAGE:: field missing).</div>'

    # Description block: prefer class DESCRIPTION, fall back to DESCRIPTIONSHORT.
    desc_block = ""
    if node.description:
        desc_block += f"<p>{_esc(node.description)}</p>"
    if short and short != node.description:
        desc_block += f"<p class=\"note\">{_esc(short)}</p>"

    # Inputs table.
    inputs_html = ""
    if node.inputs:
        rows: List[str] = []
        for p in node.inputs:
            default = "" if p.default is None else _format_value(p.default)
            min_v = "" if p.min is None else _format_value(p.min)
            max_v = "" if p.max is None else _format_value(p.max)
            step_v = "" if p.step is None else _format_value(p.step)
            range_cell = ""
            if min_v or max_v:
                range_cell = f"{min_v} .. {max_v}"
                if step_v:
                    range_cell += f" (step {step_v})"
            extra_bits = " ".join(
                f'<span class="badge">{_esc(k)}={_format_value(v)}</span>'
                for k, v in p.extra.items()
            )
            rows.append(
                "<tr>"
                f"<td><code>{_esc(p.name)}</code></td>"
                f"<td>{_esc(p.type)}</td>"
                f"<td>{_esc(p.section)}</td>"
                f"<td>{default}</td>"
                f"<td>{range_cell}</td>"
                f"<td>{extra_bits}</td>"
                "</tr>"
            )
        inputs_html = (
            "<h2>Inputs</h2>\n<table>\n"
            "<tr><th>Name</th><th>Type</th><th>Section</th><th>Default</th><th>Range</th><th>Options</th></tr>\n"
            + "\n".join(rows)
            + "\n</table>"
        )
    else:
        inputs_html = '<h2>Inputs</h2><p class="note">No INPUT_TYPES parameters parsed.</p>'

    # Outputs table.
    outputs_html = ""
    if node.return_types or node.return_names:
        count = max(len(node.return_types), len(node.return_names), 1)
        rows_out: List[str] = []
        for i in range(count):
            t = node.return_types[i] if i < len(node.return_types) else ""
            n = node.return_names[i] if i < len(node.return_names) else ""
            rows_out.append(
                "<tr>"
                f"<td>{i}</td>"
                f"<td><code>{_esc(t)}</code></td>"
                f"<td>{_esc(n)}</td>"
                "</tr>"
            )
        outputs_html = (
            "<h2>Outputs</h2>\n<table>\n"
            "<tr><th>#</th><th>Type</th><th>Name</th></tr>\n"
            + "\n".join(rows_out)
            + "\n</table>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>illumorae - {_esc(title)}</title>
<style>
{PAGE_CSS}
</style>
</head>
<body>
<div class="page">
    <div class="sidebar"><iframe src="../illumorae_menu.htm"></iframe></div>
    <div class="content">
<header>
    <h1>{_esc(title)}</h1>
    <p>{_esc(short)}</p>
</header>
<div class="container">

{img_block}

<h2>Metadata</h2>
<dl class="meta-grid">
{chr(10).join(meta_rows)}
</dl>

{desc_block}

{inputs_html}

{outputs_html}

</div>
    </div>
</div>
</body>
</html>
"""


def render_menu(nodes: List[NodeDoc]) -> str:
    # Group nodes by GROUP:: , sort groups by group_order then name; nodes by list_order then title.
    groups: Dict[str, List[NodeDoc]] = {}
    for n in nodes:
        groups.setdefault(n.group, []).append(n)

    def group_sort_key(g: str) -> Tuple[int, str]:
        # Use the minimal group_order among the group's nodes as the group's order.
        members = groups[g]
        order = min((m.group_order for m in members), default=99)
        return (order, g.lower())

    sections: List[str] = []
    for group in sorted(groups.keys(), key=group_sort_key):
        members = sorted(groups[group], key=lambda n: (n.list_order, n.title.lower()))
        links = "\n".join(
            f'<a href="nodes/{_esc(n.page_filename)}" target="_parent">{_esc(n.title)}</a>'
            for n in members
        )
        sections.append(
            f'<div class="menu-section">\n<h3>{_esc(group)}</h3>\n{links}\n</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>illumorae Menu</title>
<style>
{MENU_CSS}
</style>
</head>
<body>
<center>
<br>
<h2 style="color:#4a7fb5; margin:0; font-size:1.1rem;">ComfyUI ILLUMORAE</h2>
<a href="index.html" target="_parent">Overview</a>
{chr(10).join(sections)}
</center>
</body>
</html>
"""


def render_index(nodes: List[NodeDoc]) -> str:
    total = len(nodes)
    with_image = sum(1 for n in nodes if n.image_exists)
    groups = sorted({n.group for n in nodes}, key=lambda g: (
        next((n.group_order for n in nodes if n.group == g), 99), g.lower()
    ))

    group_rows: List[str] = []
    for g in groups:
        members = sorted([n for n in nodes if n.group == g], key=lambda n: (n.list_order, n.title.lower()))
        links = ", ".join(
            f'<a href="nodes/{_esc(n.page_filename)}">{_esc(n.title)}</a>' for n in members
        )
        group_rows.append(f"<tr><td><strong>{_esc(g)}</strong></td><td>{links}</td></tr>")

    # Build clickable card sections grouped by GROUP.
    card_sections: List[str] = []
    for g in groups:
        members = sorted([n for n in nodes if n.group == g], key=lambda n: (n.list_order, n.title.lower()))
        cards: List[str] = []
        for n in members:
            if n.image_filename and n.image_exists:
                img_html = (
                    f'<img class="card-img" src="{_esc(n.image_filename)}" '
                    f'alt="{_esc(n.title)}">'
                )
            else:
                img_html = '<div class="card-img-missing">no screenshot</div>'
            desc = n.description_short or ""
            cards.append(
                f'<a class="node-card" href="nodes/{_esc(n.page_filename)}">'
                f'{img_html}'
                f'<div class="card-body">'
                f'<div class="card-title">{_esc(n.title)}</div>'
                f'<div class="card-desc">{_esc(desc)}</div>'
                f'</div>'
                f'</a>'
            )
        card_sections.append(
            f'<div class="card-group">\n'
            f'<h3>{_esc(g)}</h3>\n'
            f'<div class="card-grid">\n'
            f'{chr(10).join(cards)}\n'
            f'</div>\n'
            f'</div>'
        )

    index_css = """
.node-card { display:flex; flex-direction:column; background:var(--surface); border:1px solid var(--border); border-radius:6px; overflow:hidden; text-decoration:none; color:inherit; transition:border-color 0.15s, transform 0.15s; }
.node-card:hover { border-color:#4a7fb5; transform:translateY(-2px); }
.card-img { width:100%; height:140px; object-fit:cover; display:block; background:#1a1a1a; }
.card-img-missing { width:100%; height:140px; display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:0.8rem; background:#1a1a1a; border-bottom:1px solid var(--border); }
.card-body { padding:0.6rem 0.8rem; }
.card-title { color:#4a7fb5; font-size:0.9rem; font-weight:600; margin-bottom:0.2rem; }
.card-desc { color:var(--muted); font-size:0.78rem; line-height:1.3; }
.card-group { margin-top:2rem; }
.card-group h3 { color:#ffffff; font-size:0.95rem; text-transform:uppercase; margin:0 0 0.8rem 0; padding-bottom:0.3rem; border-bottom:1px solid var(--border); }
.card-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:0.8rem; }
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>illumorae - Overview</title>
<style>
{PAGE_CSS}
{index_css}
</style>
</head>
<body>
<div class="page">
    <div class="sidebar"><iframe src="illumorae_menu.htm"></iframe></div>
    <div class="content">
<header>
    <h1>ComfyUI ILLUMORAE</h1>
</header>
<div class="container">

<table>
{chr(10).join(group_rows)}
</table>

{chr(10).join(card_sections)}

</div>
    </div>
</div>
</body>
</html>
"""
#endregion


#region WRITE
def _write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _match_refresh(node: NodeDoc, tokens: List[str]) -> bool:
    tokens_low = {t.lower() for t in tokens}
    if node.folder_name.lower() in tokens_low:
        return True
    if node.folder_suffix.lower() in tokens_low:
        return True
    if node.title.lower() in tokens_low:
        return True
    # Allow matching by page slug without extension.
    if node.page_filename[:-5].lower() in tokens_low:
        return True
    return False
#endregion


#region CLI
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate static HTML docs for ComfyUI_illumorae nodes."
    )
    parser.add_argument(
        "project_root",
        nargs="?",
        default=REPO_ROOT,
        help="Repository root containing ComfyUI_illumorae_* packages (default: parent of tools/).",
    )
    parser.add_argument(
        "--refresh",
        nargs="*",
        default=None,
        metavar="NODE",
        help="Force-regenerate the listed node pages (match folder name, suffix, or TITLE).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Force-regenerate every node page.",
    )
    parser.add_argument(
        "--menu-only",
        action="store_true",
        help="Only regenerate the sidebar menu and index, leave node pages untouched.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without writing any files.",
    )
    args = parser.parse_args(argv)

    project_root = os.path.abspath(args.project_root)
    if not os.path.isdir(project_root):
        print(f"ERROR: project root not found: {project_root}", file=sys.stderr)
        return 2

    nodes = scan_nodes(project_root)
    if not nodes:
        print("No ComfyUI_illumorae_* node packages found.", file=sys.stderr)
        return 1

    nodes.sort(key=lambda n: (n.group, n.list_order, n.title.lower()))

    refresh_tokens = list(args.refresh) if args.refresh else []
    force_all = bool(args.all)
    menu_only = bool(args.menu_only)
    dry = bool(args.dry_run)

    written: List[str] = []
    skipped: List[str] = []

    # Menu + index are always regenerated (they must reflect the current node set).
    menu_html = render_menu(nodes)
    index_html = render_index(nodes)
    if dry:
        print(f"[dry-run] would write {os.path.relpath(MENU_FILE, project_root)}")
        print(f"[dry-run] would write {os.path.relpath(INDEX_FILE, project_root)}")
    else:
        _write_file(MENU_FILE, menu_html)
        _write_file(INDEX_FILE, index_html)
        written.append(os.path.relpath(MENU_FILE, project_root))
        written.append(os.path.relpath(INDEX_FILE, project_root))

    if not menu_only:
        for node in nodes:
            page_path = os.path.join(NODES_DIR, node.page_filename)
            rel = os.path.relpath(page_path, project_root)
            exists = os.path.isfile(page_path)

            should_write = force_all
            if not should_write and refresh_tokens:
                should_write = _match_refresh(node, refresh_tokens)
            if not should_write and not exists:
                should_write = True

            if not should_write:
                skipped.append(rel)
                continue

            if dry:
                print(f"[dry-run] would write {rel}")
                continue

            _write_file(page_path, render_node_page(node))
            written.append(rel)

    print(f"Nodes scanned : {len(nodes)}")
    print(f"Files written : {len(written)}")
    if written:
        for w in written:
            print(f"  + {w}")
    if skipped:
        print(f"Files skipped : {len(skipped)} (already exist; use --refresh or --all to update)")
        for s in skipped[:20]:
            print(f"  - {s}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
    return 0
#endregion


if __name__ == "__main__":
    raise SystemExit(main())
