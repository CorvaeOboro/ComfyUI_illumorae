#region DOC
# Module docstring: naming conventions, validation checks, security policy, usage
"""
illumorae Node Metadata Dashboard
-------------------------------
Scans ComfyUI_illumorae_* node packages in this repository, extracts metadata from:
- Python sources (NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS / CATEGORY / FUNCTION)
- Obsidian-style DataView fields in notes/*.md (KEY:: value)

NAMING CONVENTIONS & STYLE GUIDE RULES
=======================================

1. FOLDER NAMING
   - Pattern: ComfyUI_illumorae_<FunctionalName>
   - Must start with "ComfyUI_illumorae_" prefix
   - Use PascalCase for functional name (no underscores in functional part)
   - Example: ComfyUI_illumorae_ImageResizeFluxKontextOptions
   - Example: ComfyUI_illumorae_LoadTextFileGraceful

2. PYTHON FILENAME
   - Pattern: <snake_case_functional_name>.py
   - Must be snake_case version of folder's functional name
   - Should match folder suffix converted to snake_case
   - Example: image_resize_flux_kontext_options.py (for folder ImageResizeFluxKontextOptions)
   - Example: load_text_file_graceful.py (for folder LoadTextFileGraceful)

3. PYTHON CLASS NAME
   - Pattern: illumorae<FunctionalName> or illumorae<FunctionalName>Node
   - Must start with "illumorae" prefix (lowercase)
   - Use PascalCase for functional name (same as folder suffix)
   - Optional "Node" suffix
   - Example: illumoraeImageResizeFluxKontextOptionsNode
   - Example: illumoraeLoadTextFileGraceful

4. NODE ID (NODE_CLASS_MAPPINGS key)
   - Pattern: illumorae<FunctionalName> (matches class name)
   - Must be unique identifier for ComfyUI node registration
   - Should match the class name exactly
   - Example: 'illumoraeImageResizeFluxKontextOptionsNode'
   - Example: 'illumoraeLoadTextFileGraceful'

5. DISPLAY NAME (NODE_DISPLAY_NAME_MAPPINGS value)
   - Pattern: Human-readable title with spaces
   - Use Title Case (capitalize major words)
   - No "illumorae" prefix in display name
   - Should be clear and descriptive for UI
   - Example: "Image Resize Flux Kontext Scale Options"
   - Example: "Load Text File Graceful"

6. CATEGORY
   - Must be "illumorae" (lowercase)
   - All nodes share this common category for organization
   - Example: CATEGORY = "illumorae"

7. PYTHON FRONTMATTER METADATA
   - Located in module docstring at top of .py file
   - Uses Obsidian DataView format: KEY::value (no quotes, no spaces around ::)
   - Required fields:
     * TITLE::Display Name
     * DESCRIPTIONSHORT::Brief description
     * VERSION::YYYYMMDD
   - Optional fields:
     * GROUP::Category (e.g., Image, Text, Load, Save, Lora, VLM, Select)
     * GROUPORDER::Integer (for sorting within group)
     * LISTORDER::Integer (for overall sorting)
     * IMAGE::screenshot_filename.png
   - Example:
     TITLE::Image Resize Flux Kontext Scale Options
     DESCRIPTIONSHORT::Resizes images to preferred Flux Kontext resolutions.
     VERSION::20260113
     GROUP::Image

8. CONSISTENCY RULES
   - Folder suffix = Python class suffix = Node ID suffix
   - Python filename = snake_case(folder suffix)
   - Display name = human readable version of folder suffix
   - All components must derive from same canonical word list
   - Example consistency chain:
     * Folder: ComfyUI_illumorae_LoadTextFileGraceful
     * File: load_text_file_graceful.py
     * Class: illumoraeLoadTextFileGraceful
     * Node ID: illumoraeLoadTextFileGraceful
     * Display: "Load Text File Graceful"
     * Title: Load Text File Graceful

9. GROUP CATEGORIES
   - Standard groups: Image, Text, Load, Save, Lora, Checkpoint, VLM, Select
   - Groups organize nodes by functionality
   - Should be singular form (Image not Images)
   - Title Case in metadata
   - Example: GROUP::Image, GROUP::Text, GROUP::Load

10. VERSION FORMAT
    - Pattern: YYYYMMDD (8 digits)
    - Represents last modification date
    - Example: VERSION::20260113 (January 13, 2026)

VALIDATION CHECKS
=================
The dashboard performs these automated checks:
- Folder name starts with ComfyUI_illumorae_
- Python file matches snake_case of folder suffix
- Class name matches PascalCase of folder suffix
- Node ID matches class name
- Display name is human-readable version
- CATEGORY is "illumorae"
- Required frontmatter fields present (TITLE, DESCRIPTIONSHORT, VERSION)
- Naming consistency across all components
- No duplicate node IDs across repository
- limited imports ( avoid subprocess, requests, etc.) for security 

SECURITY CONCERNS
==============
when possible avoid imports that would require the node pack to be audited each update because of potential risk of 
11. DANGEROUS IMPORTS
    - NEVER use subprocess module (RCE risk)
    - AVOID requests module unless absolutely necessary (network access)
    - AVOID os.system, eval, exec (code execution risks)
    - Dangerous imports flagged in red in dashboard
    - Dashboard shows all imports across nodes for anomaly detection
    - Whitelist: Standard library (json, os.path, re, etc.), torch, PIL, numpy, comfy

Usage:
    python illumorae_node_metadata_dashboard.py [project_root]

Optional flags:
    --no-gui
    --export-json <path>

VERSION: 20260113
"""
#endregion

#region IMPORTS
# Standard library + tkinter imports
from __future__ import annotations

import ast
import argparse
import json
import os
import re
import sys
import tkinter as tk
from dataclasses import dataclass, asdict, field
from datetime import datetime
from tkinter import ttk, filedialog
from typing import Any, Dict, List, Optional, Tuple
#endregion


#region CONST
# Dark-theme color palette for the Tkinter dashboard
COLORS = {
    "bg_app": "#1e1e1e",       # app UI background (dark gray, slightly lighter than black)
    "bg_dark": "#1a1a1a",       # header background
    "bg_medium": "#2d2d2d",     # sorted-header / hover background
    "bg_light": "#3a3a3a",      # selected-row highlight
    "bg_black": "#000000",      # pure black for text input fields
    "fg_text": "#ffffff",
    "fg_dim": "#b0b0b0",
    "fg_grey": "#808080",       # medium gray for zero-issue counts
    "fg_light_grey": "#c0c0c0", # light grey for combobox text
    "accent_blue": "#4a7ba7",
    "accent_green": "#5a8a5a",
    "accent_red": "#a75a5a",
    "accent_red_bright": "#ff4444",  # bright red for highest issue count
    "accent_yellow": "#a79a5a",
    "border": "#404040",
    "scroll_bg": "#0a0a0a",     # scrollbar trough (near-black)
    "scroll_thumb": "#5a7a9a",  # light muted blue for scrollbar thumb + arrows
}
#endregion


#region MODELS
# Dataclasses: NodeIssue (validation) and NodeMetadata (per-node extracted data)
@dataclass
class NodeIssue:
    severity: str
    message: str


@dataclass
class NodeMetadata:
    folder_name: str
    folder_path: str
    folder_suffix: str

    main_py_file: Optional[str] = None
    main_py_path: Optional[str] = None

    category: Optional[str] = None
    function_name: Optional[str] = None

    node_class_mappings: Dict[str, str] = field(default_factory=dict)
    node_display_name_mappings: Dict[str, str] = field(default_factory=dict)

    classes_defined: List[str] = field(default_factory=list)

    obsidian_fields: Dict[str, Any] = field(default_factory=dict)
    global_fields: Dict[str, Any] = field(default_factory=dict)

    python_frontmatter: Dict[str, Any] = field(default_factory=dict)

    class_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    rule_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    canonical_words: List[str] = field(default_factory=list)

    suggested_folder_name: Optional[str] = None
    suggested_py_file: Optional[str] = None
    suggested_class_name: Optional[str] = None
    suggested_node_id: Optional[str] = None
    suggested_display_name: Optional[str] = None
    suggested_group: Optional[str] = None

    imports: List[str] = field(default_factory=list)
    dangerous_imports: List[str] = field(default_factory=list)

    issues: List[NodeIssue] = field(default_factory=list)

    def issues_count(self) -> int:
        return len(self.issues)
#endregion


#region TEXTUTIL
# Identifier splitting and case conversion helpers (snake/Pascal/title)
def _split_words(identifier: str) -> List[str]:
    s = (identifier or "").strip()
    if not s:
        return []

    if "_" in s:
        parts = [p for p in s.split("_") if p]
        return [p for p in parts if p]

    if "-" in s:
        parts = [p for p in s.split("-") if p]
        return [p for p in parts if p]

    return re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+", s)


def _to_snake(words: List[str]) -> str:
    return "_".join(w.lower() for w in words if w)


def _to_pascal(words: List[str]) -> str:
    return "".join(w[:1].upper() + w[1:] for w in words if w)


def _to_title(words: List[str]) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in words if w)


def display_name_to_node_id(display_name: str) -> str:
    """Convert display name to expected NodeID format: illumorae + PascalCase + Node"""
    if not display_name:
        return ""
    # Remove spaces and convert to PascalCase
    words = display_name.split()
    pascal_case = "".join(w[:1].upper() + w[1:] for w in words if w)
    return f"illumorae{pascal_case}Node"


def display_name_to_py_file(display_name: str) -> str:
    """Convert display name to expected Python filename: snake_case.py"""
    if not display_name:
        return ""
    # Convert to lowercase and replace spaces with underscores
    snake_case = display_name.lower().replace(" ", "_")
    return f"{snake_case}.py"
#endregion


#region PARSE
# Obsidian-style DataView field parsing (KEY:: value) and value type coercion
def _merge_field(dest: Dict[str, Any], key: str, value: Any) -> None:
    if key in dest:
        existing = dest[key]
        if isinstance(existing, list):
            existing.append(value)
            return
        dest[key] = [existing, value]
        return
    dest[key] = value


def _parse_value(raw: str) -> Any:
    v = raw.strip()
    if not v:
        return ""

    if (v.startswith("\"") and v.endswith("\"")) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]

    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False

    if re.fullmatch(r"-?\d+", v):
        try:
            return int(v)
        except Exception:
            return v

    if re.fullmatch(r"-?\d*\.\d+", v):
        try:
            return float(v)
        except Exception:
            return v

    if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
        try:
            return json.loads(v)
        except Exception:
            return v

    return v


def parse_obsidian_fields(markdown_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fields: Dict[str, Any] = {}
    globals_out: Dict[str, Any] = {}

    yaml_block: List[str] = []
    lines = markdown_text.splitlines()

    if len(lines) >= 3 and lines[0].strip() == "---":
        i = 1
        while i < len(lines):
            if lines[i].strip() == "---":
                yaml_block = lines[1:i]
                break
            i += 1

    for line in yaml_block:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        val = _parse_value(v.strip())
        if key:
            _merge_field(fields, key, val)

    for line in lines:
        m = re.match(r"^\s*([A-Za-z0-9_\-/]+)\s*::\s*(.*?)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip()
        val = _parse_value(m.group(2))
        if key:
            _merge_field(fields, key, val)

    for k, v in list(fields.items()):
        if k.upper().startswith("GLOBAL"):
            globals_out[k] = v

    return fields, globals_out


def parse_dataview_fields(text: str) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_\-/]+)\s*::\s*(.*?)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip()
        val = _parse_value(m.group(2))
        if key:
            _merge_field(fields, key, val)
    return fields
#endregion


#region FILEIO
# File reading, main-py discovery, import extraction, AST-based node parsing
def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _find_main_py_file(folder_path: str, folder_suffix: str) -> Tuple[Optional[str], Optional[str]]:
    py_files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".py") and f != "__init__.py" and os.path.isfile(os.path.join(folder_path, f))
    ]
    if not py_files:
        return None, None

    if len(py_files) == 1:
        f = py_files[0]
        return f, os.path.join(folder_path, f)

    # Strategy 1: Check for NODE_CLASS_MAPPINGS in each file (most reliable)
    files_with_mappings = []
    for f in py_files:
        try:
            content = _read_text_file(os.path.join(folder_path, f))
            if "NODE_CLASS_MAPPINGS" in content:
                files_with_mappings.append(f)
        except Exception:
            pass
    
    if len(files_with_mappings) == 1:
        f = files_with_mappings[0]
        return f, os.path.join(folder_path, f)
    
    # Strategy 2: Try exact match with expected snake_case name
    expected = _to_snake(_split_words(folder_suffix)) + ".py"
    for f in py_files:
        if f.lower() == expected.lower():
            return f, os.path.join(folder_path, f)
    
    # Strategy 3: Try fuzzy match - check if filename contains key parts of suffix
    suffix_lower = folder_suffix.lower().replace("_", "")
    best_match = None
    best_score = 0
    for f in py_files:
        f_lower = f[:-3].lower().replace("_", "")  # Remove .py and underscores
        # Count how many characters match
        score = sum(1 for c in suffix_lower if c in f_lower)
        if score > best_score:
            best_score = score
            best_match = f
    
    if best_match and best_score > len(suffix_lower) * 0.6:  # 60% match threshold
        return best_match, os.path.join(folder_path, best_match)
    
    # Strategy 4: Prefer files_with_mappings if we found any
    if files_with_mappings:
        files_with_mappings.sort()
        f = files_with_mappings[0]
        return f, os.path.join(folder_path, f)

    # Strategy 5: Fall back to alphabetical
    py_files.sort()
    f = py_files[0]
    return f, os.path.join(folder_path, f)


def _extract_imports(source_text: str) -> List[str]:
    """Extract all import statements from Python source."""
    imports = []
    try:
        tree = ast.parse(source_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
    except Exception:
        pass
    return sorted(set(imports))


def _parse_node_python(
    source_text: str,
) -> Tuple[
    List[str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, Dict[str, Any]],
    Optional[str],
    Optional[str],
    Dict[str, Any],
    List[str],
]:
    classes: List[str] = []
    class_map: Dict[str, str] = {}
    display_map: Dict[str, str] = {}
    class_meta: Dict[str, Dict[str, Any]] = {}
    category: Optional[str] = None
    function_name: Optional[str] = None
    frontmatter: Dict[str, Any] = {}

    tree = ast.parse(source_text)

    module_doc = ast.get_docstring(tree, clean=False) or ""
    frontmatter = parse_dataview_fields(module_doc)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)

            meta: Dict[str, Any] = {}
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                for t in stmt.targets:
                    if not isinstance(t, ast.Name):
                        continue
                    if not (isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)):
                        continue
                    if t.id in {"CATEGORY", "FUNCTION", "DESCRIPTION"}:
                        meta[t.id] = stmt.value.value
            if meta:
                class_meta[node.name] = meta

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "CATEGORY":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        category = node.value.value
                if isinstance(t, ast.Name) and t.id == "FUNCTION":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        function_name = node.value.value

            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS":
                    if isinstance(node.value, ast.Dict):
                        for k_node, v_node in zip(node.value.keys, node.value.values):
                            if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                                continue
                            key = k_node.value
                            if isinstance(v_node, ast.Name):
                                class_map[key] = v_node.id
                            elif isinstance(v_node, ast.Attribute):
                                class_map[key] = v_node.attr

                if isinstance(t, ast.Name) and t.id == "NODE_DISPLAY_NAME_MAPPINGS":
                    if isinstance(node.value, ast.Dict):
                        for k_node, v_node in zip(node.value.keys, node.value.values):
                            if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                                continue
                            if not (isinstance(v_node, ast.Constant) and isinstance(v_node.value, str)):
                                continue
                            display_map[k_node.value] = v_node.value

    imports = _extract_imports(source_text)
    return classes, class_map, display_map, class_meta, category, function_name, frontmatter, imports
#endregion


#region SCAN
# NodeMetadataScanner: discovers packages, extracts metadata, runs validation
class NodeMetadataScanner:
    #region S-INIT
    # Constructor and main scan loop: iterates ComfyUI_illumorae_* folders
    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        self.nodes: List[NodeMetadata] = []
        self.global_variables: Dict[str, Any] = {}

    def scan(self) -> List[NodeMetadata]:
        self.nodes = []
        self.global_variables = {}

        for name in sorted(os.listdir(self.project_root)):
            folder_path = os.path.join(self.project_root, name)

            if name.startswith("backup") or name.startswith("."):
                continue
            if not os.path.isdir(folder_path):
                continue
            if not name.startswith("ComfyUI_illumorae_"):
                continue
            if not os.path.isfile(os.path.join(folder_path, "__init__.py")):
                continue

            suffix = name[len("ComfyUI_illumorae_") :]
            node = NodeMetadata(folder_name=name, folder_path=folder_path, folder_suffix=suffix)

            node.main_py_file, node.main_py_path = _find_main_py_file(folder_path, suffix)
            if node.main_py_path and os.path.isfile(node.main_py_path):
                src = _read_text_file(node.main_py_path)
                (
                    node.classes_defined,
                    node.node_class_mappings,
                    node.node_display_name_mappings,
                    node.class_metadata,
                    node.category,
                    node.function_name,
                    node.python_frontmatter,
                    node.imports,
                ) = _parse_node_python(src)
                
                # Check for dangerous imports
                dangerous = ["subprocess", "requests", "urllib", "http", "socket", "eval", "exec"]
                node.dangerous_imports = [imp for imp in node.imports if imp in dangerous]
            else:
                node.issues.append(NodeIssue("error", "No main .py file found (excluding __init__.py)."))

            self._load_notes(node)
            self._derive_conventions(node)
            self._validate(node)
            self._build_rule_checks(node)

            for k, v in node.global_fields.items():
                if k not in self.global_variables:
                    self.global_variables[k] = v

            self.nodes.append(node)

        return self.nodes
    #endregion

    #region S-NOTES
    # Load Obsidian-style fields from notes/*.md (see review §2.11 — unintended path)
    def _load_notes(self, node: NodeMetadata) -> None:
        notes_dir = os.path.join(node.folder_path, "notes")
        if not os.path.isdir(notes_dir):
            return

        for fname in sorted(os.listdir(notes_dir)):
            if not fname.lower().endswith(".md"):
                continue
            fpath = os.path.join(notes_dir, fname)
            if not os.path.isfile(fpath):
                continue
            text = _read_text_file(fpath)
            fields, globals_out = parse_obsidian_fields(text)

            for k, v in fields.items():
                _merge_field(node.obsidian_fields, k, v)
            for k, v in globals_out.items():
                _merge_field(node.global_fields, k, v)
    #endregion

    #region S-DERIVE
    # Derive canonical words and suggested folder/file/class/node-id/display names
    def _derive_conventions(self, node: NodeMetadata) -> None:
        canonical = node.obsidian_fields.get("NODE_CANONICAL")
        if isinstance(canonical, list):
            canonical = canonical[0] if canonical else None

        display = node.obsidian_fields.get("NODE_DISPLAY")
        if isinstance(display, list):
            display = display[0] if display else None

        if not canonical and isinstance(display, str) and display.strip():
            canonical = display

        if not canonical:
            if node.node_display_name_mappings:
                canonical = next(iter(node.node_display_name_mappings.values()))

        if not canonical:
            title = node.python_frontmatter.get("TITLE")
            if isinstance(title, list):
                title = title[0] if title else None
            if isinstance(title, str) and title.strip():
                canonical = title

        if not canonical:
            canonical = node.folder_suffix

        node.canonical_words = _split_words(str(canonical))

        group = node.obsidian_fields.get("NODE_GROUP")
        if isinstance(group, list):
            group = group[0] if group else None

        if not isinstance(group, str) or not group.strip():
            group = node.canonical_words[0] if node.canonical_words else ""

        node.suggested_group = group
        node.suggested_folder_name = "ComfyUI_illumorae_" + _to_pascal(node.canonical_words)
        node.suggested_py_file = _to_snake(node.canonical_words) + ".py"
        node.suggested_class_name = "illumorae" + _to_pascal(node.canonical_words) + "Node"
        node.suggested_node_id = node.suggested_class_name
        node.suggested_display_name = _to_title(node.canonical_words)

        override_py = node.obsidian_fields.get("NODE_PY_FILE")
        if isinstance(override_py, list):
            override_py = override_py[0] if override_py else None
        if isinstance(override_py, str) and override_py.strip():
            node.suggested_py_file = override_py.strip()

        override_class = node.obsidian_fields.get("NODE_CLASS")
        if isinstance(override_class, list):
            override_class = override_class[0] if override_class else None
        if isinstance(override_class, str) and override_class.strip():
            node.suggested_class_name = override_class.strip()

        override_id = node.obsidian_fields.get("NODE_ID")
        if isinstance(override_id, list):
            override_id = override_id[0] if override_id else None
        if isinstance(override_id, str) and override_id.strip():
            node.suggested_node_id = override_id.strip()

        override_display = node.obsidian_fields.get("NODE_DISPLAY")
        if isinstance(override_display, list):
            override_display = override_display[0] if override_display else None
        if isinstance(override_display, str) and override_display.strip():
            node.suggested_display_name = override_display.strip()
    #endregion

    #region S-VALID
    # Check CATEGORY, mappings, filename, frontmatter, imports; populate issues list
    def _validate(self, node: NodeMetadata) -> None:
        if node.category and node.category != "illumorae":
            node.issues.append(NodeIssue("warning", f"CATEGORY is '{node.category}' (expected 'illumorae')."))

        if not node.node_class_mappings:
            node.issues.append(NodeIssue("error", "Missing NODE_CLASS_MAPPINGS."))

        if node.node_class_mappings and not node.node_display_name_mappings:
            node.issues.append(NodeIssue("warning", "Missing NODE_DISPLAY_NAME_MAPPINGS."))

        if node.main_py_file and node.suggested_py_file and node.main_py_file != node.suggested_py_file:
            node.issues.append(
                NodeIssue(
                    "warning",
                    f"Main python filename is '{node.main_py_file}' (suggested '{node.suggested_py_file}').",
                )
            )

        for node_id, class_name in node.node_class_mappings.items():
            if class_name not in node.classes_defined:
                node.issues.append(NodeIssue("error", f"Mapping '{node_id}' references missing class '{class_name}'."))

            if node_id != class_name:
                node.issues.append(
                    NodeIssue(
                        "warning",
                        f"Mapping key '{node_id}' != class name '{class_name}'. Consider NODE_ID override if intentional.",
                    )
                )

            if node_id not in node.node_display_name_mappings:
                node.issues.append(NodeIssue("warning", f"No display name entry for node id '{node_id}'."))

        if node.folder_name != node.suggested_folder_name:
            node.issues.append(
                NodeIssue(
                    "info",
                    f"Folder name '{node.folder_name}' (suggested '{node.suggested_folder_name}').",
                )
            )

        if not node.python_frontmatter.get("TITLE"):
            node.issues.append(NodeIssue("warning", "Missing python frontmatter TITLE:: in module docstring."))
        if not node.python_frontmatter.get("VERSION"):
            node.issues.append(NodeIssue("warning", "Missing python frontmatter VERSION:: in module docstring."))
        if not node.python_frontmatter.get("DESCRIPTIONSHORT"):
            node.issues.append(NodeIssue("info", "Missing python frontmatter DESCRIPTIONSHORT:: in module docstring."))
        if not node.python_frontmatter.get("IMAGE"):
            node.issues.append(NodeIssue("info", "Missing python frontmatter IMAGE:: in module docstring."))

        for node_id, class_name in node.node_class_mappings.items():
            meta = node.class_metadata.get(class_name, {})
            if not meta.get("DESCRIPTION"):
                node.issues.append(NodeIssue("info", f"Class '{class_name}' is missing DESCRIPTION attribute."))
        
        if node.dangerous_imports:
            node.issues.append(
                NodeIssue(
                    "error",
                    f"Dangerous imports detected: {', '.join(node.dangerous_imports)}. Avoid subprocess, requests for security."
                )
            )
    #endregion

    #region S-RULES
    # Build per-rule ok/fail map for dashboard status cells (frontmatter, naming, security)
    def _build_rule_checks(self, node: NodeMetadata) -> None:
        checks: Dict[str, Dict[str, Any]] = {}

        def add(rule_id: str, ok: bool, details: str, severity: str = "info") -> None:
            checks[rule_id] = {"ok": bool(ok), "details": details, "severity": severity}

        title = node.python_frontmatter.get("TITLE")
        desc_short = node.python_frontmatter.get("DESCRIPTIONSHORT")
        version = node.python_frontmatter.get("VERSION")
        image = node.python_frontmatter.get("IMAGE")
        group = node.python_frontmatter.get("GROUP")

        add("frontmatter_title", bool(title), "TITLE:: present" if title else "TITLE:: missing", "warning")
        add("frontmatter_version", bool(version), "VERSION:: present" if version else "VERSION:: missing", "warning")
        add(
            "frontmatter_descriptionshort",
            bool(desc_short),
            "DESCRIPTIONSHORT:: present" if desc_short else "DESCRIPTIONSHORT:: missing",
            "info",
        )
        add("frontmatter_group", bool(group), "GROUP:: present" if group else "GROUP:: missing", "info")
        
        # Check IMAGE:: field and validate file exists
        image_exists = False
        image_details = "IMAGE:: missing"
        if image:
            image_str = str(image)
            docs_path = os.path.join(self.project_root, "docs", image_str)
            if os.path.isfile(docs_path):
                image_exists = True
                image_details = f"IMAGE:: {image_str} (exists)"
            else:
                image_details = f"IMAGE:: {image_str} (file not found)"
        add("frontmatter_image", image_exists, image_details, "info")

        add(
            "folder_name",
            node.folder_name == node.suggested_folder_name,
            f"{node.folder_name} == {node.suggested_folder_name}",
            "info",
        )
        py_file_match = bool(node.main_py_file) and (node.main_py_file == node.suggested_py_file)
        add(
            "main_py_file",
            py_file_match,
            f"py file: {node.main_py_file or ''} {'==' if py_file_match else '!='} {node.suggested_py_file or ''}",
            "warning",
        )
        
        # Check class name follows convention
        for node_id, class_name in node.node_class_mappings.items():
            expected_class = node.suggested_class_name
            class_match = class_name == expected_class
            add(
                f"class_name_convention::{class_name}",
                class_match,
                f"class: {class_name} {'==' if class_match else '!='} {expected_class}",
                "info",
            )

        add("has_class_mappings", bool(node.node_class_mappings), "NODE_CLASS_MAPPINGS present" if node.node_class_mappings else "NODE_CLASS_MAPPINGS missing", "error")
        add(
            "has_display_mappings",
            bool(node.node_display_name_mappings),
            "NODE_DISPLAY_NAME_MAPPINGS present" if node.node_display_name_mappings else "NODE_DISPLAY_NAME_MAPPINGS missing",
            "warning",
        )
        
        # Check if NodeID ends with "Node"
        for node_id, class_name in node.node_class_mappings.items():
            ends_with_node = node_id.endswith("Node")
            add(
                f"node_id_ends_with_node::{node_id}",
                ends_with_node,
                f"NodeID '{node_id}' {'ends with' if ends_with_node else 'does not end with'} 'Node'",
                "info",
            )
        
        # Check NodeID matches expected format from display name
        for node_id in node.node_class_mappings.keys():
            display_name = node.node_display_name_mappings.get(node_id, "")
            if display_name:
                expected_node_id = display_name_to_node_id(display_name)
                matches = node_id == expected_node_id
                add(
                    f"node_id_matches_display::{node_id}",
                    matches,
                    f"NodeID '{node_id}' {'matches' if matches else 'does not match'} expected '{expected_node_id}' from display '{display_name}'",
                    "info",
                )
        
        # Check Python filename matches expected format from display name
        if node.main_py_file and node.node_display_name_mappings:
            # Use first display name as reference
            display_name = next(iter(node.node_display_name_mappings.values()), "")
            if display_name:
                expected_py_file = display_name_to_py_file(display_name)
                matches = node.main_py_file == expected_py_file
                add(
                    "py_file_matches_display",
                    matches,
                    f"Python file '{node.main_py_file}' {'matches' if matches else 'does not match'} expected '{expected_py_file}' from display '{display_name}'",
                    "info",
                )

        for node_id, class_name in node.node_class_mappings.items():
            meta = node.class_metadata.get(class_name, {})
            add(
                f"mapping_id_eq_class::{node_id}",
                node_id == class_name,
                f"{node_id} == {class_name}",
                "warning",
            )
            add(
                f"display_mapping_exists::{node_id}",
                node_id in node.node_display_name_mappings,
                "display mapping present" if node_id in node.node_display_name_mappings else "display mapping missing",
                "warning",
            )
            add(
                f"class_description_present::{class_name}",
                bool(meta.get("DESCRIPTION")),
                "DESCRIPTION present" if meta.get("DESCRIPTION") else "DESCRIPTION missing",
                "info",
            )
            if isinstance(desc_short, str) and isinstance(meta.get("DESCRIPTION"), str):
                fm = " ".join(desc_short.split())
                cd = " ".join(meta.get("DESCRIPTION", "").split())
                add(
                    f"description_matches_frontmatter::{class_name}",
                    fm == cd,
                    "matches DESCRIPTIONSHORT" if fm == cd else "differs from DESCRIPTIONSHORT",
                    "info",
                )

        # Security check: dangerous imports
        has_dangerous = len(node.dangerous_imports) > 0
        dangerous_list = ", ".join(node.dangerous_imports) if has_dangerous else "none"
        add(
            "no_dangerous_imports",
            not has_dangerous,
            f"Dangerous imports: {dangerous_list}",
            "error" if has_dangerous else "info",
        )

        node.rule_checks = checks
    #endregion
#endregion


#region UI
# NodeMetadataDashboard: Tkinter GUI with canvas table, details panel, actions
class NodeMetadataDashboard:
    #region U-INIT
    # Constructor: wires scanner, builds UI, triggers first load
    def __init__(self, root: tk.Tk, project_root: str):
        self.root = root
        self.project_root = os.path.abspath(project_root)
        self.scanner = NodeMetadataScanner(self.project_root)

        self.nodes: List[NodeMetadata] = []
        self.filtered_nodes: List[NodeMetadata] = []
        self.selected_node: Optional[NodeMetadata] = None

        self._setup_ui()
        self.load_nodes()
    #endregion

    #region U-THEME
    # Dark-theme styling for all ttk widgets
    def _apply_dark_theme(self) -> None:
        self.root.configure(bg=COLORS["bg_app"])
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            ".",
            background=COLORS["bg_app"],
            foreground=COLORS["fg_text"],
            fieldbackground=COLORS["bg_black"],
            bordercolor=COLORS["border"],
        )
        style.configure(
            "Treeview",
            background=COLORS["bg_app"],
            foreground=COLORS["fg_text"],
            fieldbackground=COLORS["bg_app"],
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background=COLORS["bg_dark"],
            foreground=COLORS["fg_text"],
            relief="flat",
        )
        style.configure(
            "TButton",
            background=COLORS["accent_blue"],
            foreground=COLORS["fg_text"],
            borderwidth=1,
            relief="flat",
            padding=6,
        )
        style.map("TButton", background=[("active", COLORS["bg_light"])])

        # Entry fields: pure black background
        style.configure("TEntry", fieldbackground=COLORS["bg_black"], foreground=COLORS["fg_text"], bordercolor=COLORS["border"], lightcolor=COLORS["border"], darkcolor=COLORS["border"])
        style.map("TEntry", fieldbackground=[("focus", COLORS["bg_black"])], bordercolor=[("focus", COLORS["accent_blue"])])

        # Combobox (groups dropdown): pure black background, light grey text
        style.configure("TCombobox", fieldbackground=COLORS["bg_black"], foreground=COLORS["fg_light_grey"], background=COLORS["bg_black"], bordercolor=COLORS["border"], lightcolor=COLORS["border"], darkcolor=COLORS["border"], arrowcolor=COLORS["scroll_thumb"])
        style.map("TCombobox", fieldbackground=[("readonly", COLORS["bg_black"])], foreground=[("readonly", COLORS["fg_light_grey"])], background=[("active", COLORS["bg_black"])])
        # Combobox dropdown list popup
        style.configure("TCombobox.Spinbox", background=COLORS["bg_black"], foreground=COLORS["fg_light_grey"])
        self.root.option_add("*TCombobox*Listbox.background", COLORS["bg_black"])
        self.root.option_add("*TCombobox*Listbox.foreground", COLORS["fg_light_grey"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", COLORS["bg_medium"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", COLORS["fg_text"])

        # Scrollbars: black trough, light muted blue thumb + arrows
        style.configure("TScrollbar", background=COLORS["scroll_thumb"], troughcolor=COLORS["scroll_bg"], bordercolor=COLORS["scroll_bg"], arrowcolor=COLORS["scroll_thumb"], lightcolor=COLORS["scroll_bg"], darkcolor=COLORS["scroll_bg"])
        style.map("TScrollbar", background=[("active", COLORS["scroll_thumb"]), ("pressed", COLORS["accent_blue"])], arrowcolor=[("active", COLORS["scroll_thumb"])])
        style.configure("Vertical.TScrollbar", background=COLORS["scroll_thumb"], troughcolor=COLORS["scroll_bg"], arrowcolor=COLORS["scroll_thumb"])
        style.configure("Horizontal.TScrollbar", background=COLORS["scroll_thumb"], troughcolor=COLORS["scroll_bg"], arrowcolor=COLORS["scroll_thumb"])
    #endregion

    #region U-SETUP
    # UI construction: layout, toolbar, node list, canvas table, details, logging
    def _setup_ui(self) -> None:
        self.root.title("illumorae Node Metadata Dashboard")
        self._apply_dark_theme()

        main_frame = tk.Frame(self.root, bg=COLORS["bg_app"], padx=10, pady=10)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=4)  # Main content gets 4/5
        main_frame.rowconfigure(2, weight=1)  # Logging gets 1/5

        self._create_toolbar(main_frame)

        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        self._create_node_list(left)
        self._create_details(right)

        # Logging area at bottom (1/5th of window)
        self._create_logging_area(main_frame)

    def _create_toolbar(self, parent: tk.Widget) -> None:
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        bar.columnconfigure(6, weight=1)

        ttk.Label(bar, text="Project Root:").grid(row=0, column=0, sticky=tk.W)
        self.root_var = tk.StringVar(value=self.project_root)
        ttk.Entry(bar, textvariable=self.root_var, width=60).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(bar, text="Browse", command=self._browse_root).grid(row=0, column=2, sticky=tk.W)

        self.issues_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Issues Only", variable=self.issues_only, command=self.apply_filters).grid(
            row=0, column=3, sticky=tk.W, padx=(10, 0)
        )

        ttk.Label(bar, text="Group:").grid(row=0, column=4, sticky=tk.W, padx=(10, 0))
        self.group_filter = ttk.Combobox(bar, width=18, state="readonly")
        self.group_filter["values"] = ["All"]
        self.group_filter.current(0)
        self.group_filter.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())
        self.group_filter.grid(row=0, column=5, sticky=tk.W, padx=5)

        ttk.Button(bar, text="Refresh", command=self.load_nodes).grid(row=0, column=7, sticky=tk.E, padx=(10, 0))
        ttk.Button(bar, text="View All Imports", command=self.show_imports_summary).grid(row=0, column=8, sticky=tk.E, padx=5)
        ttk.Button(bar, text="Export JSON Reports", command=self.export_json_reports).grid(row=0, column=9, sticky=tk.E, padx=5)

    def _create_node_list(self, parent: tk.Widget) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(header, text="Nodes", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        self.count_label = ttk.Label(header, text="(0)")
        self.count_label.pack(side=tk.LEFT, padx=5)
        
        # Canvas-based table for per-cell color control
        self._create_canvas_table(parent)

    def _create_canvas_table(self, parent: tk.Widget) -> None:
        """Create custom canvas-based table with per-cell color control"""
        table_frame = ttk.Frame(parent)
        table_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        # Scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical")
        hsb = ttk.Scrollbar(table_frame, orient="horizontal")

        # Canvas for custom rendering
        self.table_canvas = tk.Canvas(
            table_frame,
            bg=COLORS["bg_app"],
            highlightthickness=0,
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )

        vsb.config(command=self.table_canvas.yview)
        hsb.config(command=self.table_canvas.xview)

        self.table_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Table configuration
        self.table_data = []
        self.selected_row = None
        self.row_height = 24
        self.sort_column = None
        self.sort_reverse = False
        self.button_widgets = []  # embedded tk.Button widgets (cleaned up on redraw)
        self.columns = [
            {"key": "category", "label": "Category", "width": 100, "align": "left"},
            {"key": "group", "label": "Group", "width": 100, "align": "left"},
            {"key": "node_id", "label": "Node ID", "width": 200, "align": "left"},
            {"key": "display", "label": "Display Name", "width": 180, "align": "left"},
            {"key": "grp", "label": "G", "width": 30, "align": "center", "status": True},
            {"key": "title", "label": "T", "width": 30, "align": "center", "status": True},
            {"key": "ver", "label": "V", "width": 30, "align": "center", "status": True},
            {"key": "desc", "label": "D", "width": 30, "align": "center", "status": True},
            {"key": "img", "label": "I", "width": 30, "align": "center", "status": True},
            {"key": "map", "label": "M", "width": 30, "align": "center", "status": True},
            {"key": "dispmap", "label": "DM", "width": 30, "align": "center", "status": True},
            {"key": "nodeid", "label": "ID", "width": 30, "align": "center", "status": True},
            {"key": "nodeid_fmt", "label": "NF", "width": 30, "align": "center", "status": True},
            {"key": "pyfile_fmt", "label": "PF", "width": 30, "align": "center", "status": True},
            {"key": "folder", "label": "FD", "width": 30, "align": "center", "status": True},
            {"key": "pyfile", "label": "PY", "width": 30, "align": "center", "status": True},
            {"key": "class", "label": "CL", "width": 30, "align": "center", "status": True},
            {"key": "issues", "label": "Issues", "width": 50, "align": "center"},
            {"key": "update_ver", "label": "[refresh]V", "width": 35, "align": "center", "button": True},
        ]

        # Bind mouse events
        self.table_canvas.bind("<Button-1>", self._on_canvas_click)
        self.table_canvas.bind("<Configure>", self._on_canvas_configure)
        self.table_canvas.bind("<Motion>", self._on_canvas_motion)
        
        # Tooltip for status cells
        self.tooltip = None
        self.tooltip_window = None

        # Draw header
        self._draw_table_header()

        # Legend with color indicators
        legend_frame = ttk.Frame(parent)
        legend_frame.grid(row=2, column=0, sticky=tk.W, pady=(3, 0))

        legend_text = "Legend: G=GROUP | T=TITLE | V=VERSION | D=DESC | I=IMAGE | M=ClassMappings | DM=DisplayMappings | ID=NodeID ends 'Node' | NF=NodeID format | PF=PyFile format | FD=Folder | PY=PyFile | CL=Class | \u21bb=Update Version | "
        ttk.Label(legend_frame, text=legend_text, font=("TkDefaultFont", 8), foreground=COLORS["fg_dim"]).pack(side=tk.LEFT)

        # Color-coded status indicators (small colored canvas squares)
        for color, label_text in [
            (COLORS["accent_green"], "Pass "),
            (COLORS["accent_yellow"], "Warn "),
            (COLORS["accent_red"], "Fail"),
        ]:
            sq = tk.Canvas(legend_frame, width=12, height=12, bg=COLORS["bg_app"], highlightthickness=0)
            sq.create_rectangle(1, 1, 11, 11, fill=color, outline=color)
            sq.pack(side=tk.LEFT, padx=(4, 2))
            ttk.Label(legend_frame, text=label_text, font=("TkDefaultFont", 8), foreground=COLORS["fg_dim"]).pack(side=tk.LEFT)

    def _create_details(self, parent: tk.Widget) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        # Scrollable details frame
        details_canvas = tk.Canvas(parent, bg=COLORS["bg_app"], highlightthickness=0)
        details_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=details_canvas.yview)
        
        self.details_frame = ttk.Frame(details_canvas)
        self.details_frame.columnconfigure(1, weight=1)
        
        details_canvas.create_window((0, 0), window=self.details_frame, anchor="nw")
        details_canvas.configure(yscrollcommand=details_scrollbar.set)
        
        details_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Update scroll region when frame changes
        self.details_frame.bind("<Configure>", lambda e: details_canvas.configure(scrollregion=details_canvas.bbox("all")))
        
        # Store references for editable fields
        self.edit_fields = {}

    def _create_logging_area(self, parent: tk.Widget) -> None:
        """Create logging/status area at bottom 1/5th of window."""
        log_frame = ttk.Frame(parent)
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)

        ttk.Label(log_frame, text="Status & Logging", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 3)
        )

        self.log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            bg=COLORS["bg_black"],
            fg=COLORS["fg_dim"],
            insertbackground=COLORS["fg_text"],
            height=6,
        )
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.configure(state="disabled")

        # Initialize with ready message
        self._log("Dashboard initialized. Ready to scan nodes.")
    #endregion

    #region U-TABLE
    # Canvas table rendering: header, rows with per-cell colors, button cells
    def _draw_table_header(self) -> None:
        """Draw table header row with sort arrow on active column"""
        self.table_canvas.delete("header")
        x = 0
        y = 0
        for col in self.columns:
            is_sorted = (self.sort_column == col["key"])
            header_bg = COLORS["bg_medium"] if is_sorted else COLORS["bg_dark"]
            # Header background
            self.table_canvas.create_rectangle(
                x, y, x + col["width"], y + self.row_height,
                fill=header_bg, outline=COLORS["border"], tags="header"
            )
            # Header text (append arrow if sorted)
            label = col["label"]
            if is_sorted:
                label = label + (" \u25BC" if self.sort_reverse else " \u25B2")
            text_x = x + col["width"] // 2 if col.get("align") == "center" else x + 5
            anchor = "center" if col.get("align") == "center" else "w"
            self.table_canvas.create_text(
                text_x, y + self.row_height // 2,
                text=label, fill=COLORS["fg_text"],
                anchor=anchor, font=("TkDefaultFont", 9, "bold"), tags="header"
            )
            x += col["width"]

    def _draw_table_rows(self) -> None:
        """Draw all table rows with per-cell coloring"""
        self.table_canvas.delete("row")  # Clear existing rows

        # Compute max issues count for color-coding the Issues column
        max_issues = 0
        for rd in self.table_data:
            try:
                max_issues = max(max_issues, int(rd.get("issues", 0)))
            except (TypeError, ValueError):
                pass

        y = self.row_height  # Start after header
        for row_idx, row_data in enumerate(self.table_data):
            x = 0
            bg_color = COLORS["bg_light"] if row_idx == self.selected_row else COLORS["bg_app"]
            
            for col in self.columns:
                cell_value = row_data.get(col["key"], "")
                
                # Cell background
                self.table_canvas.create_rectangle(
                    x, y, x + col["width"], y + self.row_height,
                    fill=bg_color, outline=COLORS["border"], tags="row"
                )
                
                # Cell content: colored square for status, text for regular cells
                if col.get("status") and isinstance(cell_value, dict):
                    # Status cell: draw a colored square centered in the cell
                    color = cell_value.get("color", COLORS["fg_text"])
                    icon = cell_value.get("icon", "")
                    sq_size = 14
                    cx = x + col["width"] // 2
                    cy = y + self.row_height // 2
                    if icon == ".":
                        # Neutral/info: hollow square (dim outline only)
                        self.table_canvas.create_rectangle(
                            cx - sq_size // 2, cy - sq_size // 2,
                            cx + sq_size // 2, cy + sq_size // 2,
                            fill=bg_color, outline=color, width=1, tags="row"
                        )
                    else:
                        # Pass/warning/error: filled colored square
                        self.table_canvas.create_rectangle(
                            cx - sq_size // 2, cy - sq_size // 2,
                            cx + sq_size // 2, cy + sq_size // 2,
                            fill=color, outline=color, width=1, tags="row"
                        )
                else:
                    # Regular text cell
                    text_x = x + col["width"] // 2 if col.get("align") == "center" else x + 5
                    anchor = "center" if col.get("align") == "center" else "w"
                    # Issues column: bold, gray if zero, bright red if highest
                    if col["key"] == "issues":
                        try:
                            issues_val = int(cell_value)
                        except (TypeError, ValueError):
                            issues_val = 0
                        if issues_val == 0:
                            issue_color = COLORS["fg_grey"]
                        elif max_issues > 0 and issues_val == max_issues:
                            issue_color = COLORS["accent_red_bright"]
                        else:
                            issue_color = COLORS["fg_text"]
                        self.table_canvas.create_text(
                            text_x, y + self.row_height // 2,
                            text=str(cell_value), fill=issue_color,
                            anchor=anchor, font=("TkDefaultFont", 9, "bold"), tags="row"
                        )
                    else:
                        self.table_canvas.create_text(
                            text_x, y + self.row_height // 2,
                            text=str(cell_value), fill=COLORS["fg_text"],
                            anchor=anchor, font=("TkDefaultFont", 9), tags="row"
                        )
                
                x += col["width"]
            
            y += self.row_height
        
        # Draw button column backgrounds
        self._draw_button_cells()
        
        # Update scroll region
        total_width = sum(col["width"] for col in self.columns)
        total_height = self.row_height * (len(self.table_data) + 1)
        self.table_canvas.configure(scrollregion=(0, 0, total_width, total_height))

    def _draw_button_cells(self) -> None:
        """Embed real tk.Button widgets in the update_ver column via create_window."""
        # Destroy previous button widgets (from prior redraw)
        for btn in self.button_widgets:
            try:
                btn.destroy()
            except Exception:
                pass
        self.button_widgets = []

        # Find the update_ver column position
        x = 0
        button_col_x = 0
        button_col_width = 35
        for col in self.columns:
            if col["key"] == "update_ver":
                button_col_x = x
                button_col_width = col["width"]
                break
            x += col["width"]

        y = self.row_height  # Start after header
        for row_idx in range(len(self.table_data)):
            btn = tk.Button(
                self.table_canvas,
                text="\u21bb",  # clockwise refresh arrow
                bg=COLORS["accent_blue"], fg=COLORS["fg_text"],
                activebackground=COLORS["bg_light"],
                activeforeground=COLORS["fg_text"],
                relief="flat", borderwidth=0,
                font=("TkDefaultFont", 10, "bold"),
                command=lambda r=row_idx: self._update_node_version(r),
            )
            # Embed button in canvas at cell position
            self.table_canvas.create_window(
                button_col_x + button_col_width // 2, y + self.row_height // 2,
                window=btn, anchor="center",
                width=button_col_width - 4, height=self.row_height - 4,
                tags=("row", f"btn_{row_idx}"),
            )
            self.button_widgets.append(btn)
            y += self.row_height
    #endregion

    #region U-EVT
    # Canvas mouse events: click selection, configure, motion tooltips
    def _on_canvas_click(self, event: Any) -> None:
        """Handle mouse click on canvas table (header = sort, row = select)"""
        canvas_x = self.table_canvas.canvasx(event.x)
        canvas_y = self.table_canvas.canvasy(event.y)

        # Header click: sort by that column
        if canvas_y < self.row_height:
            x = 0
            for col in self.columns:
                if x <= canvas_x < x + col["width"]:
                    self._sort_by_column(col["key"])
                    return
                x += col["width"]
            return

        # Row click: select row (button column has its own tk.Button handling)
        row_idx = int((canvas_y - self.row_height) // self.row_height)

        if 0 <= row_idx < len(self.table_data):
            self.selected_row = row_idx
            self._draw_table_rows()

            # Trigger details view update
            row_data = self.table_data[row_idx]
            node_id = row_data.get("node_id", "")
            for n in self.filtered_nodes:
                if n.node_class_mappings and node_id in n.node_class_mappings:
                    self.selected_node = n
                    self._render_details(n)
                    break

    def _sort_by_column(self, col_key: str) -> None:
        """Sort table_data and filtered_nodes in tandem by the given column key.
        Clicking the same column again toggles ascending/descending."""
        if not self.table_data:
            return

        # Toggle direction if same column, else default ascending
        if self.sort_column == col_key:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = col_key
            self.sort_reverse = False

        self._apply_sort()

    def _apply_sort(self) -> None:
        """Re-order table_data and filtered_nodes by self.sort_column/self.sort_reverse."""
        if not self.sort_column or not self.table_data:
            self._draw_table_header()
            self._draw_table_rows()
            return

        col_key = self.sort_column

        # Build sort key extractor for this column type
        def extract(row: dict):
            val = row.get(col_key, "")
            # Status cells are dicts with icon/color -> sort by ok-ness then severity
            if isinstance(val, dict):
                r = self._lookup_rule_for_cell(col_key, row)
                if r is None:
                    return (2, "")
                ok = r.get("ok", False)
                sev = r.get("severity", "info")
                # ascending: pass(True) first, then warning, then error
                sev_rank = {"info": 0, "warning": 1, "error": 2}.get(sev, 3)
                return (0 if ok else 1, sev_rank)
            # Numeric columns
            if isinstance(val, (int, float)):
                return (0, val)
            # String columns: case-insensitive
            return (0, str(val).lower())

        # Pair, sort, unpair (keep table_data and filtered_nodes in sync)
        paired = list(zip(self.table_data, self.filtered_nodes))
        paired.sort(key=lambda pair: extract(pair[0]), reverse=self.sort_reverse)
        self.table_data = [p[0] for p in paired]
        self.filtered_nodes = [p[1] for p in paired]

        # Clear selection (row index may have changed) and redraw
        self.selected_row = None
        self._draw_table_header()
        self._draw_table_rows()

    def _lookup_rule_for_cell(self, col_key: str, row: dict) -> Optional[dict]:
        """Map a status-cell column key to the corresponding rule_checks entry
        for the node represented by this row."""
        node_id = row.get("node_id", "")
        # Find the node for this row
        node = None
        for n in self.filtered_nodes:
            if n.node_class_mappings and node_id in n.node_class_mappings:
                node = n
                break
        if not node:
            return None
        rule_map = {
            "grp": "frontmatter_group",
            "title": "frontmatter_title",
            "ver": "frontmatter_version",
            "desc": "frontmatter_descriptionshort",
            "img": "frontmatter_image",
            "map": "has_class_mappings",
            "dispmap": f"display_mapping_exists::{node_id}",
            "nodeid": f"node_id_ends_with_node::{node_id}",
            "nodeid_fmt": f"node_id_matches_display::{node_id}",
            "pyfile_fmt": "py_file_matches_display",
            "folder": "folder_name",
            "pyfile": "main_py_file",
            "class": f"class_name_convention::{node.node_class_mappings.get(node_id, '')}",
        }
        rule_id = rule_map.get(col_key)
        if not rule_id:
            return None
        return node.rule_checks.get(rule_id)

    def _on_canvas_configure(self, event: Any) -> None:
        """Handle canvas resize"""
        self._draw_table_rows()

    def _on_canvas_motion(self, event: Any) -> None:
        """Handle mouse motion for tooltips"""
        canvas_x = self.table_canvas.canvasx(event.x)
        canvas_y = self.table_canvas.canvasy(event.y)
        
        # Check if hovering over a status cell
        row_idx = int((canvas_y - self.row_height) // self.row_height)
        
        if 0 <= row_idx < len(self.table_data):
            x = 0
            for col_idx, col in enumerate(self.columns):
                if x <= canvas_x < x + col["width"]:
                    if col.get("status"):
                        # Show tooltip for status cell
                        row_data = self.table_data[row_idx]
                        node = self.filtered_nodes[row_idx]
                        self._show_tooltip(event, col["key"], node)
                        return
                x += col["width"]
        
        self._hide_tooltip()

    def _show_tooltip(self, event: Any, col_key: str, node: NodeMetadata) -> None:
        """Show tooltip with rule check details"""
        # Map column keys to rule IDs
        rule_map = {
            "grp": "frontmatter_group",
            "title": "frontmatter_title",
            "ver": "frontmatter_version",
            "desc": "frontmatter_descriptionshort",
            "img": "frontmatter_image",
            "map": "has_class_mappings",
            "dispmap": "display_mapping_exists",
            "nodeid": "node_id_ends_with_node",
            "nodeid_fmt": "node_id_matches_display",
            "pyfile_fmt": "py_file_matches_display",
            "folder": "folder_name",
            "pyfile": "main_py_file",
            "class": "class_name_convention",
        }
        
        rule_id = rule_map.get(col_key)
        if not rule_id:
            return
        
        # For node-specific rules, find the specific rule for this node
        node_id = next(iter(node.node_class_mappings.keys()), "") if node.node_class_mappings else ""
        class_name = node.node_class_mappings.get(node_id, "") if node_id else ""
        
        if rule_id == "node_id_ends_with_node":
            rule_id = f"node_id_ends_with_node::{node_id}"
        elif rule_id == "node_id_matches_display":
            rule_id = f"node_id_matches_display::{node_id}"
        elif rule_id == "display_mapping_exists":
            rule_id = f"display_mapping_exists::{node_id}"
        elif rule_id == "class_name_convention":
            rule_id = f"class_name_convention::{class_name}"
        
        if rule_id not in node.rule_checks:
            return
        
        rule = node.rule_checks[rule_id]
        tooltip_text = f"{rule_id}\n{rule.get('details', '')}\nSeverity: {rule.get('severity', 'info')}"
        
        # Create or update tooltip window
        if self.tooltip_window:
            self.tooltip_window.destroy()
        
        self.tooltip_window = tk.Toplevel(self.root)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
        
        label = tk.Label(
            self.tooltip_window,
            text=tooltip_text,
            background=COLORS["bg_light"],
            foreground=COLORS["fg_text"],
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=4,
            font=("TkDefaultFont", 9)
        )
        label.pack()

    def _hide_tooltip(self) -> None:
        """Hide tooltip window"""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
    #endregion

    #region U-LOAD
    # Data loading: scan, filter, populate canvas table rows
    def load_nodes(self) -> None:
        self._log("Scanning nodes...")
        self.root.update()

        self.nodes = self.scanner.scan()

        groups = {"All"}
        for n in self.nodes:
            if n.suggested_group:
                groups.add(n.suggested_group)

        self.group_filter["values"] = sorted(groups)
        if self.group_filter.get() not in self.group_filter["values"]:
            self.group_filter.current(0)

        self.apply_filters()

        errors = sum(1 for n in self.nodes for i in n.issues if i.severity == "error")
        warnings = sum(1 for n in self.nodes for i in n.issues if i.severity == "warning")
        self._log(f"[OK] Loaded {len(self.nodes)} nodes | Errors: {errors} | Warnings: {warnings}")

    def apply_filters(self) -> None:
        group = self.group_filter.get() if self.group_filter.get() else "All"
        issues_only = self.issues_only.get()

        out: List[NodeMetadata] = []
        for n in self.nodes:
            if group != "All" and (n.suggested_group or "") != group:
                continue
            if issues_only and n.issues_count() == 0:
                continue
            out.append(n)

        self.filtered_nodes = out
        self._populate_tree()

    def _populate_tree(self) -> None:
        """Populate canvas table with node data"""
        self.table_data = []

        for n in self.filtered_nodes:
            # Get group from frontmatter, not auto-derived
            group = n.python_frontmatter.get("GROUP", "")
            
            # Get category from class CATEGORY variable
            node_id = next(iter(n.node_class_mappings.keys()), "") if n.node_class_mappings else ""
            class_name = n.node_class_mappings.get(node_id, "") if node_id else ""
            category = ""
            if class_name and class_name in n.class_metadata:
                category = str(n.class_metadata[class_name].get("CATEGORY", ""))
            if not category:
                category = n.category or ""
            
            display = n.node_display_name_mappings.get(node_id, "") if node_id else ""
            if not display:
                display = n.suggested_display_name or ""

            # Helper to create status cell with icon and color
            def status_cell(rule_id: str) -> dict:
                """Returns dict with icon and color for status cell"""
                if rule_id not in n.rule_checks:
                    return {"icon": "?", "color": COLORS["fg_dim"]}
                r = n.rule_checks[rule_id]
                if r.get("ok"):
                    return {"icon": "#", "color": COLORS["accent_green"]}
                sev = r.get("severity", "info")
                if sev == "error":
                    return {"icon": "#", "color": COLORS["accent_red"]}
                elif sev == "warning":
                    return {"icon": "#", "color": COLORS["accent_yellow"]}
                return {"icon": ".", "color": COLORS["fg_dim"]}

            # Display mapping exists check
            dispmap_rule = f"display_mapping_exists::{node_id}"
            if dispmap_rule in n.rule_checks:
                dispmap_cell = status_cell(dispmap_rule)
            else:
                dispmap_cell = {"icon": ".", "color": COLORS["fg_dim"]}
            
            # Class name convention check
            class_rule = f"class_name_convention::{class_name}"
            if class_rule in n.rule_checks:
                class_cell = status_cell(class_rule)
            else:
                class_cell = {"icon": ".", "color": COLORS["fg_dim"]}

            # NodeID ends with 'Node' check
            nodeid_rule = f"node_id_ends_with_node::{node_id}"
            if nodeid_rule in n.rule_checks:
                nodeid_cell = status_cell(nodeid_rule)
            else:
                nodeid_cell = {"icon": ".", "color": COLORS["fg_dim"]}
            
            # NodeID format matches display name check
            nodeid_fmt_rule = f"node_id_matches_display::{node_id}"
            if nodeid_fmt_rule in n.rule_checks:
                nodeid_fmt_cell = status_cell(nodeid_fmt_rule)
            else:
                nodeid_fmt_cell = {"icon": ".", "color": COLORS["fg_dim"]}
            
            # Python file format matches display name check
            pyfile_fmt_cell = status_cell("py_file_matches_display")

            # Build row data
            row = {
                "category": category,
                "group": group,
                "node_id": node_id,
                "display": display,
                "grp": status_cell("frontmatter_group"),
                "title": status_cell("frontmatter_title"),
                "ver": status_cell("frontmatter_version"),
                "desc": status_cell("frontmatter_descriptionshort"),
                "img": status_cell("frontmatter_image"),
                "map": status_cell("has_class_mappings"),
                "dispmap": dispmap_cell,
                "nodeid": nodeid_cell,
                "nodeid_fmt": nodeid_fmt_cell,
                "pyfile_fmt": pyfile_fmt_cell,
                "folder": status_cell("folder_name"),
                "pyfile": status_cell("main_py_file"),
                "class": class_cell,
                "issues": n.issues_count(),
            }
            self.table_data.append(row)

        self.count_label.config(text=f"({len(self.filtered_nodes)})")

        # Re-apply current sort if any (preserves column/direction across refreshes)
        if self.sort_column:
            self._apply_sort()
        else:
            self._draw_table_header()
            self._draw_table_rows()
    #endregion

    #region U-DETAIL
    # Details panel: editable frontmatter fields + read-only info sections
    def _render_details(self, node: NodeMetadata) -> None:
        """Render editable details panel for selected node"""
        # Clear existing widgets
        for widget in self.details_frame.winfo_children():
            widget.destroy()
        
        self.edit_fields = {}
        row = 0
        
        # Header
        ttk.Label(self.details_frame, text="Node Details", font=("TkDefaultFont", 11, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 10)
        )
        row += 1
        
        # Basic info (read-only)
        ttk.Label(self.details_frame, text="Folder:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        ttk.Label(self.details_frame, text=node.folder_name, foreground=COLORS["fg_dim"]).grid(
            row=row, column=1, sticky=tk.W, pady=2
        )
        row += 1
        
        ttk.Label(self.details_frame, text="Path:", font=("TkDefaultFont", 9, "bold")).grid(
            row=row, column=0, sticky=tk.W, pady=2
        )
        ttk.Label(self.details_frame, text=node.folder_path, foreground=COLORS["fg_dim"]).grid(
            row=row, column=1, sticky=tk.W, pady=2
        )
        row += 1
        
        ttk.Separator(self.details_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )
        row += 1
        
        # Editable frontmatter fields
        ttk.Label(self.details_frame, text="Python Frontmatter (Editable)", font=("TkDefaultFont", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 5)
        )
        row += 1
        
        # TITLE field
        ttk.Label(self.details_frame, text="TITLE::").grid(row=row, column=0, sticky=tk.W, pady=2)
        title_var = tk.StringVar(value=node.python_frontmatter.get("TITLE", ""))
        title_entry = ttk.Entry(self.details_frame, textvariable=title_var, width=40)
        title_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.edit_fields["TITLE"] = title_var
        row += 1
        
        # VERSION field
        ttk.Label(self.details_frame, text="VERSION::").grid(row=row, column=0, sticky=tk.W, pady=2)
        version_var = tk.StringVar(value=node.python_frontmatter.get("VERSION", ""))
        version_entry = ttk.Entry(self.details_frame, textvariable=version_var, width=40)
        version_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.edit_fields["VERSION"] = version_var
        row += 1
        
        # DESCRIPTIONSHORT field
        ttk.Label(self.details_frame, text="DESCRIPTIONSHORT::").grid(row=row, column=0, sticky=tk.W, pady=2)
        desc_var = tk.StringVar(value=node.python_frontmatter.get("DESCRIPTIONSHORT", ""))
        desc_entry = ttk.Entry(self.details_frame, textvariable=desc_var, width=40)
        desc_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.edit_fields["DESCRIPTIONSHORT"] = desc_var
        row += 1
        
        # IMAGE field
        ttk.Label(self.details_frame, text="IMAGE::").grid(row=row, column=0, sticky=tk.W, pady=2)
        image_var = tk.StringVar(value=node.python_frontmatter.get("IMAGE", ""))
        image_entry = ttk.Entry(self.details_frame, textvariable=image_var, width=40)
        image_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.edit_fields["IMAGE"] = image_var
        row += 1
        
        # GROUP field
        ttk.Label(self.details_frame, text="GROUP::").grid(row=row, column=0, sticky=tk.W, pady=2)
        group_var = tk.StringVar(value=node.python_frontmatter.get("GROUP", ""))
        group_entry = ttk.Entry(self.details_frame, textvariable=group_var, width=40)
        group_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        self.edit_fields["GROUP"] = group_var
        row += 1
        
        # Save button
        ttk.Button(
            self.details_frame,
            text="Save Changes",
            command=lambda: self._save_frontmatter(node)
        ).grid(row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        ttk.Separator(self.details_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10
        )
        row += 1
        
        # Read-only info sections
        self._add_readonly_section(node, row)

    def _add_readonly_section(self, node: NodeMetadata, start_row: int) -> None:
        """Add read-only information sections"""
        row = start_row
        
        lines: List[str] = []

        lines.append("=== Python ===")
        lines.append(f"Main py: {node.main_py_file or ''}")
        lines.append(f"CATEGORY: {node.category or ''}")
        lines.append(f"FUNCTION: {node.function_name or ''}")
        lines.append(f"Classes: {', '.join(node.classes_defined) if node.classes_defined else ''}")
        lines.append("")

        lines.append("NODE_CLASS_MAPPINGS:")
        for k, v in node.node_class_mappings.items():
            lines.append(f"  {k} -> {v}")
        lines.append("")

        lines.append("NODE_DISPLAY_NAME_MAPPINGS:")
        for k, v in node.node_display_name_mappings.items():
            lines.append(f"  {k} -> {v}")
        lines.append("")

        lines.append("=== Class Metadata (CATEGORY/FUNCTION/DESCRIPTION) ===")
        if node.class_metadata:
            for cname in sorted(node.class_metadata.keys()):
                meta = node.class_metadata[cname]
                lines.append(f"  {cname}:")
                for k in ("CATEGORY", "FUNCTION", "DESCRIPTION"):
                    if k in meta:
                        lines.append(f"    {k}: {repr(meta[k])}")
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("=== Obsidian Fields ===")
        if node.obsidian_fields:
            for k in sorted(node.obsidian_fields.keys()):
                lines.append(f"  {k}: {repr(node.obsidian_fields[k])}")
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("=== Imports (Security Check) ===")
        if node.imports:
            dangerous = ["subprocess", "requests", "urllib", "http", "socket", "eval", "exec"]
            for imp in node.imports:
                if imp in dangerous:
                    lines.append(f"  [DANGEROUS] {imp}")
                else:
                    lines.append(f"  {imp}")
        else:
            lines.append("  (none)")
        lines.append("")
        if node.dangerous_imports:
            lines.append(f"(!) WARNING: {len(node.dangerous_imports)} dangerous import(s) detected!")
            lines.append("")

        lines.append("=== Suggested Naming ===")
        lines.append(f"Group: {node.suggested_group or ''}")
        lines.append(f"Folder: {node.suggested_folder_name or ''}")
        lines.append(f"Py file: {node.suggested_py_file or ''}")
        lines.append(f"Class: {node.suggested_class_name or ''}")
        lines.append(f"Node ID: {node.suggested_node_id or ''}")
        lines.append(f"Display: {node.suggested_display_name or ''}")
        lines.append("")

        lines.append("=== Rule Checks (hover over status squares for details) ===")
        if node.rule_checks:
            for rid in sorted(node.rule_checks.keys()):
                r = node.rule_checks[rid]
                mark = "PASS" if r.get("ok") else "FAIL"
                sev = r.get('severity', 'info').upper()
                lines.append(f"[{mark}] [{sev}] {rid}: {r.get('details','')}")
        else:
            lines.append("(none)")
        lines.append("")

        lines.append("=== Issues ===")
        if node.issues:
            for i in node.issues:
                lines.append(f"[{i.severity.upper()}] {i.message}")
        else:
            lines.append("No issues.")

        # Display read-only info in text widget
        info_text = tk.Text(
            self.details_frame,
            wrap=tk.WORD,
            bg=COLORS["bg_black"],
            fg=COLORS["fg_text"],
            height=20,
            width=50
        )
        info_text.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        info_text.insert("1.0", "\n".join(lines))
        info_text.configure(state="disabled")
    #endregion

    #region U-ACT
    # Button actions: browse, export reports, update version, save, imports summary
    def _browse_root(self) -> None:
        path = filedialog.askdirectory(initialdir=self.project_root)
        if path:
            self.root_var.set(path)
            self.project_root = os.path.abspath(path)
            self.scanner = NodeMetadataScanner(self.project_root)
            self.load_nodes()

    def export_json_reports(self) -> None:
        """Export detailed JSON reports for each node to their respective folders"""
        try:
            exported_count = 0
            failed_count = 0
            
            for node in self.nodes:
                try:
                    # Build comprehensive report
                    report = {
                        "node_info": {
                            "folder_name": node.folder_name,
                            "folder_path": node.folder_path,
                            "folder_suffix": node.folder_suffix,
                            "main_py_file": node.main_py_file,
                            "main_py_path": node.main_py_path,
                            "category": node.category,
                        },
                        "class_info": {
                            "classes_defined": node.classes_defined,
                            "node_class_mappings": node.node_class_mappings,
                            "node_display_name_mappings": node.node_display_name_mappings,
                        },
                        "metadata": {
                            "python_frontmatter": node.python_frontmatter,
                            "obsidian_fields": node.obsidian_fields,
                            "global_fields": node.global_fields,
                            "class_metadata": node.class_metadata,
                        },
                        "suggestions": {
                            "suggested_folder_name": node.suggested_folder_name,
                            "suggested_py_file": node.suggested_py_file,
                            "suggested_class_name": node.suggested_class_name,
                            "suggested_node_id": node.suggested_node_id,
                            "suggested_display_name": node.suggested_display_name,
                            "suggested_group": node.suggested_group,
                        },
                        "rule_checks": {},
                        "summary": {
                            "total_checks": 0,
                            "passed": 0,
                            "warnings": 0,
                            "errors": 0,
                            "failed_checks": [],
                        }
                    }
                    
                    # Process all rule checks
                    for rule_id, check_data in node.rule_checks.items():
                        report["rule_checks"][rule_id] = {
                            "status": "PASS" if check_data["ok"] else "FAIL",
                            "ok": check_data["ok"],
                            "details": check_data["details"],
                            "severity": check_data.get("severity", "info"),
                        }
                        
                        report["summary"]["total_checks"] += 1
                        
                        if check_data["ok"]:
                            report["summary"]["passed"] += 1
                        else:
                            severity = check_data.get("severity", "info")
                            if severity == "warning":
                                report["summary"]["warnings"] += 1
                            elif severity == "error":
                                report["summary"]["errors"] += 1
                            
                            report["summary"]["failed_checks"].append({
                                "rule_id": rule_id,
                                "details": check_data["details"],
                                "severity": severity,
                            })
                    
                    # Write JSON to node folder
                    output_path = os.path.join(node.folder_path, "node_validation_report.json")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
                    
                    exported_count += 1
                    self._log(f"Exported: {node.folder_name} -> {output_path}")
                    
                except Exception as e:
                    failed_count += 1
                    self._log(f"ERROR: Failed to export {node.folder_name}: {e}")
            
            # Show summary message
            summary_msg = f"Export complete: {exported_count} reports generated"
            if failed_count > 0:
                summary_msg += f", {failed_count} failed"
            
            self._log(summary_msg)
            
            # Show popup confirmation
            from tkinter import messagebox
            messagebox.showinfo(
                "Export Complete",
                f"Successfully exported {exported_count} JSON reports.\n"
                f"Failed: {failed_count}\n\n"
                f"Reports saved as 'node_validation_report.json' in each node folder."
            )
            
        except Exception as e:
            self._log(f"ERROR: Export failed: {e}")
            from tkinter import messagebox
            messagebox.showerror("Export Error", f"Failed to export reports:\n{e}")

    def _update_node_version(self, row_idx: int) -> None:
        """Update the VERSION:: frontmatter field to current date YYYYMMDD"""
        if row_idx < 0 or row_idx >= len(self.filtered_nodes):
            return

        node = self.filtered_nodes[row_idx]
        if not node.main_py_path:
            self._log(f"Error: No Python file path for {node.folder_name}")
            return

        try:
            # Get current date as YYYYMMDD
            new_version = datetime.now().strftime("%Y%m%d")

            # Read current file
            with open(node.main_py_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')

            # Find docstring boundaries
            in_docstring = False
            docstring_start = -1
            docstring_end = -1
            quote_style = None

            for i, line in enumerate(lines):
                stripped = line.strip()
                if not in_docstring:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        in_docstring = True
                        docstring_start = i
                        quote_style = '"""' if stripped.startswith('"""') else "'''"
                        if stripped.endswith(quote_style) and len(stripped) > 6:
                            docstring_end = i
                            break
                else:
                    if quote_style in line:
                        docstring_end = i
                        break

            if docstring_start == -1 or docstring_end == -1:
                self._log(f"Error: Could not find module docstring in {node.main_py_file}")
                return

            # Find and update VERSION:: field
            found = False
            for i in range(docstring_start, docstring_end + 1):
                if "VERSION::" in lines[i]:
                    # Preserve any leading whitespace
                    leading_ws = len(lines[i]) - len(lines[i].lstrip())
                    lines[i] = " " * leading_ws + f"VERSION::{new_version}"
                    found = True
                    break

            # If VERSION:: not found, add it before closing quotes
            if not found:
                lines.insert(docstring_end, f"VERSION::{new_version}")

            # Write back to file
            new_content = '\n'.join(lines)
            with open(node.main_py_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            self._log(f"[OK] Updated VERSION::{new_version} in {node.main_py_file}")

            # Reload nodes to reflect changes
            self.load_nodes()

        except Exception as e:
            self._log(f"Error updating version for {node.folder_name}: {str(e)}")

    def _save_frontmatter(self, node: NodeMetadata) -> None:
        """Save edited frontmatter back to node file"""
        if not node.main_py_path:
            self._log("Error: No Python file path for this node")
            return
        
        try:
            # Read current file
            with open(node.main_py_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Find docstring boundaries
            in_docstring = False
            docstring_start = -1
            docstring_end = -1
            quote_style = None
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not in_docstring:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        in_docstring = True
                        docstring_start = i
                        quote_style = '"""' if stripped.startswith('"""') else "'''"
                        if stripped.endswith(quote_style) and len(stripped) > 6:
                            docstring_end = i
                            break
                else:
                    if quote_style in line:
                        docstring_end = i
                        break
            
            if docstring_start == -1 or docstring_end == -1:
                self._log("Error: Could not find module docstring")
                return
            
            # Update frontmatter fields
            for field_name, var in self.edit_fields.items():
                new_value = var.get()
                pattern = f"{field_name}::"
                
                # Find and update the field
                found = False
                for i in range(docstring_start, docstring_end + 1):
                    if pattern in lines[i]:
                        lines[i] = f"{field_name}::{new_value}"
                        found = True
                        break
                
                # If not found, add it before closing quotes
                if not found and new_value:
                    lines.insert(docstring_end, f"{field_name}::{new_value}")
                    docstring_end += 1
            
            # Write back to file
            new_content = '\n'.join(lines)
            with open(node.main_py_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self._log(f"[OK] Saved changes to {node.main_py_file}")
            
            # Reload nodes to reflect changes
            self.load_nodes()
            
        except Exception as e:
            self._log(f"Error saving frontmatter: {str(e)}")

    def show_imports_summary(self) -> None:
        """Show a popup window with all imports across all nodes, highlighting dangerous ones."""
        from tkinter import messagebox
        
        # Collect all imports across all nodes
        all_imports = {}
        dangerous = ["subprocess", "requests", "urllib", "http", "socket", "eval", "exec"]
        
        for node in self.nodes:
            for imp in node.imports:
                if imp not in all_imports:
                    all_imports[imp] = []
                all_imports[imp].append(node.folder_name)
        
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("All Imports Summary - Security Check")
        popup.geometry("800x600")
        popup.configure(bg=COLORS["bg_app"])
        
        # Header
        header_frame = ttk.Frame(popup)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            header_frame,
            text="All Imports Across Nodes",
            font=("TkDefaultFont", 12, "bold")
        ).pack(side=tk.LEFT)
        
        dangerous_count = sum(1 for imp in all_imports.keys() if imp in dangerous)
        if dangerous_count > 0:
            ttk.Label(
                header_frame,
                text=f"(!) {dangerous_count} DANGEROUS",
                font=("TkDefaultFont", 11, "bold"),
                foreground=COLORS["accent_red"]
            ).pack(side=tk.RIGHT)
        
        # Scrollable text area
        text_frame = ttk.Frame(popup)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            bg=COLORS["bg_black"],
            fg=COLORS["fg_text"],
            insertbackground=COLORS["fg_text"],
            yscrollcommand=scrollbar.set,
            font=("Courier", 10)
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Configure text tags for colors
        text_widget.tag_configure("dangerous", foreground=COLORS["accent_red"], font=("Courier", 10, "bold"))
        text_widget.tag_configure("safe", foreground=COLORS["accent_green"])
        text_widget.tag_configure("header", foreground=COLORS["accent_blue"], font=("Courier", 10, "bold"))
        
        # Add content
        text_widget.insert("end", "IMPORT SECURITY ANALYSIS\n", "header")
        text_widget.insert("end", "=" * 80 + "\n\n")
        
        # Sort imports: dangerous first, then alphabetically
        sorted_imports = sorted(all_imports.keys(), key=lambda x: (x not in dangerous, x))
        
        for imp in sorted_imports:
            nodes_using = all_imports[imp]
            is_dangerous = imp in dangerous
            
            if is_dangerous:
                text_widget.insert("end", f"[DANGEROUS] {imp}", "dangerous")
                text_widget.insert("end", f" (used in {len(nodes_using)} node(s))\n")
                text_widget.insert("end", "  (!) SECURITY RISK: Avoid subprocess, requests, and similar imports\n", "dangerous")
            else:
                text_widget.insert("end", f"{imp}", "safe")
                text_widget.insert("end", f" (used in {len(nodes_using)} node(s))\n")
            
            # Show which nodes use this import
            for node_name in sorted(nodes_using)[:5]:  # Show first 5
                text_widget.insert("end", f"    - {node_name}\n")
            if len(nodes_using) > 5:
                text_widget.insert("end", f"    ... and {len(nodes_using) - 5} more\n")
            text_widget.insert("end", "\n")
        
        text_widget.insert("end", "\n" + "=" * 80 + "\n")
        text_widget.insert("end", f"Total unique imports: {len(all_imports)}\n")
        text_widget.insert("end", f"Dangerous imports: {dangerous_count}\n", "dangerous" if dangerous_count > 0 else "safe")
        text_widget.insert("end", f"Safe imports: {len(all_imports) - dangerous_count}\n", "safe")
        
        text_widget.configure(state="disabled")
        
        # Close button
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)
        
        self._log(f"Imports summary: {len(all_imports)} unique imports, {dangerous_count} dangerous")

    def export_json(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        export_report(self.project_root, self.nodes, self.scanner.global_variables, path)
        self._log(f"[OK] Exported JSON: {os.path.basename(path)}")
    #endregion

    #region U-LOG
    # Logging utility: append messages to the status text widget
    def _log(self, message: str) -> None:
        """Add a message to the logging area."""
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
    #endregion
#endregion


#region EXPORT
# Standalone JSON report exporter (used by CLI --export-json)
def export_report(project_root: str, nodes: List[NodeMetadata], globals_map: Dict[str, Any], out_path: str) -> None:
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": os.path.abspath(project_root),
        "nodes": [],
        "globals": globals_map,
    }

    for n in nodes:
        d = asdict(n)
        d["issues_count"] = n.issues_count()
        payload["nodes"].append(d)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
#endregion


#region MAIN
# CLI entry point: argparse, scanner, optional GUI launch
def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("project_root", nargs="?", default=None)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--export-json", default=None)

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(args.project_root or os.path.join(script_dir, ".."))

    scanner = NodeMetadataScanner(project_root)
    nodes = scanner.scan()

    if args.export_json:
        export_report(project_root, nodes, scanner.global_variables, args.export_json)

    if args.no_gui:
        errors = sum(1 for n in nodes for i in n.issues if i.severity == "error")
        warnings = sum(1 for n in nodes for i in n.issues if i.severity == "warning")
        print(f"Scanned {len(nodes)} nodes")
        print(f"Errors: {errors}")
        print(f"Warnings: {warnings}")
        return 0 if errors == 0 else 2

    root = tk.Tk()
    NodeMetadataDashboard(root, project_root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#endregion
