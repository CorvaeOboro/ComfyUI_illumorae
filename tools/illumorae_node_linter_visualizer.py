"""
illumorae Node Linter Visualizer
---------------------------------
Interactive code visualization tool for linter reports with syntax highlighting
and color-coded overlay system.

Features:
- Syntax-highlighted Python code (Obsidian dark theme)
- Multi-layer linter overlays with transparency
- Clickable filter buttons to toggle linter visibility
- Hover tooltips showing detailed linter messages
- Line selection for detailed issue inspection
- Visual indication of issue severity through color intensity
- Support for multiple files and nodes

Usage:
    python illumorae_node_linter_visualizer.py [project_root]
    
    # Or specify a specific report
    python illumorae_node_linter_visualizer.py --report path/to/node_linter_report.json

VERSION: 20260118
"""

#region IMPORTS
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
#endregion IMPORTS

#region CONST
# Obsidian-inspired dark theme colors
COLORS = {
    "bg_black": "#0d0d0d",
    "bg_dark": "#1a1a1a",
    "bg_medium": "#2d2d2d",
    "bg_light": "#3a3a3a",
    "fg_text": "#6b6b6b",      # Medium gray for code text
    "fg_dim": "#8b8b8b",
    "fg_bright": "#ffffff",
    
    # Syntax highlighting (Obsidian-inspired)
    "syntax_keyword": "#c792ea",      # Purple for keywords
    "syntax_string": "#c3e88d",       # Green for strings
    "syntax_comment": "#676e95",      # Gray-blue for comments
    "syntax_function": "#82aaff",     # Blue for functions
    "syntax_class": "#ffcb6b",        # Yellow for classes
    "syntax_number": "#f78c6c",       # Orange for numbers
    "syntax_operator": "#89ddff",     # Cyan for operators
    "syntax_builtin": "#ff5370",      # Red for builtins
    "syntax_decorator": "#c792ea",    # Purple for decorators
    
    # Linter overlay colors (with transparency)
    "linter_black": "#9b59b6",        # Purple
    "linter_ruff": "#3498db",         # Blue
    "linter_flake8": "#e67e22",       # Orange
    "linter_mypy": "#1abc9c",         # Teal
    "linter_pylint": "#e74c3c",       # Red
    
    # Severity colors
    "severity_error": "#ff5370",
    "severity_warning": "#ffcb6b",
    "severity_info": "#82aaff",
    
    "border": "#404040",
    "accent_blue": "#4a7ba7",
}

# Linter color mapping with alpha values for overlays
LINTER_COLORS = {
    "black": {"color": COLORS["linter_black"], "alpha": 0.10},
    "ruff": {"color": COLORS["linter_ruff"], "alpha": 0.10},
    "flake8": {"color": COLORS["linter_flake8"], "alpha": 0.10},
    "mypy": {"color": COLORS["linter_mypy"], "alpha": 0.10},
    "pylint": {"color": COLORS["linter_pylint"], "alpha": 0.10},
}
#endregion CONST

#region MODELS
@dataclass
class LinterIssue:
    """Represents a single linter issue"""
    linter_name: str
    line: Optional[int]
    column: Optional[int]
    code: Optional[str]
    message: str
    severity: str

@dataclass
class FileReport:
    """Represents linter report for a single file"""
    file_path: str
    file_name: str
    node_folder: str
    issues_by_line: Dict[int, List[LinterIssue]]
    all_issues: List[LinterIssue]
    linters_with_issues: Set[str]
#endregion MODELS


#region SYNTAX
class SyntaxHighlighter:
    """Syntax highlighter for Python code"""

    #region S-WORDS
    # Python keywords
    KEYWORDS = {
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
        'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
        'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
        'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
        'try', 'while', 'with', 'yield'
    }
    
    # Python builtins
    BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr',
        'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
        'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr',
        'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
        'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
        'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord',
        'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
        'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
        'super', 'tuple', 'type', 'vars', 'zip', '__import__'
    }
    #endregion S-WORDS

    #region S-TAGS
    def __init__(self, text_widget: tk.Text):
        self.text = text_widget
        self._configure_tags()

    def _configure_tags(self):
        """Configure text widget tags for syntax highlighting"""
        self.text.tag_configure("keyword", foreground=COLORS["syntax_keyword"])
        self.text.tag_configure("string", foreground=COLORS["syntax_string"])
        self.text.tag_configure("comment", foreground=COLORS["syntax_comment"])
        self.text.tag_configure("function", foreground=COLORS["syntax_function"])
        self.text.tag_configure("class", foreground=COLORS["syntax_class"])
        self.text.tag_configure("number", foreground=COLORS["syntax_number"])
        self.text.tag_configure("operator", foreground=COLORS["syntax_operator"])
        self.text.tag_configure("builtin", foreground=COLORS["syntax_builtin"])
        self.text.tag_configure("decorator", foreground=COLORS["syntax_decorator"])
    #endregion S-TAGS

    #region S-HL
    def highlight(self, code: str):
        """Apply syntax highlighting to code"""
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", code)
        
        # Remove all existing tags
        for tag in ["keyword", "string", "comment", "function", "class", 
                    "number", "operator", "builtin", "decorator"]:
            self.text.tag_remove(tag, "1.0", tk.END)
        
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            self._highlight_line(line, line_num)
    
    def _highlight_line(self, line: str, line_num: int):
        """Highlight a single line of code"""
        # Comments
        comment_match = re.search(r'#.*$', line)
        if comment_match:
            start = comment_match.start()
            end = comment_match.end()
            self.text.tag_add("comment", f"{line_num}.{start}", f"{line_num}.{end}")
            # Don't process rest of line if it's a comment
            line = line[:start]
        
        # Strings (triple quotes first, then single/double)
        for pattern in [r'""".*?"""', r"'''.*?'''", r'"(?:[^"\\]|\\.)*"', r"'(?:[^'\\]|\\.)*'"]:
            for match in re.finditer(pattern, line):
                start, end = match.span()
                self.text.tag_add("string", f"{line_num}.{start}", f"{line_num}.{end}")
        
        # Decorators
        for match in re.finditer(r'@\w+', line):
            start, end = match.span()
            self.text.tag_add("decorator", f"{line_num}.{start}", f"{line_num}.{end}")
        
        # Numbers
        for match in re.finditer(r'\b\d+\.?\d*\b', line):
            start, end = match.span()
            self.text.tag_add("number", f"{line_num}.{start}", f"{line_num}.{end}")
        
        # Class definitions
        for match in re.finditer(r'\bclass\s+(\w+)', line):
            start, end = match.span(1)
            self.text.tag_add("class", f"{line_num}.{start}", f"{line_num}.{end}")
        
        # Function definitions
        for match in re.finditer(r'\bdef\s+(\w+)', line):
            start, end = match.span(1)
            self.text.tag_add("function", f"{line_num}.{start}", f"{line_num}.{end}")
        
        # Function calls
        for match in re.finditer(r'\b(\w+)\s*\(', line):
            word = match.group(1)
            if word in self.BUILTINS:
                start, end = match.span(1)
                self.text.tag_add("builtin", f"{line_num}.{start}", f"{line_num}.{end}")
        
        # Keywords and builtins
        for match in re.finditer(r'\b\w+\b', line):
            word = match.group()
            start, end = match.span()
            if word in self.KEYWORDS:
                self.text.tag_add("keyword", f"{line_num}.{start}", f"{line_num}.{end}")
            elif word in self.BUILTINS:
                self.text.tag_add("builtin", f"{line_num}.{start}", f"{line_num}.{end}")
        
        # Operators
        for match in re.finditer(r'[+\-*/%=<>!&|^~]+', line):
            start, end = match.span()
            self.text.tag_add("operator", f"{line_num}.{start}", f"{line_num}.{end}")
    #endregion S-HL
#endregion SYNTAX


#region VIEWER
class LinterVisualizer:
    """Main visualizer GUI application"""

    #region V-INIT
    def __init__(self, root: tk.Tk, project_root: str):
        self.root = root
        self.project_root = os.path.abspath(project_root)
        self.current_file: Optional[FileReport] = None
        self.file_reports: List[FileReport] = []
        self.active_linters: Set[str] = set(LINTER_COLORS.keys())
        self.active_severities: Set[str] = {"error", "warning", "info"}
        self.hidden_issue_types: Set[str] = set()  # Issue codes/patterns to hide
        self.linter_buttons: Dict[str, tk.Button] = {}
        self.severity_buttons: Dict[str, tk.Button] = {}
        self.issue_type_buttons: Dict[str, tk.Button] = {}

        self._setup_ui()
        self._load_reports()
    #endregion V-INIT

    #region V-THEME
    def _apply_dark_theme(self):
        """Apply Obsidian-inspired dark theme"""
        self.root.configure(bg=COLORS["bg_black"])
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure(
            ".",
            background=COLORS["bg_black"],
            foreground=COLORS["fg_text"],
            fieldbackground=COLORS["bg_dark"],
            bordercolor=COLORS["border"],
        )
        
        style.configure("TButton",
            background=COLORS["bg_medium"],
            foreground=COLORS["fg_text"],
            borderwidth=1,
            relief="flat",
            padding=6,
        )
        
        style.map("TButton",
            background=[("active", COLORS["bg_light"])],
            foreground=[("active", COLORS["fg_bright"])]
        )
        
        style.configure("Active.TButton",
            background=COLORS["accent_blue"],
            foreground=COLORS["fg_bright"],
        )
        
        # Combobox with pure black background and white text
        style.configure("TCombobox",
            fieldbackground="#000000",  # Pure black
            background="#000000",  # Pure black
            foreground="#ffffff",  # White text
            arrowcolor="#82aaff",  # Blue arrow
            selectbackground="#82aaff",  # Blue selection
            selectforeground="#ffffff",  # White selected text
            borderwidth=1,
            relief="flat"
        )
        
        style.map("TCombobox",
            fieldbackground=[("readonly", "#000000")],
            selectbackground=[("readonly", "#000000")],
            foreground=[("readonly", "#ffffff")]
        )
        
        # Custom scrollbar styling - pure black background with blue thumb
        style.configure("Custom.Vertical.TScrollbar",
            background="#82aaff",  # Blue thumb
            troughcolor="#000000",  # Pure black trough
            bordercolor="#000000",
            arrowcolor="#82aaff",  # Blue arrows
            darkcolor="#82aaff",
            lightcolor="#82aaff",
            relief="flat",
            borderwidth=0
        )
        
        style.map("Custom.Vertical.TScrollbar",
            background=[("active", "#ffffff"), ("!active", "#82aaff")],  # White when active, blue otherwise
            arrowcolor=[("active", "#ffffff"), ("!active", "#82aaff")]
        )
        
        style.configure("Custom.Horizontal.TScrollbar",
            background="#82aaff",  # Blue thumb
            troughcolor="#000000",  # Pure black trough
            bordercolor="#000000",
            arrowcolor="#82aaff",  # Blue arrows
            darkcolor="#82aaff",
            lightcolor="#82aaff",
            relief="flat",
            borderwidth=0
        )
        
        style.map("Custom.Horizontal.TScrollbar",
            background=[("active", "#ffffff"), ("!active", "#82aaff")],  # White when active, blue otherwise
            arrowcolor=[("active", "#ffffff"), ("!active", "#82aaff")]
        )
    #endregion V-THEME

    #region V-UI
    def _setup_ui(self):
        """Setup main UI layout"""
        self.root.title("illumorae Linter Visualizer")
        self.root.geometry("1400x900")
        self._apply_dark_theme()
        
        # Main container
        main_frame = tk.Frame(self.root, bg=COLORS["bg_black"], padx=10, pady=10)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        self._create_toolbar(main_frame)
        self._create_linter_filters(main_frame)
        self._create_code_viewer(main_frame)
        self._create_status_bar(main_frame)
    
    def _create_toolbar(self, parent: tk.Widget):
        """Create top toolbar with file selection"""
        toolbar = tk.Frame(parent, bg=COLORS["bg_dark"], pady=8, padx=8)
        toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        toolbar.columnconfigure(1, weight=1)
        
        tk.Label(
            toolbar,
            text="File:",
            bg=COLORS["bg_dark"],
            fg=COLORS["fg_text"],
            font=("Segoe UI", 9)
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.file_combo = ttk.Combobox(
            toolbar,
            state="readonly",
            width=80,
            font=("Consolas", 9)
        )
        self.file_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.file_combo.bind("<<ComboboxSelected>>", self._on_file_selected)
        
        ttk.Button(
            toolbar,
            text="Load Report...",
            command=self._load_report_file
        ).grid(row=0, column=2, sticky=tk.W, padx=5)
        
        ttk.Button(
            toolbar,
            text="Refresh",
            command=self._load_reports
        ).grid(row=0, column=3, sticky=tk.W, padx=5)
    
    def _create_linter_filters(self, parent: tk.Widget):
        """Create linter and severity filter buttons"""
        filter_frame = tk.Frame(parent, bg=COLORS["bg_dark"], pady=8, padx=8)
        filter_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Linter filters
        tk.Label(
            filter_frame,
            text="Linter Overlays:",
            bg=COLORS["bg_dark"],
            fg=COLORS["fg_text"],
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 15))
        
        for linter_name, linter_info in LINTER_COLORS.items():
            btn = tk.Button(
                filter_frame,
                text=linter_name,
                bg=linter_info["color"],
                fg=COLORS["fg_bright"],
                activebackground=linter_info["color"],
                activeforeground=COLORS["fg_bright"],
                font=("Segoe UI", 9, "bold"),
                relief="raised",
                borderwidth=2,
                padx=12,
                pady=4,
                command=lambda ln=linter_name: self._toggle_linter(ln)
            )
            btn.pack(side=tk.LEFT, padx=5)
            self.linter_buttons[linter_name] = btn
        
        # Separator
        tk.Frame(filter_frame, bg=COLORS["border"], width=2, height=30).pack(side=tk.LEFT, padx=10)
        
        # Severity filters
        tk.Label(
            filter_frame,
            text="Severity:",
            bg=COLORS["bg_dark"],
            fg=COLORS["fg_text"],
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        for severity, color in [("error", COLORS["severity_error"]), 
                                ("warning", COLORS["severity_warning"]), 
                                ("info", COLORS["severity_info"])]:
            btn = tk.Button(
                filter_frame,
                text=severity.capitalize(),
                bg=color,
                fg=COLORS["fg_bright"],
                activebackground=color,
                activeforeground=COLORS["fg_bright"],
                font=("Segoe UI", 9, "bold"),
                relief="raised",
                borderwidth=2,
                padx=12,
                pady=4,
                command=lambda s=severity: self._toggle_severity(s)
            )
            btn.pack(side=tk.LEFT, padx=3)
            self.severity_buttons[severity] = btn
        
        # Separator
        tk.Frame(filter_frame, bg=COLORS["border"], width=2, height=30).pack(side=tk.LEFT, padx=10)
        
        # Issue type filters (on same row)
        tk.Label(
            filter_frame,
            text="Hide:",
            bg=COLORS["bg_dark"],
            fg=COLORS["fg_text"],
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 8))
        
        # Define common issue types to filter
        issue_types = [
            ("line-too-long", "Line Length", ["E501", "C0301", "line-too-long"]),
            ("trailing-whitespace", "Whitespace", ["W291", "W293", "C0303", "trailing-whitespace"]),
            ("missing-docstring", "Docstrings", ["C0114", "C0115", "C0116", "D100", "D101", "D102", "D103"]),
            ("unused-import", "Unused Imports", ["F401", "unused-import"]),
            ("no-type-annotation", "Type Hints", ["no-untyped-def", "var-annotated", "ANN"]),
        ]
        
        for issue_id, display_name, codes in issue_types:
            btn = tk.Button(
                filter_frame,
                text=display_name,
                bg=COLORS["bg_medium"],
                fg=COLORS["fg_text"],
                activebackground=COLORS["bg_light"],
                activeforeground=COLORS["fg_bright"],
                font=("Segoe UI", 8),
                relief="raised",
                borderwidth=1,
                padx=6,
                pady=3,
                command=lambda iid=issue_id, c=codes: self._toggle_issue_type(iid, c)
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.issue_type_buttons[issue_id] = btn

    def _create_code_viewer(self, parent: tk.Widget):
        """Create main code viewer with canvas overlay"""
        viewer_frame = tk.Frame(parent, bg=COLORS["bg_black"])
        viewer_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        viewer_frame.columnconfigure(1, weight=1)
        viewer_frame.rowconfigure(0, weight=1)
        
        # Line numbers
        self.line_numbers = tk.Text(
            viewer_frame,
            width=5,
            bg=COLORS["bg_dark"],
            fg=COLORS["fg_dim"],
            font=("Consolas", 10),
            state="disabled",
            relief="flat",
            borderwidth=0,
            padx=5,
            pady=5,
            cursor="arrow"
        )
        self.line_numbers.grid(row=0, column=0, sticky=(tk.N, tk.S))
        
        # Code text widget
        code_container = tk.Frame(viewer_frame, bg=COLORS["bg_black"])
        code_container.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        code_container.columnconfigure(0, weight=1)
        code_container.rowconfigure(0, weight=1)
        
        # Scrollbars with custom ttk style
        v_scroll = ttk.Scrollbar(
            viewer_frame, 
            orient="vertical",
            style="Custom.Vertical.TScrollbar"
        )
        v_scroll.grid(row=0, column=2, sticky=(tk.N, tk.S))
        
        h_scroll = ttk.Scrollbar(
            code_container, 
            orient="horizontal",
            style="Custom.Horizontal.TScrollbar"
        )
        h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Code text widget
        self.code_text = tk.Text(
            code_container,
            bg=COLORS["bg_black"],
            fg=COLORS["fg_text"],
            font=("Consolas", 10),
            insertbackground=COLORS["fg_bright"],
            selectbackground=COLORS["bg_light"],
            selectforeground=COLORS["fg_bright"],
            wrap="none",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=5,
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
            state="disabled"
        )
        # Canvas overlay for drawing colored borders (behind text)
        self.border_canvas = tk.Canvas(
            code_container,
            bg=COLORS["bg_black"],
            highlightthickness=0,
            borderwidth=0
        )
        self.border_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.code_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Make canvas pass-through for mouse events by raising text widget
        self.code_text.lift()
        
        v_scroll.config(command=self._on_scroll)
        h_scroll.config(command=self.code_text.xview)
        
        # Bind text widget events
        self.code_text.bind("<Motion>", self._on_mouse_motion)
        self.code_text.bind("<<Selection>>", self._on_selection_changed)
        self.code_text.bind("<Configure>", lambda e: self._draw_border_overlays())
        
        # Tooltip
        self.tooltip = None
        self.tooltip_window = None
        
        # Initialize syntax highlighter
        self.highlighter = SyntaxHighlighter(self.code_text)
        
        # Configure linter highlight tags
        # Background tags for 10% opacity color fill
        for linter_name, linter_info in LINTER_COLORS.items():
            color = linter_info["color"]
            alpha = linter_info["alpha"]
            bg_color = self._apply_alpha(color, alpha)
            
            self.code_text.tag_configure(
                f"linter_bg_{linter_name}",
                background=bg_color
            )
        
        # No border tags needed - borders will be drawn on canvas
        
        # Ensure syntax highlighting tags have higher priority than linter tags
        self.code_text.tag_raise("keyword")
        self.code_text.tag_raise("string")
        self.code_text.tag_raise("comment")
        self.code_text.tag_raise("number")
        self.code_text.tag_raise("function")
        self.code_text.tag_raise("decorator")
        
        # Details panel
        self._create_details_panel(parent)
    
    def _create_details_panel(self, parent: tk.Widget):
        """Create details panel for selected line issues"""
        details_frame = tk.Frame(parent, bg=COLORS["bg_dark"], pady=8, padx=8)
        details_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        details_frame.columnconfigure(0, weight=1)
        
        tk.Label(
            details_frame,
            text="Issue Details (click a line to see details)",
            bg=COLORS["bg_dark"],
            fg=COLORS["fg_text"],
            font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        details_scroll = ttk.Scrollbar(
            details_frame, 
            orient="vertical",
            style="Custom.Vertical.TScrollbar"
        )
        details_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        self.details_text = tk.Text(
            details_frame,
            bg=COLORS["bg_black"],
            fg=COLORS["fg_text"],
            font=("Consolas", 9),
            height=6,
            wrap="word",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=5,
            state="disabled",
            yscrollcommand=details_scroll.set
        )
        self.details_text.grid(row=1, column=0, sticky=(tk.W, tk.E))
        details_scroll.config(command=self.details_text.yview)
        
        # Configure tags for details
        self.details_text.tag_configure("linter", foreground=COLORS["accent_blue"], font=("Consolas", 9, "bold"))
        self.details_text.tag_configure("code", foreground=COLORS["syntax_number"], font=("Consolas", 9, "bold"))
        self.details_text.tag_configure("error", foreground=COLORS["severity_error"])
        self.details_text.tag_configure("warning", foreground=COLORS["severity_warning"])
        self.details_text.tag_configure("info", foreground=COLORS["severity_info"])
    
    def _create_status_bar(self, parent: tk.Widget):
        """Create bottom status bar"""
        status_frame = tk.Frame(parent, bg=COLORS["bg_dark"], pady=4, padx=8)
        status_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            bg=COLORS["bg_dark"],
            fg=COLORS["fg_dim"],
            font=("Segoe UI", 8),
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.stats_label = tk.Label(
            status_frame,
            text="",
            bg=COLORS["bg_dark"],
            fg=COLORS["fg_dim"],
            font=("Segoe UI", 8),
            anchor=tk.E
        )
        self.stats_label.pack(side=tk.RIGHT)
    #endregion V-UI

    #region V-LOAD
    def _load_reports(self):
        """Load all linter reports from project"""
        self.file_reports.clear()
        
        try:
            for root_dir, dirs, files in os.walk(self.project_root):
                if "node_linter_report.json" in files:
                    report_path = os.path.join(root_dir, "node_linter_report.json")
                    self._load_single_report(report_path)
            
            # Update file combo
            file_list = [f"{fr.node_folder} / {fr.file_name}" for fr in self.file_reports]
            self.file_combo['values'] = file_list
            
            if file_list:
                self.file_combo.current(0)
                self._on_file_selected(None)
                
                # Check if any reports have line numbers
                total_issues_with_lines = sum(
                    len(fr.issues_by_line) for fr in self.file_reports
                )
                if total_issues_with_lines == 0:
                    self.status_label.config(
                        text=f"Loaded {len(self.file_reports)} file(s) - WARNING: No line numbers found. Re-run linters to generate new reports."
                    )
                else:
                    self.status_label.config(text=f"Loaded {len(self.file_reports)} file(s) from reports")
            else:
                self.status_label.config(text="No linter reports found. Run linter tool first.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load reports:\n{e}")
    
    def _load_single_report(self, report_path: str):
        """Load a single linter report JSON file"""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            node_folder = report_data.get("node_folder", "Unknown")
            file_results = report_data.get("file_results", {})
            
            for file_name, linter_results in file_results.items():
                issues_by_line: Dict[int, List[LinterIssue]] = {}
                all_issues: List[LinterIssue] = []
                linters_with_issues: Set[str] = set()
                
                for linter_result in linter_results:
                    linter_name = linter_result.get("linter_name", "unknown")
                    issues = linter_result.get("issues", [])
                    
                    for issue_data in issues:
                        line = issue_data.get("line")
                        if line is None:
                            continue
                        
                        issue = LinterIssue(
                            linter_name=linter_name,
                            line=line,
                            column=issue_data.get("column"),
                            code=issue_data.get("code"),
                            message=issue_data.get("message", ""),
                            severity=issue_data.get("severity", "info")
                        )
                        
                        all_issues.append(issue)
                        linters_with_issues.add(linter_name)
                        
                        if line not in issues_by_line:
                            issues_by_line[line] = []
                        issues_by_line[line].append(issue)
                
                # Get file path from the first result that has one
                file_path = ""
                for result in linter_results:
                    fp = result.get("file_path", "")
                    if fp:
                        file_path = fp
                        break
                
                file_report = FileReport(
                    file_path=file_path,
                    file_name=file_name,
                    node_folder=node_folder,
                    issues_by_line=issues_by_line,
                    all_issues=all_issues,
                    linters_with_issues=linters_with_issues
                )
                
                self.file_reports.append(file_report)
                
        except Exception as e:
            print(f"Error loading report {report_path}: {e}")
    
    def _load_report_file(self):
        """Load a specific report file via file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Linter Report",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self.project_root
        )
        
        if file_path:
            self.file_reports.clear()
            self._load_single_report(file_path)
            
            file_list = [f"{fr.node_folder} / {fr.file_name}" for fr in self.file_reports]
            self.file_combo['values'] = file_list
            
            if file_list:
                self.file_combo.current(0)
                self._on_file_selected(None)
    #endregion V-LOAD

    #region V-DISP
    def _on_file_selected(self, event):
        """Handle file selection from combo box"""
        idx = self.file_combo.current()
        if idx >= 0 and idx < len(self.file_reports):
            self.current_file = self.file_reports[idx]
            self._display_file()
    
    def _display_file(self):
        """Display the currently selected file with syntax highlighting and overlays"""
        if not self.current_file:
            return
        
        # Load source code
        try:
            with open(self.current_file.file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                code = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            return
        
        # Enable text widget for editing
        self.code_text.config(state="normal")
        
        # Apply syntax highlighting
        self.highlighter.highlight(code)
        
        # Update line numbers
        line_count = int(self.code_text.index('end-1c').split('.')[0])
        self._update_line_numbers(line_count)
        
        # Disable text widget
        self.code_text.config(state="disabled")
        
        # Draw overlays
        self._draw_overlays()
        
        # Update stats
        total_issues = len(self.current_file.all_issues)
        linters = ", ".join(sorted(self.current_file.linters_with_issues))
        self.stats_label.config(
            text=f"{total_issues} issue(s) | Linters: {linters}"
        )
        self.status_label.config(
            text=f"Viewing: {self.current_file.node_folder} / {self.current_file.file_name}"
        )
    
    def _update_line_numbers(self, line_count: int):
        """Update line numbers display"""
        self.line_numbers.config(state="normal")
        self.line_numbers.delete("1.0", tk.END)
        
        line_nums = "\n".join(str(i) for i in range(1, line_count + 1))
        self.line_numbers.insert("1.0", line_nums)
        self.line_numbers.config(state="disabled")
    #endregion V-DISP

    #region V-OVER
    def _filter_issues(self, issues: List[LinterIssue]) -> List[LinterIssue]:
        """Filter issues by active linters, severities, and hidden issue types"""
        return [i for i in issues
                if i.linter_name in self.active_linters
                and i.severity in self.active_severities
                and not self._is_issue_hidden(i)]

    def _draw_overlays(self):
        """Draw linter overlays on visible lines using text tags"""
        if not self.current_file:
            return
        
        # Remove all linter background tags
        for linter_name in LINTER_COLORS.keys():
            self.code_text.tag_remove(f"linter_bg_{linter_name}", "1.0", "end")
        
        # Apply linter background tags to lines with issues
        for line_num, issues in self.current_file.issues_by_line.items():
            active_issues = self._filter_issues(issues)
            if not active_issues:
                continue

            start_idx = f"{line_num}.0"
            end_idx = f"{line_num}.end"
            
            # Apply all background color fills (10% opacity)
            # These stack to create overlapping translucent colors
            for issue in active_issues:
                if issue.linter_name not in LINTER_COLORS:
                    continue
                self.code_text.tag_add(f"linter_bg_{issue.linter_name}", start_idx, end_idx)
        
        # Draw borders on canvas
        self._draw_border_overlays()
    
    def _draw_border_overlays(self):
        """Draw bright colored borders on canvas overlay"""
        if not self.current_file:
            return
        
        # Clear existing borders
        self.border_canvas.delete("all")
        
        # Get text widget dimensions
        try:
            # Apply borders to lines with issues
            for line_num, issues in self.current_file.issues_by_line.items():
                active_issues = self._filter_issues(issues)
                if not active_issues:
                    continue
                
                # Get line position in text widget
                try:
                    bbox = self.code_text.bbox(f"{line_num}.0")
                    if not bbox:
                        continue
                    
                    x, y, width, height = bbox
                    
                    # Get full line width
                    line_end_bbox = self.code_text.bbox(f"{line_num}.end")
                    if line_end_bbox:
                        line_width = line_end_bbox[0] - x
                    else:
                        line_width = self.code_text.winfo_width() - x - 20
                    
                    # Draw borders for each linter, stacking outward
                    for idx, issue in enumerate(active_issues):
                        if issue.linter_name not in LINTER_COLORS:
                            continue
                        
                        color = LINTER_COLORS[issue.linter_name]["color"]
                        offset = idx  # 0, 1, 2, 3, 4 pixels
                        
                        # Draw rectangle border with offset
                        self.border_canvas.create_rectangle(
                            x - offset,
                            y - offset,
                            x + line_width + offset,
                            y + height + offset,
                            outline=color,
                            width=1,
                            fill=""
                        )
                except tk.TclError:
                    continue
        except Exception as e:
            pass
    #endregion V-OVER

    #region V-EVT
    def _on_scroll(self, *args):
        """Handle scrolling - sync line numbers and redraw borders"""
        self.code_text.yview(*args)
        self.line_numbers.yview(*args)
        # Redraw borders after scrolling
        self.root.after(10, self._draw_border_overlays)
    
    def _apply_alpha(self, hex_color: str, alpha: float) -> str:
        """Apply alpha transparency to a color by blending with background"""
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # Background color (pure black #0d0d0d)
        bg_r, bg_g, bg_b = 13, 13, 13
        
        # Blend colors based on alpha (10% color, 90% background)
        r = int(r * alpha + bg_r * (1 - alpha))
        g = int(g * alpha + bg_g * (1 - alpha))
        b = int(b * alpha + bg_b * (1 - alpha))
        
        return f'#{r:02x}{g:02x}{b:02x}'

    def _on_mouse_motion(self, event):
        """Handle mouse motion for tooltips"""
        if not self.current_file:
            return
        
        # Get line number at mouse position
        x = self.code_text.winfo_rootx() + event.x
        y = self.code_text.winfo_rooty() + event.y
        
        try:
            idx = self.code_text.index(f"@{event.x},{event.y}")
            line_num = int(idx.split('.')[0])
            
            if line_num in self.current_file.issues_by_line:
                issues = self._filter_issues(self.current_file.issues_by_line[line_num])
                if issues:
                    self._show_tooltip(x, y, issues)
                    return
        except Exception:
            pass
        
        self._hide_tooltip()
    
    def _show_tooltip(self, x: int, y: int, issues: List[LinterIssue]):
        """Show tooltip with issue details"""
        self._hide_tooltip()
        
        self.tooltip_window = tk.Toplevel(self.root)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x+10}+{y+10}")
        
        frame = tk.Frame(
            self.tooltip_window,
            bg=COLORS["bg_medium"],
            borderwidth=1,
            relief="solid"
        )
        frame.pack()
        
        for issue in issues[:5]:  # Limit to 5 issues
            linter_color = LINTER_COLORS.get(issue.linter_name, {}).get("color", COLORS["fg_text"])
            
            issue_frame = tk.Frame(frame, bg=COLORS["bg_medium"], pady=2, padx=8)
            issue_frame.pack(fill=tk.X)
            
            tk.Label(
                issue_frame,
                text=f"[{issue.linter_name}]",
                bg=COLORS["bg_medium"],
                fg=linter_color,
                font=("Consolas", 8, "bold")
            ).pack(side=tk.LEFT, padx=(0, 5))
            
            if issue.code:
                tk.Label(
                    issue_frame,
                    text=issue.code,
                    bg=COLORS["bg_medium"],
                    fg=COLORS["syntax_number"],
                    font=("Consolas", 8, "bold")
                ).pack(side=tk.LEFT, padx=(0, 5))
            
            tk.Label(
                issue_frame,
                text=issue.message[:80] + ("..." if len(issue.message) > 80 else ""),
                bg=COLORS["bg_medium"],
                fg=COLORS["fg_text"],
                font=("Consolas", 8)
            ).pack(side=tk.LEFT)
        
        if len(issues) > 5:
            tk.Label(
                frame,
                text=f"... and {len(issues) - 5} more",
                bg=COLORS["bg_medium"],
                fg=COLORS["fg_dim"],
                font=("Consolas", 8, "italic"),
                padx=8,
                pady=2
            ).pack()
    
    def _hide_tooltip(self):
        """Hide tooltip"""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
    
    def _on_selection_changed(self, event):
        """Handle text selection - show details for selected line"""
        try:
            # Get current cursor position
            idx = self.code_text.index(tk.INSERT)
            line_num = int(idx.split('.')[0])
            
            if self.current_file and line_num in self.current_file.issues_by_line:
                issues = self._filter_issues(self.current_file.issues_by_line[line_num])
                self._show_details(line_num, issues)
            else:
                self._clear_details()
        except Exception:
            pass
    
    def _show_details(self, line_num: int, issues: List[LinterIssue]):
        """Show detailed issue information in details panel"""
        self.details_text.config(state="normal")
        self.details_text.delete("1.0", tk.END)
        
        self.details_text.insert(tk.END, f"Line {line_num} Issues:\n\n", "linter")
        
        for idx, issue in enumerate(issues, 1):
            linter_color = LINTER_COLORS.get(issue.linter_name, {}).get("color", COLORS["fg_text"])
            
            self.details_text.insert(tk.END, f"{idx}. ", "")
            self.details_text.insert(tk.END, f"[{issue.linter_name}] ", "linter")
            
            if issue.code:
                self.details_text.insert(tk.END, f"{issue.code} ", "code")
            
            self.details_text.insert(tk.END, f"({issue.severity})\n", issue.severity)
            self.details_text.insert(tk.END, f"   {issue.message}\n\n", "")
        
        self.details_text.config(state="disabled")
    
    def _clear_details(self):
        """Clear details panel"""
        self.details_text.config(state="normal")
        self.details_text.delete("1.0", tk.END)
        self.details_text.insert(tk.END, "Click on a line with issues to see details")
        self.details_text.config(state="disabled")
    #endregion V-EVT

    #region V-TOG
    def _toggle_linter(self, linter_name: str):
        """Toggle linter visibility"""
        if linter_name in self.active_linters:
            self.active_linters.remove(linter_name)
            self.linter_buttons[linter_name].config(relief="sunken", bg=COLORS["bg_medium"])
        else:
            self.active_linters.add(linter_name)
            color = LINTER_COLORS[linter_name]["color"]
            self.linter_buttons[linter_name].config(relief="raised", bg=color)
        
        self._draw_overlays()
    
    def _toggle_severity(self, severity: str):
        """Toggle severity visibility"""
        if severity in self.active_severities:
            self.active_severities.remove(severity)
            self.severity_buttons[severity].config(relief="sunken", bg=COLORS["bg_medium"])
        else:
            self.active_severities.add(severity)
            severity_colors = {
                "error": COLORS["severity_error"],
                "warning": COLORS["severity_warning"],
                "info": COLORS["severity_info"]
            }
            color = severity_colors[severity]
            self.severity_buttons[severity].config(relief="raised", bg=color)
        
        self._draw_overlays()
    
    def _toggle_issue_type(self, issue_id: str, codes: List[str]):
        """Toggle hiding specific issue types by their codes"""
        # Check if any of these codes are currently hidden
        codes_set = set(codes)
        is_hidden = any(code in self.hidden_issue_types for code in codes)
        
        if is_hidden:
            # Show these issue types
            self.hidden_issue_types -= codes_set
            self.issue_type_buttons[issue_id].config(relief="raised", bg=COLORS["bg_medium"])
        else:
            # Hide these issue types
            self.hidden_issue_types.update(codes_set)
            self.issue_type_buttons[issue_id].config(relief="sunken", bg=COLORS["bg_dark"])
        
        self._draw_overlays()
    
    def _is_issue_hidden(self, issue: LinterIssue) -> bool:
        """Check if an issue should be hidden based on its code"""
        if issue.code and issue.code in self.hidden_issue_types:
            return True
        return False
    #endregion V-TOG
#endregion VIEWER


#region MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Visualize linter reports with syntax highlighting and overlay system"
    )
    parser.add_argument(
        'project_root',
        nargs='?',
        default=None,
        help='Project root directory to scan for reports'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Specific report file to load'
    )
    
    args = parser.parse_args()
    
    # Determine project root
    if args.project_root:
        project_root = args.project_root
    else:
        # Default to parent directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
    
    if not os.path.exists(project_root):
        print(f"ERROR: Project root '{project_root}' does not exist")
        sys.exit(1)
    
    try:
        root = tk.Tk()
        app = LinterVisualizer(root, project_root)

        # Load specific report if provided
        if args.report and os.path.exists(args.report):
            app._load_single_report(args.report)
            file_list = [f"{fr.node_folder} / {fr.file_name}" for fr in app.file_reports]
            app.file_combo['values'] = file_list
            if file_list:
                app.file_combo.current(0)
                app._on_file_selected(None)

        root.mainloop()
    except tk.TclError as e:
        print(f"ERROR: Cannot launch GUI ({e}).")
        print("Hint: This tool requires a display. Run on a system with a GUI.")
        sys.exit(1)


if __name__ == "__main__":
    main()
#endregion MAIN
