"""
illumorae Node Linter Runner
-----------------------------
Scans ComfyUI_illumorae_* node packages in this repository and runs multiple linters
(Ruff, Flake8, Black, Mypy, Pylint) on each node's Python files.

Generates per-node linter reports in JSON format (node_linter_report.json) similar to
node_validation_report.json, containing:
- Linter outputs and issue counts
- Severity classifications
- Actionable fix suggestions
- Summary statistics

Features:
- Interactive tkinter GUI with node visualization
- Real-time status updates during linting
- Color-coded status indicators (Green=Pass, Yellow=Warnings, Red=Errors)
- Per-node and consolidated reports
- Configurable linter selection

Usage:
    python illumorae_node_linter_runner.py [project_root]

Optional flags:
    --no-gui                                  (run in CLI mode without GUI)
    --linters black,ruff,flake8,mypy,pylint  (comma-separated list, default: all available)
    --skip-clean                              (skip nodes with no issues)
    --export-summary <path>                   (export consolidated summary JSON)

VERSION: 20260117
"""

#region IMPORTS
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
#endregion IMPORTS

#region CONST
COLORS = {
    "bg_dark": "#1a1a1a",
    "bg_medium": "#2d2d2d",
    "bg_light": "#3a3a3a",
    "bg_black": "#000000",
    "fg_text": "#ffffff",
    "fg_dim": "#b0b0b0",
    "accent_blue": "#4a7ba7",
    "accent_green": "#5a8a5a",
    "accent_red": "#a75a5a",
    "accent_yellow": "#a79a5a",
    "border": "#404040",
}
#endregion CONST

#region MODELS
@dataclass
class LinterIssue:
    """Individual linter issue/violation"""
    line: Optional[int] = None
    column: Optional[int] = None
    code: Optional[str] = None
    message: str = ""
    severity: str = "info"
    
@dataclass
class LinterResult:
    """Result from running a single linter on a file"""
    linter_name: str
    file_path: str
    return_code: int
    raw_output: str = ""
    error_message: Optional[str] = None
    issues: List[LinterIssue] = field(default_factory=list)
    issue_count: int = 0
    status: str = "pass"
    
@dataclass
class NodeLinterReport:
    """Complete linter report for a single node package"""
    node_folder: str
    node_id: Optional[str] = None
    display_name: Optional[str] = None
    scan_date: str = ""
    python_files: List[str] = field(default_factory=list)
    linters_run: List[str] = field(default_factory=list)
    linters_available: List[str] = field(default_factory=list)
    linters_skipped: List[str] = field(default_factory=list)
    file_results: Dict[str, List[LinterResult]] = field(default_factory=dict)
    total_issues: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    total_info: int = 0
    overall_status: str = "pending"
    fix_suggestions: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
#endregion MODELS


#region LINTER
class LinterRunner:
    """Runs multiple linters and parses their output"""

    #region C-CONFIG
    LINTER_CONFIGS = {
        'black': {
            'cmd': [sys.executable, '-m', 'black', '--check', '--diff'],
            'severity': 'info',
            'description': 'Code formatter - checks formatting consistency'
        },
        'ruff': {
            'cmd': [sys.executable, '-m', 'ruff', 'check'],
            'severity': 'warning',
            'description': 'Fast Python linter - checks code quality and style'
        },
        'flake8': {
            'cmd': [sys.executable, '-m', 'flake8'],
            'severity': 'warning',
            'description': 'PEP 8 style guide checker'
        },
        'mypy': {
            'cmd': [sys.executable, '-m', 'mypy', '--strict'],
            'severity': 'info',
            'description': 'Static type checker'
        },
        'pylint': {
            'cmd': [sys.executable, '-m', 'pylint'],
            'severity': 'warning',
            'description': 'Comprehensive code analysis'
        }
    }
    #endregion C-CONFIG

    #region C-INIT
    def __init__(self, linters: Optional[List[str]] = None, log_callback=None):
        self.available_linters: Set[str] = set()
        self.missing_linters: Set[str] = set()
        self.requested_linters = linters or list(self.LINTER_CONFIGS.keys())
        self.log_callback = log_callback
        
        # Default issue codes to exclude from reports (line too long)
        self.excluded_codes = {"E501", "C0301"}  # flake8 and pylint line-too-long
        
        self._check_availability()
        
    def _log(self, message: str):
        """Log message via callback or print"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
        
    def _check_availability(self) -> None:
        """Check which requested linters are installed"""
        self._log("Checking linter availability...")
        for linter in self.requested_linters:
            if linter not in self.LINTER_CONFIGS:
                self._log(f"  [SKIP] {linter:10} - Unknown linter")
                continue
                
            try:
                result = subprocess.run(
                    [sys.executable, '-m', linter, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.available_linters.add(linter)
                    version = result.stdout.strip().split('\n')[0]
                    self._log(f"  [OK]   {linter:10} - {version}")
                else:
                    self.missing_linters.add(linter)
                    self._log(f"  [MISS] {linter:10} - Not available")
            except Exception:
                self.missing_linters.add(linter)
                self._log(f"  [MISS] {linter:10} - Not available")
        
        if self.missing_linters:
            self._log(f"\nNote: {len(self.missing_linters)} linter(s) not installed: {', '.join(sorted(self.missing_linters))}")
            self._log("Install with: pip install " + " ".join(sorted(self.missing_linters)))
        
        if not self.available_linters:
            self._log("\nERROR: No linters available!")
            return
            
        self._log(f"Running with {len(self.available_linters)} linter(s): {', '.join(sorted(self.available_linters))}\n")
    #endregion C-INIT

    #region C-RUN
    def run_linter(self, linter_name: str, file_path: str) -> LinterResult:
        """Run a single linter on a file and parse results"""
        if linter_name not in self.available_linters:
            return LinterResult(
                linter_name=linter_name,
                file_path=file_path,
                return_code=-1,
                status="skipped",
                error_message=f"{linter_name} not available"
            )
        
        config = self.LINTER_CONFIGS[linter_name]
        cmd = config['cmd'] + [file_path]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace invalid UTF-8 chars instead of crashing
                timeout=60
            )
            
            linter_result = LinterResult(
                linter_name=linter_name,
                file_path=file_path,
                return_code=result.returncode,
                raw_output=result.stdout or result.stderr or "",
                status="pass" if result.returncode == 0 else "fail"
            )
            
            linter_result.issues = self._parse_output(linter_name, linter_result.raw_output)
            linter_result.issue_count = len(linter_result.issues)
            
            return linter_result
            
        except subprocess.TimeoutExpired:
            return LinterResult(
                linter_name=linter_name,
                file_path=file_path,
                return_code=-1,
                status="error",
                error_message="Timeout after 60 seconds"
            )
        except Exception as e:
            return LinterResult(
                linter_name=linter_name,
                file_path=file_path,
                return_code=-1,
                status="error",
                error_message=str(e)
            )
    #endregion C-RUN

    #region C-PARSE
    def _parse_output(self, linter_name: str, output: str) -> List[LinterIssue]:
        """Parse linter output to extract structured issues"""
        issues = []
        
        if not output or not output.strip():
            return issues
        
        # Black outputs a diff format, not line-by-line issues
        if linter_name == 'black':
            return self._parse_black_diff(output)
        
        # Ruff has a special visual format with arrows and context
        if linter_name == 'ruff':
            return self._parse_ruff_output(output)
        
        # Mypy has a special format with error/note/warning prefixes
        if linter_name == 'mypy':
            return self._parse_mypy_output(output)
        
        # Standard line-by-line parsing for other linters
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            issue = self._parse_line(linter_name, line)
            if issue:
                issues.append(issue)
        
        return issues
    
    def _parse_black_diff(self, diff_output: str) -> List[LinterIssue]:
        """Parse Black's diff output to extract affected line ranges"""
        issues = []
        
        # Parse unified diff format to find changed lines
        # Format: @@ -start,count +start,count @@
        for line in diff_output.split('\n'):
            # Find hunk headers that show line ranges
            match = re.match(r'^@@ -(\d+),?\d* \+(\d+),?\d* @@', line)
            if match:
                line_num = int(match.group(2))  # Use the "new" line number
                issues.append(LinterIssue(
                    line=line_num,
                    column=None,
                    code="BLACK",
                    message="Formatting changes suggested (run 'black' to auto-fix)",
                    severity="info"
                ))
        
        # If no hunks found but there's output, file needs formatting
        if not issues and diff_output.strip():
            issues.append(LinterIssue(
                line=1,
                column=None,
                code="BLACK",
                message="File would be reformatted by Black",
                severity="info"
            ))
        
        return issues
    
    def _parse_ruff_output(self, output: str) -> List[LinterIssue]:
        """Parse Ruff's default output format.

        Ruff (v0.1+) outputs one issue per line:
            file.py:line:col: CODE [*] message
        The ``[*]`` marker indicates an autofixable issue and is optional.
        """
        issues = []

        # Format: file.py:line:col: CODE [*] message
        pattern = re.compile(
            r'^.+?:(\d+):(\d+):\s*([A-Z]\d+)\s+(?:\[\*\]\s+)?(.+)$'
        )

        for raw_line in output.split('\n'):
            line = raw_line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if not match:
                continue

            line_num = int(match.group(1))
            col_num = int(match.group(2))
            code = match.group(3)
            message = match.group(4).strip()

            # Filter out excluded issue codes
            if code in self.excluded_codes:
                continue

            issues.append(LinterIssue(
                line=line_num,
                column=col_num,
                code=code,
                message=message,
                severity=self._determine_severity('ruff', code, message)
            ))

        return issues
    
    def _parse_mypy_output(self, output: str) -> List[LinterIssue]:
        """Parse Mypy's output format"""
        issues = []
        
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Skip summary lines
            if line.startswith('Found ') or line.startswith('Success:'):
                continue
            
            # Format: file.py:line: error: message [error-code]
            # Format: file.py:line: warning: message [error-code]
            # Format: file.py:line: note: message
            match = re.match(r'^.+?:(\d+):\s*(error|warning|note):\s*(.+?)(?:\s+\[([^\]]+)\])?$', line)
            if match:
                line_num = int(match.group(1))
                severity_type = match.group(2)  # error, warning, or note
                message = match.group(3).strip()
                code = match.group(4) if match.group(4) else None

                # Filter out excluded issue codes
                if code and code in self.excluded_codes:
                    continue

                # Map mypy severity types to our severity levels
                if severity_type == 'error':
                    severity = 'error'
                elif severity_type == 'warning':
                    severity = 'warning'
                else:  # note
                    severity = 'info'
                
                issues.append(LinterIssue(
                    line=line_num,
                    column=None,
                    code=code,
                    message=message,
                    severity=severity
                ))
        
        return issues
    
    def _parse_line(self, linter_name: str, line: str) -> Optional[LinterIssue]:
        """Parse a single line of linter output"""
        # Handle format: file.py:line:col: CODE: message (pylint, some others)
        match = re.match(r'^.+?:(\d+):(\d+):\s*([A-Z]\d+):\s*(.+)$', line)
        if match:
            line_num = int(match.group(1))
            col_num = int(match.group(2))
            code = match.group(3)
            message = match.group(4).strip()
            
            # Filter out excluded issue codes
            if code in self.excluded_codes:
                return None
            
            severity = self._determine_severity(linter_name, code, message)
            
            return LinterIssue(
                line=line_num,
                column=col_num,
                code=code,
                message=message,
                severity=severity
            )
        
        # Handle format: file.py:line:col: CODE message (flake8, ruff)
        match = re.match(r'^.+?:(\d+):(\d+):\s*([A-Z]\d+)\s+(.+)$', line)
        if match:
            line_num = int(match.group(1))
            col_num = int(match.group(2))
            code = match.group(3)
            message = match.group(4).strip()
            
            # Filter out excluded issue codes
            if code in self.excluded_codes:
                return None
            
            severity = self._determine_severity(linter_name, code, message)
            
            return LinterIssue(
                line=line_num,
                column=col_num,
                code=code,
                message=message,
                severity=severity
            )
        
        # Handle format: file.py:line:col: message (no code)
        match = re.match(r'^.+?:(\d+):(\d+):\s*(.+)$', line)
        if match:
            line_num = int(match.group(1))
            col_num = int(match.group(2))
            message = match.group(3).strip()
            
            severity = self._determine_severity(linter_name, "", message)
            
            return LinterIssue(
                line=line_num,
                column=col_num,
                code="",
                message=message,
                severity=severity
            )
        
        # Handle format: file.py:line: CODE: message
        match = re.match(r'^.+?:(\d+):\s*([A-Z]\d+):\s*(.+)$', line)
        if match:
            line_num = int(match.group(1))
            code = match.group(2)
            message = match.group(3).strip()
            
            severity = self._determine_severity(linter_name, code, message)
            
            return LinterIssue(
                line=line_num,
                code=code,
                message=message,
                severity=severity
            )

        # Unrecognized lines are linter meta-output (e.g., pylint's
        # "***** Module xxx" or "Your code has been rated..."), not issues.
        return None
    #endregion C-PARSE

    #region C-SEVER
    def _determine_severity(self, linter_name: str, code: str, message: str) -> str:
        """Determine severity based on linter, code, and message.

        Severity mapping is linter-aware:
        - flake8/ruff: F codes are errors (pyflakes), E/W codes are style warnings
        - pylint:     E codes are errors, W are warnings, C/R are info
        - fallback:    message-keyword heuristic when no code is present
        """
        # Flake8 and Ruff share the same code system (pycodestyle + pyflakes)
        if linter_name in ('flake8', 'ruff'):
            if code:
                if code.startswith('F'):
                    return "error"
                elif code.startswith(('E', 'W')):
                    return "warning"
            # No code: use message-keyword heuristic
            message_lower = message.lower()
            if any(word in message_lower for word in ['error', 'undefined', 'missing', 'invalid']):
                return "error"
            return "warning"

        # Pylint has its own code system
        if linter_name == 'pylint':
            if code:
                if code.startswith('E'):
                    return "error"
                elif code.startswith('W'):
                    return "warning"
                elif code.startswith(('C', 'R')):
                    return "info"
                elif code.startswith('F'):
                    return "error"
            # No code: use message-keyword heuristic
            message_lower = message.lower()
            if any(word in message_lower for word in ['error', 'undefined', 'missing', 'invalid']):
                return "error"
            return self.LINTER_CONFIGS.get(linter_name, {}).get('severity', 'info')

        # Other linters: message-keyword heuristic first, then code prefix
        message_lower = message.lower()
        if any(word in message_lower for word in ['error', 'undefined', 'missing', 'invalid']):
            return "error"

        if code:
            if code.startswith(('E', 'F')):
                return "error"
            elif code.startswith('W'):
                return "warning"
            elif code.startswith(('C', 'R')):
                return "info"

        return self.LINTER_CONFIGS.get(linter_name, {}).get('severity', 'info')
    #endregion C-SEVER
#endregion LINTER


#region SCANNER
class NodeLinterScanner:
    """Scans ComfyUI_illumorae_* node packages and runs linters"""

    #region S-INIT
    def __init__(self, project_root: str, linters: Optional[List[str]] = None, 
                 skip_clean: bool = False, log_callback=None, progress_callback=None):
        self.project_root = os.path.abspath(project_root)
        self.linter_runner = LinterRunner(linters, log_callback)
        self.skip_clean = skip_clean
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.reports: List[NodeLinterReport] = []
        self.node_folders: List[Tuple[str, str]] = []
        
    def _log(self, message: str):
        """Log message via callback or print"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)
    
    def _progress(self, current: int, total: int, node_name: str, status: str,
                  report: Optional[NodeLinterReport] = None):
        """Report progress via callback"""
        if self.progress_callback:
            self.progress_callback(current, total, node_name, status, report)
    #endregion S-INIT

    #region S-DISC
    def discover_nodes(self) -> List[Tuple[str, str]]:
        """Discover all ComfyUI_illumorae_* node packages"""
        self._log(f"Scanning project root: {self.project_root}\n")
        
        node_folders = []
        for name in sorted(os.listdir(self.project_root)):
            folder_path = os.path.join(self.project_root, name)
            
            if not name.startswith("ComfyUI_illumorae_"):
                continue
            if not os.path.isdir(folder_path):
                continue
            if not os.path.isfile(os.path.join(folder_path, "__init__.py")):
                continue
            
            node_folders.append((name, folder_path))
        
        self.node_folders = node_folders
        self._log(f"Found {len(node_folders)} node package(s) to analyze\n")
        return node_folders
    #endregion S-DISC

    #region S-SCAN
    def scan_all_nodes(self) -> List[NodeLinterReport]:
        """Scan all ComfyUI_illumorae_* node packages"""
        if not self.node_folders:
            self.discover_nodes()
        
        self._log("=" * 80)
        
        for idx, (folder_name, folder_path) in enumerate(self.node_folders, 1):
            self._log(f"\n[{idx}/{len(self.node_folders)}] Processing: {folder_name}")
            self._log("-" * 80)
            self._progress(idx, len(self.node_folders), folder_name, "scanning")

            try:
                report = self.scan_node(folder_name, folder_path)
            except Exception as e:
                self._log(f"  ERROR: Failed to scan node: {e}")
                report = NodeLinterReport(
                    node_folder=folder_name,
                    scan_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    overall_status="error",
                    fix_suggestions=[f"Scan failed with error: {e}"],
                )
                self.reports.append(report)
                self._progress(idx, len(self.node_folders), folder_name, "error", report)
                continue

            if self.skip_clean and report.total_issues == 0:
                self._log(f"  Status: CLEAN - Skipping report file generation")
                self.reports.append(report)
                self._progress(idx, len(self.node_folders), folder_name, "skipped", report)
                continue

            self.reports.append(report)
            self._save_node_report(report, folder_path)

            self._log(f"  Status: {report.overall_status.upper()}")
            self._log(f"  Issues: {report.total_issues} total ({report.total_errors} errors, {report.total_warnings} warnings)")
            self._progress(idx, len(self.node_folders), folder_name, report.overall_status, report)
        
        self._log("\n" + "=" * 80)
        self._log(f"Completed scanning {len(self.node_folders)} node(s)")
        self._log(f"Generated {len(self.reports)} report(s)")
        
        return self.reports
    
    def scan_node(self, folder_name: str, folder_path: str) -> NodeLinterReport:
        """Scan a single node package"""
        report = NodeLinterReport(
            node_folder=folder_name,
            scan_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            linters_available=list(self.linter_runner.available_linters),
            linters_skipped=list(self.linter_runner.missing_linters)
        )
        
        py_files = []
        for fname in os.listdir(folder_path):
            if fname.endswith('.py') and fname != '__init__.py':
                fpath = os.path.join(folder_path, fname)
                if os.path.isfile(fpath):
                    py_files.append((fname, fpath))
        
        report.python_files = [fname for fname, _ in py_files]
        
        if not py_files:
            self._log(f"  No Python files found (excluding __init__.py)")
            report.overall_status = "skipped"
            return report
        
        report.node_id, report.display_name = self._extract_node_metadata(py_files)
        report.linters_run = sorted(self.linter_runner.available_linters)

        for fname, fpath in py_files:
            self._log(f"  File: {fname}")
            file_results = []
            
            for linter_name in sorted(self.linter_runner.available_linters):
                result = self.linter_runner.run_linter(linter_name, fpath)
                file_results.append(result)
                
                status_symbol = "[OK]" if result.status == "pass" else "[FAIL]"
                issue_text = f"{result.issue_count} issues" if result.issue_count > 0 else "clean"
                self._log(f"    {linter_name:10} ... {status_symbol} {issue_text}")
                
                for issue in result.issues:
                    if issue.severity == "error":
                        report.total_errors += 1
                    elif issue.severity == "warning":
                        report.total_warnings += 1
                    else:
                        report.total_info += 1
            
            report.file_results[fname] = file_results

        report.total_issues = report.total_errors + report.total_warnings + report.total_info
        
        if report.total_errors > 0:
            report.overall_status = "error"
        elif report.total_warnings > 0:
            report.overall_status = "warning"
        else:
            report.overall_status = "pass"
        
        report.fix_suggestions = self._generate_fix_suggestions(report)
        report.summary = self._generate_summary(report)
        
        return report
    #endregion S-SCAN

    #region S-META
    def _extract_node_metadata(self, py_files: List[Tuple[str, str]]) -> Tuple[Optional[str], Optional[str]]:
        """Extract node ID and display name from Python files"""
        for fname, fpath in py_files:
            try:
                with open(fpath, 'r', encoding='utf-8-sig') as f:
                    content = f.read()

                node_id_match = re.search(r'NODE_CLASS_MAPPINGS\s*=\s*\{[^}]*["\']([^"\']+)["\']\s*:', content)
                if node_id_match:
                    node_id = node_id_match.group(1)

                    # Match the specific node_id as a literal quoted dict key,
                    # then capture the display name value.
                    display_match = re.search(
                        rf'NODE_DISPLAY_NAME_MAPPINGS\s*=\s*\{{[^}}]*["\']({re.escape(node_id)})["\']\s*:\s*["\']([^"\']+)["\']',
                        content
                    )
                    display_name = display_match.group(2) if display_match else None

                    return node_id, display_name
            except Exception:
                continue

        return None, None
    #endregion S-META

    #region S-REPT
    def _generate_fix_suggestions(self, report: NodeLinterReport) -> List[str]:
        """Generate actionable fix suggestions based on linter results"""
        suggestions = []
        
        linter_issue_counts = {}
        for file_results in report.file_results.values():
            for result in file_results:
                if result.issue_count > 0:
                    linter_issue_counts[result.linter_name] = linter_issue_counts.get(result.linter_name, 0) + result.issue_count
        
        if 'black' in linter_issue_counts:
            suggestions.append(f"Run 'black {report.node_folder}/*.py' to auto-format code")
        
        if 'ruff' in linter_issue_counts:
            suggestions.append(f"Run 'ruff check --fix {report.node_folder}/*.py' to auto-fix some issues")
        
        if 'flake8' in linter_issue_counts:
            suggestions.append("Review PEP 8 style violations (E/W codes) and fix manually")
        
        if 'mypy' in linter_issue_counts:
            suggestions.append("Add type hints to functions and variables")
        
        if 'pylint' in linter_issue_counts:
            suggestions.append("Add docstrings, improve naming, and reduce complexity")
        
        if report.total_errors > 0:
            suggestions.insert(0, f"PRIORITY: Fix {report.total_errors} error(s) first")
        
        return suggestions
    
    def _generate_summary(self, report: NodeLinterReport) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            'total_files': len(report.python_files),
            'total_linters': len(report.linters_run),
            'total_issues': report.total_issues,
            'by_severity': {
                'errors': report.total_errors,
                'warnings': report.total_warnings,
                'info': report.total_info
            },
            'by_linter': {}
        }
        
        for file_results in report.file_results.values():
            for result in file_results:
                if result.linter_name not in summary['by_linter']:
                    summary['by_linter'][result.linter_name] = {
                        'total_issues': 0,
                        'status': 'pass'
                    }
                summary['by_linter'][result.linter_name]['total_issues'] += result.issue_count
                if result.status != 'pass':
                    summary['by_linter'][result.linter_name]['status'] = 'fail'
        
        return summary
    
    def _save_node_report(self, report: NodeLinterReport, folder_path: str) -> None:
        """Save node linter report as JSON"""
        output_file = os.path.join(folder_path, "node_linter_report.json")
        
        report_dict = asdict(report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2)
        
        self._log(f"  Report saved: node_linter_report.json")
    
    def export_summary(self, output_path: str) -> None:
        """Export consolidated summary of all reports"""
        summary = {
            'scan_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'project_root': self.project_root,
            'total_nodes': len(self.reports),
            'linters_used': list(self.linter_runner.available_linters),
            'nodes': []
        }
        
        for report in self.reports:
            summary['nodes'].append({
                'folder': report.node_folder,
                'node_id': report.node_id,
                'display_name': report.display_name,
                'status': report.overall_status,
                'total_issues': report.total_issues,
                'errors': report.total_errors,
                'warnings': report.total_warnings,
                'info': report.total_info,
                'fix_suggestions': report.fix_suggestions
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self._log(f"\nSummary exported to: {output_path}")
    #endregion S-REPT
#endregion SCANNER


#region GUI
class LinterDashboardGUI:
    """Tkinter GUI for linter runner"""

    #region G-INIT
    def __init__(self, root: tk.Tk, project_root: str, linters: Optional[List[str]] = None):
        self.root = root
        self.project_root = os.path.abspath(project_root)
        self.requested_linters = linters
        self.scanner: Optional[NodeLinterScanner] = None
        self.node_reports: Dict[str, NodeLinterReport] = {}
        self.is_running = False
        
        self._setup_ui()
        self._discover_nodes()
    #endregion G-INIT

    #region G-THEME
    def _apply_dark_theme(self) -> None:
        """Apply dark theme to GUI"""
        self.root.configure(bg=COLORS["bg_black"])
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure(
            ".",
            background=COLORS["bg_black"],
            foreground=COLORS["fg_text"],
            fieldbackground=COLORS["bg_black"],
            bordercolor=COLORS["border"],
        )
        style.configure(
            "Treeview",
            background=COLORS["bg_black"],
            foreground=COLORS["fg_text"],
            fieldbackground=COLORS["bg_black"],
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
    #endregion G-THEME

    #region G-UI
    def _setup_ui(self) -> None:
        """Setup GUI layout"""
        self.root.title("illumorae Node Linter Runner")
        self._apply_dark_theme()
        
        main_frame = tk.Frame(self.root, bg=COLORS["bg_black"], padx=10, pady=10)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=3)
        main_frame.rowconfigure(2, weight=1)
        
        self._create_toolbar(main_frame)
        self._create_node_list(main_frame)
        self._create_logging_area(main_frame)
        
    def _create_toolbar(self, parent: tk.Widget) -> None:
        """Create toolbar with controls"""
        bar = ttk.Frame(parent)
        bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        bar.columnconfigure(1, weight=1)
        
        ttk.Label(bar, text="Project Root:").grid(row=0, column=0, sticky=tk.W)
        self.root_var = tk.StringVar(value=self.project_root)
        ttk.Entry(bar, textvariable=self.root_var, width=60).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(bar, text="Browse", command=self._browse_root).grid(row=0, column=2, sticky=tk.W, padx=5)
        
        self.run_button = ttk.Button(bar, text="Run Linters", command=self._start_linting)
        self.run_button.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Button(bar, text="Export Summary", command=self._export_summary).grid(row=0, column=4, sticky=tk.W, padx=5)
        
        linter_frame = ttk.Frame(bar)
        linter_frame.grid(row=1, column=0, columnspan=5, sticky=tk.W, pady=(5, 0))
        
        ttk.Label(linter_frame, text="Linters:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.linter_vars = {}
        for linter in ['black', 'ruff', 'flake8', 'mypy', 'pylint']:
            var = tk.BooleanVar(value=True)
            self.linter_vars[linter] = var
            ttk.Checkbutton(linter_frame, text=linter, variable=var).pack(side=tk.LEFT, padx=5)
        
        self.skip_clean_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(linter_frame, text="Skip Clean Nodes", variable=self.skip_clean_var).pack(side=tk.LEFT, padx=(20, 5))
        
    def _create_node_list(self, parent: tk.Widget) -> None:
        """Create node list with status indicators"""
        list_frame = ttk.Frame(parent)
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        header = ttk.Frame(list_frame)
        header.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(header, text="Nodes", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)
        self.count_label = ttk.Label(header, text="(0)")
        self.count_label.pack(side=tk.LEFT, padx=5)
        
        tree_frame = ttk.Frame(list_frame)
        tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("status", "node_id", "display", "errors", "warnings", "info", "total"),
            show="tree headings",
            yscrollcommand=vsb.set
        )
        
        vsb.config(command=self.tree.yview)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.tree.heading("#0", text="Node Folder")
        self.tree.heading("status", text="Status")
        self.tree.heading("node_id", text="Node ID")
        self.tree.heading("display", text="Display Name")
        self.tree.heading("errors", text="Errors")
        self.tree.heading("warnings", text="Warnings")
        self.tree.heading("info", text="Info")
        self.tree.heading("total", text="Total")
        
        self.tree.column("#0", width=250)
        self.tree.column("status", width=80, anchor="center")
        self.tree.column("node_id", width=200)
        self.tree.column("display", width=200)
        self.tree.column("errors", width=60, anchor="center")
        self.tree.column("warnings", width=80, anchor="center")
        self.tree.column("info", width=60, anchor="center")
        self.tree.column("total", width=60, anchor="center")
        
        self.tree.tag_configure("pending", foreground=COLORS["fg_dim"])
        self.tree.tag_configure("scanning", foreground=COLORS["accent_blue"])
        self.tree.tag_configure("pass", foreground=COLORS["accent_green"])
        self.tree.tag_configure("warning", foreground=COLORS["accent_yellow"])
        self.tree.tag_configure("error", foreground=COLORS["accent_red"])
        self.tree.tag_configure("skipped", foreground=COLORS["fg_dim"])
        
        legend_frame = ttk.Frame(list_frame)
        legend_frame.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        ttk.Label(legend_frame, text="Status: ", font=("TkDefaultFont", 8)).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="#", font=("TkDefaultFont", 10, "bold"), foreground=COLORS["accent_green"]).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="Pass ", font=("TkDefaultFont", 8), foreground=COLORS["fg_dim"]).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="#", font=("TkDefaultFont", 10, "bold"), foreground=COLORS["accent_yellow"]).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="Warnings ", font=("TkDefaultFont", 8), foreground=COLORS["fg_dim"]).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="#", font=("TkDefaultFont", 10, "bold"), foreground=COLORS["accent_red"]).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="Errors", font=("TkDefaultFont", 8), foreground=COLORS["fg_dim"]).pack(side=tk.LEFT)
        
    def _create_logging_area(self, parent: tk.Widget) -> None:
        """Create logging area at bottom"""
        log_frame = ttk.Frame(parent)
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(1, weight=1)
        
        header = ttk.Frame(log_frame)
        header.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        
        ttk.Label(header, text="Status & Logging", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(header, textvariable=self.progress_var, foreground=COLORS["fg_dim"]).pack(side=tk.RIGHT)
        
        log_scroll_frame = ttk.Frame(log_frame)
        log_scroll_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scroll_frame.columnconfigure(0, weight=1)
        log_scroll_frame.rowconfigure(0, weight=1)
        
        log_vsb = ttk.Scrollbar(log_scroll_frame, orient="vertical")
        
        self.log_text = tk.Text(
            log_scroll_frame,
            wrap=tk.WORD,
            bg=COLORS["bg_black"],
            fg=COLORS["fg_dim"],
            insertbackground=COLORS["fg_text"],
            height=8,
            yscrollcommand=log_vsb.set
        )
        
        log_vsb.config(command=self.log_text.yview)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(state="disabled")
        
        self._log("Dashboard initialized. Click 'Run Linters' to start.")
    #endregion G-UI

    #region G-LOG
    def _log(self, message: str) -> None:
        """Add message to log area"""
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()
    #endregion G-LOG

    #region G-DISC
    def _browse_root(self) -> None:
        """Browse for project root directory"""
        path = filedialog.askdirectory(initialdir=self.project_root)
        if path:
            self.root_var.set(path)
            self.project_root = os.path.abspath(path)
            self._discover_nodes()
            
    def _discover_nodes(self) -> None:
        """Discover nodes and populate tree"""
        self._log(f"Discovering nodes in: {self.project_root}")
        
        self.tree.delete(*self.tree.get_children())
        self.node_reports.clear()
        
        node_folders = []
        try:
            for name in sorted(os.listdir(self.project_root)):
                folder_path = os.path.join(self.project_root, name)
                
                if not name.startswith("ComfyUI_illumorae_"):
                    continue
                if not os.path.isdir(folder_path):
                    continue
                if not os.path.isfile(os.path.join(folder_path, "__init__.py")):
                    continue
                
                node_folders.append((name, folder_path))
        except Exception as e:
            self._log(f"ERROR: {e}")
            return
        
        for folder_name, folder_path in node_folders:
            report = NodeLinterReport(
                node_folder=folder_name,
                overall_status="pending"
            )
            self.node_reports[folder_name] = report
            
            self.tree.insert(
                "",
                "end",
                folder_name,
                text=folder_name,
                values=("Pending", "", "", "-", "-", "-", "-"),
                tags=("pending",)
            )
        
        self.count_label.config(text=f"({len(node_folders)})")
        self._log(f"Found {len(node_folders)} node package(s)")
    #endregion G-DISC

    #region G-RUN
    def _start_linting(self) -> None:
        """Start linting process in background thread"""
        if self.is_running:
            messagebox.showwarning("Already Running", "Linting is already in progress")
            return
        
        if not self.node_reports:
            messagebox.showwarning("No Nodes", "No nodes found to lint")
            return
        
        selected_linters = [name for name, var in self.linter_vars.items() if var.get()]
        if not selected_linters:
            messagebox.showwarning("No Linters", "Please select at least one linter")
            return
        
        self.is_running = True
        self.run_button.config(state="disabled", text="Running...")
        self._log("\n" + "=" * 80)
        self._log("Starting linter scan...")
        self._log("=" * 80)
        
        thread = threading.Thread(
            target=self._run_linting_thread,
            args=(selected_linters, self.skip_clean_var.get()),
            daemon=True
        )
        thread.start()
        
    def _run_linting_thread(self, linters: List[str], skip_clean: bool) -> None:
        """Run linting in background thread"""
        try:
            self.scanner = NodeLinterScanner(
                self.project_root,
                linters,
                skip_clean,
                log_callback=self._log,
                progress_callback=self._update_progress
            )
            
            self.scanner.discover_nodes()
            reports = self.scanner.scan_all_nodes()

            # Update node_reports on the main thread to avoid concurrent dict access
            self.root.after(0, lambda: self._merge_reports(reports))
            self.root.after(0, self._linting_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self._log(f"\nERROR: {e}"))
            self.root.after(0, self._linting_complete)

    def _merge_reports(self, reports: List[NodeLinterReport]) -> None:
        """Merge completed reports into node_reports (called on main thread)"""
        for report in reports:
            self.node_reports[report.node_folder] = report
    #endregion G-RUN

    #region G-PROG
    def _update_progress(self, current: int, total: int, node_name: str, status: str,
                         report: Optional[NodeLinterReport] = None) -> None:
        """Update progress display (called from background thread)"""
        self.root.after(0, lambda: self.progress_var.set(f"Progress: {current}/{total} - {node_name}"))

        if report:
            self.root.after(0, lambda: self._update_tree_item(node_name, report))
            
    def _update_tree_item(self, folder_name: str, report: NodeLinterReport) -> None:
        """Update tree item with report data"""
        if not self.tree.exists(folder_name):
            return
        
        status_text = report.overall_status.upper()
        if report.overall_status == "pending":
            status_text = "Pending"
        elif report.overall_status == "scanning":
            status_text = "Scanning..."
        elif report.overall_status == "pass":
            status_text = "[PASS]"
        elif report.overall_status == "warning":
            status_text = "[WARN]"
        elif report.overall_status == "error":
            status_text = "[ERROR]"
        elif report.overall_status == "skipped":
            status_text = "Skipped"
        
        self.tree.item(
            folder_name,
            values=(
                status_text,
                report.node_id or "",
                report.display_name or "",
                report.total_errors if report.total_errors > 0 else "-",
                report.total_warnings if report.total_warnings > 0 else "-",
                report.total_info if report.total_info > 0 else "-",
                report.total_issues if report.total_issues > 0 else "-"
            ),
            tags=(report.overall_status,)
        )
        
    def _linting_complete(self) -> None:
        """Handle linting completion"""
        self.is_running = False
        self.run_button.config(state="normal", text="Run Linters")
        self.progress_var.set("Complete")
        
        total_issues = sum(r.total_issues for r in self.node_reports.values() if r.overall_status != "pending")
        total_errors = sum(r.total_errors for r in self.node_reports.values() if r.overall_status != "pending")
        total_warnings = sum(r.total_warnings for r in self.node_reports.values() if r.overall_status != "pending")
        
        self._log("\n" + "=" * 80)
        self._log("LINTING COMPLETE")
        self._log("=" * 80)
        self._log(f"Total Issues: {total_issues}")
        self._log(f"  Errors: {total_errors}")
        self._log(f"  Warnings: {total_warnings}")
        self._log(f"\nReports saved to: <node_folder>/node_linter_report.json")
    #endregion G-PROG

    #region G-EXPORT
    def _export_summary(self) -> None:
        """Export consolidated summary"""
        if not self.scanner or not self.scanner.reports:
            messagebox.showwarning("No Data", "No linting data to export. Run linters first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="linter_summary.json"
        )
        
        if file_path:
            try:
                self.scanner.export_summary(file_path)
                messagebox.showinfo("Success", f"Summary exported to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export summary:\n{e}")
    #endregion G-EXPORT
#endregion GUI


#region MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Run linters on ComfyUI_illumorae_* node packages and generate reports"
    )
    
    # Default to parent directory of this script (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.dirname(script_dir)
    
    parser.add_argument(
        'project_root',
        nargs='?',
        default=default_root,
        help=f'Project root directory (default: {default_root})'
    )
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run in CLI mode without GUI'
    )
    parser.add_argument(
        '--linters',
        type=str,
        help='Comma-separated list of linters to run (default: all available)'
    )
    parser.add_argument(
        '--skip-clean',
        action='store_true',
        help='Skip generating reports for nodes with no issues'
    )
    parser.add_argument(
        '--export-summary',
        type=str,
        metavar='PATH',
        help='Export consolidated summary JSON to specified path'
    )
    
    args = parser.parse_args()
    
    linters = None
    if args.linters:
        linters = [l.strip() for l in args.linters.split(',')]
    
    if not os.path.exists(args.project_root):
        print(f"ERROR: Project root '{args.project_root}' does not exist")
        sys.exit(1)
    
    if args.no_gui:
        print("=" * 80)
        print("illumorae Node Linter Runner (CLI Mode)")
        print("=" * 80)
        
        scanner = NodeLinterScanner(args.project_root, linters, args.skip_clean)
        reports = scanner.scan_all_nodes()
        
        if args.export_summary:
            scanner.export_summary(args.export_summary)
        
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        
        total_issues = sum(r.total_issues for r in reports)
        total_errors = sum(r.total_errors for r in reports)
        total_warnings = sum(r.total_warnings for r in reports)
        
        print(f"Nodes scanned: {len(reports)}")
        print(f"Total issues: {total_issues}")
        print(f"  Errors: {total_errors}")
        print(f"  Warnings: {total_warnings}")
        print(f"  Info: {sum(r.total_info for r in reports)}")
        
        nodes_with_errors = [r for r in reports if r.total_errors > 0]
        if nodes_with_errors:
            print(f"\nNodes with errors ({len(nodes_with_errors)}):")
            for r in nodes_with_errors:
                print(f"  - {r.node_folder}: {r.total_errors} error(s)")
        
        print("\nReports saved to: <node_folder>/node_linter_report.json")
        print()
    else:
        try:
            root = tk.Tk()
            root.geometry("1200x800")
            app = LinterDashboardGUI(root, args.project_root, linters)
            root.mainloop()
        except tk.TclError as e:
            print(f"ERROR: Cannot launch GUI ({e}).")
            print("Hint: Use --no-gui for CLI mode on headless systems.")
            sys.exit(1)


if __name__ == "__main__":
    main()
#endregion MAIN
