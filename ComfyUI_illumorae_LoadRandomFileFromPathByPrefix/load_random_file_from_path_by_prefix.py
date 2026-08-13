"""
illumorae Load Random File From Path By Prefix - a ComfyUI custom_node

Pick a random file from a folder, restricted by filename prefix and
file extension. Designed for selecting one variant of a prompt-style
text file that shares a model-class prefix.

Example
    folder:    D:/prompts
    prefix:    prompt_wan22_
    extension: .md

    Matches:
        prompt_wan22_english.md
        prompt_wan22_japanese.md
    Skips:
        prompt_sdxl_english.md   (prefix mismatch)
        prompt_wan22_notes.txt   (extension mismatch)

Multiple extensions are accepted as a comma-separated list, 
``.md,.txt``. A single asterisk ``*`` extension disables the extension filter.

TITLE::Load Random File From Path By Prefix
DESCRIPTIONSHORT::Pick a random file from a folder filtered by filename prefix and extension; return its text contents and path.
VERSION::20260517
IMAGE::comfyui_illumorae_load_random_file_from_path_by_prefix.png
GROUP::Load
"""

import os
import random
from pathlib import Path


class illumoraeLoadRandomFileFromPathByPrefixNode:
    """Load a random text file from a folder by prefix and extension.

    The folder is scanned (optionally recursively) for files whose
    base name starts with ``prefix`` and whose extension matches one
    of the entries in ``extension``. One match is selected; either
    randomly using ``seed``, or deterministically when
    ``index_override`` is a positive 1-based index.

    Returns the file text, the file stem, the absolute file path, and
    the parent folder path. If no file matches, empty strings are
    returned and a status string describes the cause.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": r"C:/prompts"}),
                "prefix": ("STRING", {"default": "prompt_wan22_"}),
                "extension": ("STRING", {"default": ".md"}),
                "recursive": ("BOOLEAN", {"default": False}),
                "case_sensitive": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1}),
                "index_override": ("INT", {"default": -1, "min": -1}),
            },
            "optional": {
                "debug_mode": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "illumorae"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "file_name", "file_path", "folder_path", "status")
    FUNCTION = "load_random_file"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Pick a random file from a folder filtered by filename prefix "
        "and extension; return text contents and path metadata."
    )

    def _dprint(self, debug_mode, *args, **kwargs):
        if debug_mode:
            print(*args, **kwargs)

    @staticmethod
    def _parse_extensions(extension_field):
        """Parse the extension input into a normalized list.

        Returns a list of lowercase extensions starting with ``.``.
        A single entry ``*`` (or empty string after split) disables
        the extension filter and is signalled by returning ``None``.
        """
        if extension_field is None:
            return [".md"]
        raw = [e.strip() for e in str(extension_field).split(",") if e.strip()]
        if not raw:
            return None
        if any(e == "*" for e in raw):
            return None
        out = []
        for e in raw:
            if not e.startswith("."):
                e = "." + e
            out.append(e.lower())
        return out

    def _collect_matches(self, folder_path, prefix, extensions, recursive,
                         case_sensitive, debug_mode):
        """Walk ``folder_path`` and classify every file.

        Returns a tuple ``(matches, skipped)`` where ``matches`` is a
        list of ``Path`` and ``skipped`` is a list of
        ``(Path, reason)`` tuples. ``reason`` is a short string such
        as ``"prefix mismatch"`` or ``"extension mismatch (.txt)"``.
        """
        matches = []
        skipped = []
        if not folder_path.exists() or not folder_path.is_dir():
            return matches, skipped

        cmp_prefix = prefix if case_sensitive else prefix.lower()

        if recursive:
            iterator = folder_path.rglob("*")
        else:
            iterator = folder_path.iterdir()

        for entry in iterator:
            if not entry.is_file():
                continue
            name = entry.name
            stem = entry.stem
            ext = entry.suffix.lower()

            cmp_name = name if case_sensitive else name.lower()
            cmp_stem = stem if case_sensitive else stem.lower()

            # Prefix is matched against the file name (without folder),
            # so a prefix like "prompt_wan22_" will match either
            # "prompt_wan22_english.md" or "prompt_wan22_v2.md".
            prefix_ok = (
                cmp_prefix == ""
                or cmp_name.startswith(cmp_prefix)
                or cmp_stem.startswith(cmp_prefix)
            )
            ext_ok = extensions is None or ext in extensions

            if not prefix_ok and not ext_ok:
                skipped.append((entry, f"prefix mismatch and extension mismatch ({ext or '<none>'})"))
                continue
            if not prefix_ok:
                skipped.append((entry, f"prefix mismatch (expected {prefix!r})"))
                continue
            if not ext_ok:
                skipped.append((entry, f"extension mismatch ({ext or '<none>'})"))
                continue

            matches.append(entry)
            self._dprint(debug_mode, f"  match: {entry}")

        # Deterministic order across platforms before any random pick.
        matches.sort(key=lambda p: str(p).lower())
        skipped.sort(key=lambda t: str(t[0]).lower())
        return matches, skipped

    def load_random_file(self, folder, prefix, extension, recursive,
                         case_sensitive, seed, index_override,
                         debug_mode=False):
        # Input validation:  non-string inputs are converted to strings.
        folder = "" if folder is None else str(folder)
        prefix = "" if prefix is None else str(prefix)

        if not folder:
            return "", "", "", "", "Folder path is empty"

        folder_path = Path(folder)
        self._dprint(debug_mode, "[LoadRandomFileFromPathByPrefix] inputs:")
        self._dprint(debug_mode, f"  folder: {folder_path}")
        self._dprint(debug_mode, f"  prefix: {prefix!r}")
        self._dprint(debug_mode, f"  extension: {extension!r}")
        self._dprint(debug_mode, f"  recursive: {recursive}")
        self._dprint(debug_mode, f"  case_sensitive: {case_sensitive}")
        self._dprint(debug_mode, f"  seed: {seed}, index_override: {index_override}")

        if not folder_path.exists() or not folder_path.is_dir():
            msg = f"Folder not found: {folder_path}"
            print(f"[LoadRandomFileFromPathByPrefix] {msg}")
            return "", "", "", "", msg

        extensions = self._parse_extensions(extension)
        self._dprint(debug_mode, f"  parsed extensions: {extensions}")

        matches, skipped = self._collect_matches(
            folder_path, prefix, extensions, recursive, case_sensitive, debug_mode,
        )

        ext_display = "<any>" if extensions is None else ",".join(extensions)

        if not matches:
            header = (
                f"No files match prefix={prefix!r} extension={ext_display} "
                f"in {folder_path} (recursive={recursive})"
            )
            report = self._format_report(
                folder_path, prefix, ext_display, recursive,
                case_sensitive, matches, skipped, chosen=None, header=header,
            )
            print(f"[LoadRandomFileFromPathByPrefix] {header}")
            return "", "", "", str(folder_path), report

        # Seed for reproducibility. seed=-1 uses non-deterministic state.
        if seed is not None and seed >= 0:
            random.seed(seed)

        chosen = None
        if index_override is not None and index_override > 0:
            idx = index_override - 1
            if 0 <= idx < len(matches):
                chosen = matches[idx]
                self._dprint(debug_mode, f"  override index {index_override} -> {chosen}")
            else:
                self._dprint(
                    debug_mode,
                    f"  index_override {index_override} out of range "
                    f"(have {len(matches)}); falling back to random",
                )

        if chosen is None:
            chosen = random.choice(matches)
            self._dprint(debug_mode, f"  random pick: {chosen}")

        try:
            with open(chosen, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # Fallback for non-utf8 files; lossy decode preserves a usable string.
            with open(chosen, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            msg = f"Error reading {chosen}: {e}"
            print(f"[LoadRandomFileFromPathByPrefix] {msg}")
            return "", chosen.stem, str(chosen), str(chosen.parent), msg

        header = (
            f"Loaded {chosen.name} "
            f"({len(matches)} match{'es' if len(matches) != 1 else ''}, "
            f"{len(skipped)} skipped)"
        )
        report = self._format_report(
            folder_path, prefix, ext_display, recursive,
            case_sensitive, matches, skipped, chosen=chosen, header=header,
        )
        self._dprint(debug_mode, f"  {header}")
        return text, chosen.stem, str(chosen), str(chosen.parent), report

    @staticmethod
    def _format_report(folder_path, prefix, ext_display, recursive,
                       case_sensitive, matches, skipped, chosen, header):
        """Build a multi-line status report.

        Sections: header, filter summary, matches list (with the
        chosen entry marked), skipped list grouped by reason.
        Paths are rendered relative to ``folder_path`` when possible,
        otherwise as absolute paths.
        """
        def rel(p):
            try:
                return str(p.relative_to(folder_path))
            except ValueError:
                return str(p)

        lines = [header]
        lines.append("--- filter ---")
        lines.append(f"folder: {folder_path}")
        lines.append(f"prefix: {prefix!r}")
        lines.append(f"extension: {ext_display}")
        lines.append(f"recursive: {recursive} | case_sensitive: {case_sensitive}")

        lines.append(f"--- matches ({len(matches)}) ---")
        if matches:
            for i, p in enumerate(matches, 1):
                marker = " <-- chosen" if chosen is not None and p == chosen else ""
                lines.append(f"  [{i}] {rel(p)}{marker}")
        else:
            lines.append("  (none)")

        lines.append(f"--- skipped ({len(skipped)}) ---")
        if skipped:
            grouped = {}
            for p, reason in skipped:
                grouped.setdefault(reason, []).append(p)
            for reason in sorted(grouped):
                paths = grouped[reason]
                lines.append(f"  [{reason}] ({len(paths)})")
                for p in paths:
                    lines.append(f"    - {rel(p)}")
        else:
            lines.append("  (none)")

        return "\n".join(lines)

    @classmethod
    def IS_CHANGED(cls, folder="", prefix="", extension="", recursive=False,
                   case_sensitive=False, seed=-1, index_override=-1,
                   debug_mode=False, **kwargs):
        # Re-run when any input changes, and when seed=-1 (non-deterministic).
        if seed is None or seed < 0:
            return float("nan")
        return f"{folder}|{prefix}|{extension}|{recursive}|{case_sensitive}|{seed}|{index_override}"


NODE_CLASS_MAPPINGS = {
    "illumoraeLoadRandomFileFromPathByPrefixNode": illumoraeLoadRandomFileFromPathByPrefixNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeLoadRandomFileFromPathByPrefixNode": "Load Random File From Path By Prefix",
}
