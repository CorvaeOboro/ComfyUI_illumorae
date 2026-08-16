"""
TITLE::Enclosure Visualizer
DESCRIPTIONSHORT::Visualizes prompt text enclosure nesting and mismatched parenthesis, outputting an image and analysis strings.
VERSION::20260816
IMAGE::comfyui_illumorae_text_enclosure_visualize.png
GROUP::Text
GROUPORDER::5
LISTORDER::40
STATUS::working
"""
#region imports
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

logger = logging.getLogger(__name__)
#endregion


class illumoraeEnclosureVisualizerNode:
    """
    A ComfyUI node that visualizes nested parenthesis in prompt text.

    Features:
    - Color-codes nested parenthesis levels with muted colors
    - Detects and highlights hanging/mismatched parenthesis in red
    - Warns about multiple layered enclosures
    - Outputs an image showing the text structure
    """

    #region init
    def __init__(self):
        """Initialize with color palette for different nesting levels."""
        self.muted_colors = [
            (120, 150, 120),  # Muted green
            (100, 120, 150),  # Muted blue
            (130, 110, 150),  # Muted purple
            (110, 140, 130),  # Muted teal
            (140, 120, 140),  # Muted lavender
        ]
        self.error_color = (200, 80, 80)  # Red for errors
        self.warning_color = (200, 150, 80)  # Orange for warnings
        self.base_color = (200, 200, 200)  # Light gray for base text
    #endregion

    #region parse
    def parse_enclosures(self, text: str) -> tuple[list[dict], list[tuple[str, str]], int]:
        """
        Parse text to identify nested parenthesis and their depths.

        Returns:
            - List of character info dicts with position, char, depth, and flags
            - List of (category, message) warning tuples; category is "hanging" or "nesting"
            - Maximum nesting depth reached
        """
        char_info: list[dict] = []
        # Each stack entry: (position, char_info_index, is_deep)
        depth_stack: list[tuple[int, int, bool]] = []
        warnings: list[tuple[str, str]] = []
        max_depth = 0

        for i, char in enumerate(text):
            if char == '\n':
                # Newlines are not rendered; skip them so char_info stays
                # aligned with the visible characters produced by text.split('\n')
                continue

            depth = len(depth_stack)

            if char == '(':
                is_deep = depth + 1 > 3
                # Warn once per threshold crossing (depth == 4), not per character
                if depth + 1 == 4:
                    warnings.append(("nesting", f"Deep nesting begins at position {i}: depth {depth + 1} (not preferred)"))
                depth_stack.append((i, len(char_info), is_deep))
                depth = len(depth_stack)
                max_depth = max(max_depth, depth)
                char_info.append({
                    'pos': i,
                    'char': char,
                    'depth': depth,
                    'is_error': False,
                    'is_warning': is_deep,
                })

            elif char == ')':
                if not depth_stack:
                    warnings.append(("hanging", f"Hanging closing parenthesis at position {i}"))
                    char_info.append({
                        'pos': i,
                        'char': char,
                        'depth': 0,
                        'is_error': True,
                        'is_warning': False,
                    })
                else:
                    _, opener_idx, opener_was_deep = depth_stack.pop()
                    depth = len(depth_stack) + 1  # Color at the level it closes
                    char_info.append({
                        'pos': i,
                        'char': char,
                        'depth': depth,
                        'is_error': False,
                        'is_warning': opener_was_deep,
                    })
            else:
                char_info.append({
                    'pos': i,
                    'char': char,
                    'depth': depth,
                    'is_error': False,
                    'is_warning': False,
                })

        # Mark unclosed opening parens as errors (O(1) via stored index)
        for pos, opener_idx, _ in depth_stack:
            warnings.append(("hanging", f"Unclosed opening parenthesis at position {pos}"))
            char_info[opener_idx]['is_error'] = True

        return char_info, warnings, max_depth
    #endregion

    #region color
    def get_color_for_depth(self, depth: int, is_error: bool, is_warning: bool) -> tuple[int, int, int]:
        """Get color based on depth level, error, or warning status."""
        if is_error:
            return self.error_color
        if is_warning:
            return self.warning_color
        if depth == 0:
            return self.base_color

        # Cycle through muted colors for different depths
        color_index = (depth - 1) % len(self.muted_colors)
        return self.muted_colors[color_index]
    #endregion

    #region font
    _FONT_CANDIDATES = [
        "consola.ttf",        # Windows Consolas
        "cour.ttf",           # Windows Courier New
        "DejaVuSansMono.ttf", # Linux DejaVu
        "LiberationMono.ttf", # Linux Liberation
        "Menlo.ttf",          # macOS Menlo
        "Courier New.ttf",    # macOS Courier New
    ]

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        """Load a monospace font by trying cross-platform candidates, then default."""
        for name in self._FONT_CANDIDATES:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _measure_char_width(self, font: ImageFont.ImageFont, font_size: int) -> float:
        """Measure the advance width of a monospace glyph; fall back to estimate."""
        try:
            width = font.getlength("M")
            if width > 0:
                return width
        except (AttributeError, OSError):
            pass
        return font_size * 0.6
    #endregion

    #region render
    def create_visualization_image(
        self,
        text: str,
        char_info: list[dict],
        warnings: list[tuple[str, str]],
        font_size: int,
        line_height: float,
        padding: int,
        background_color: str
    ) -> Image.Image:
        """Create a PIL Image with color-coded text visualization."""

        #region R-setup
        # Set background color
        if background_color == "dark":
            bg_color = (30, 30, 35)
        else:
            bg_color = (240, 240, 245)

        font = self._load_font(font_size)
        #endregion

        #region R-dims
        # Calculate image dimensions
        lines = text.split('\n')
        max_line_length = max(len(line) for line in lines)

        # Measure actual character width from the loaded font
        char_width = self._measure_char_width(font, font_size)
        line_height_px = int(font_size * line_height)

        # Prepare warning font and measure warning block width
        warning_font = font
        max_warning_width = 0
        if warnings:
            warning_font_size = max(12, font_size - 4)
            warning_font = self._load_font(warning_font_size)
            warning_char_width = self._measure_char_width(warning_font, warning_font_size)
            label_width = len("WARNINGS:") * warning_char_width
            max_warning_width = label_width
            for _, msg in warnings:
                w = len(f"- {msg}") * warning_char_width
                max_warning_width = max(max_warning_width, w)

        text_width = int(max_line_length * char_width + padding * 2)
        warning_width = int(max_warning_width + padding * 2 + 20) if warnings else 0
        img_width = max(text_width, warning_width)
        img_height = int(len(lines) * line_height_px + padding * 2)

        # Add space for warnings at the bottom
        if warnings:
            img_height += (len(warnings) + 1) * line_height_px + padding

        # Create image
        img = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(img)
        #endregion

        #region R-text
        # Draw each character with its color
        x, y = padding, padding
        char_idx = 0

        for line in lines:
            x = padding
            for char in line:
                if char_idx < len(char_info):
                    info = char_info[char_idx]
                    color = self.get_color_for_depth(info['depth'], info['is_error'], info['is_warning'])

                    # Draw character
                    draw.text((x, y), char, fill=color, font=font)
                    x += char_width
                    char_idx += 1

            y += line_height_px
            # No char_idx increment for newline: parse_enclosures skips '\n' entries
        #endregion

        #region R-warn
        # Draw warnings at the bottom
        if warnings:
            y += padding
            draw.text((padding, y), "WARNINGS:", fill=self.error_color, font=warning_font)
            y += line_height_px

            for _, warning in warnings:
                draw.text((padding + 20, y), f"- {warning}", fill=self.warning_color, font=warning_font)
                y += line_height_px
        #endregion

        return img
    #endregion

    #region main
    def visualize(
        self,
        text: str,
        font_size: int = 24,
        line_height: float = 1.5,
        padding: int = 20,
        background_color: str = "dark",
    ) -> tuple[torch.Tensor, str, str]:
        """
        Main processing function that creates the visualization.

        Returns:
            - Image tensor in ComfyUI format [B, H, W, C]
            - Analysis report string
            - Issues text string (hanging parenthesis and deep nesting warnings)
        """

        # Parse the text
        char_info, warnings, max_depth = self.parse_enclosures(text)

        #region C-report
        # Create analysis report
        depth_counts: dict[int, int] = {}
        for info in char_info:
            depth = info['depth']
            if depth > 0:
                depth_counts[depth] = depth_counts.get(depth, 0) + 1

        report_lines = [
            "=== ENCLOSURE ANALYSIS ===",
            f"Total characters: {len(text)}",
            f"Max nesting depth: {max_depth}",
            ""
        ]

        if depth_counts:
            report_lines.append("Depth distribution:")
            for depth in sorted(depth_counts.keys()):
                report_lines.append(f"  Level {depth}: {depth_counts[depth]} characters")
            report_lines.append("")

        if warnings:
            report_lines.append(f"WARNINGS ({len(warnings)}):")
            for category, msg in warnings:
                report_lines.append(f"  - {msg}")
                logger.warning("[%s] %s", category, msg)
        else:
            report_lines.append("No issues found - all parenthesis properly matched!")

        analysis_report = "\n".join(report_lines)
        #endregion

        #region C-issues
        # Create issues text output (categorized by structured warning category)
        hanging_issues = [msg for cat, msg in warnings if cat == "hanging"]
        nesting_issues = [msg for cat, msg in warnings if cat == "nesting"]

        issues_lines: list[str] = []

        if hanging_issues:
            issues_lines.append("HANGING/UNCLOSED PARENTHESIS:")
            for issue in hanging_issues:
                issues_lines.append(f"  - {issue}")
            issues_lines.append("")

        if nesting_issues:
            issues_lines.append("MULTIPLE NESTED ENCLOSURES:")
            for issue in nesting_issues:
                issues_lines.append(f"  - {issue}")
            issues_lines.append("")

        if not issues_lines:
            issues_text = "No issues found."
        else:
            issues_text = "\n".join(issues_lines).strip()
        #endregion

        #region C-image
        # Create visualization image
        pil_image = self.create_visualization_image(
            text, char_info, warnings, font_size, line_height, padding, background_color
        )

        # Convert PIL Image to ComfyUI tensor format [B, H, W, C]
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension
        #endregion

        return (img_tensor, analysis_report, issues_text)
    #endregion

    #region ui
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "font_size": ("INT", {"default": 24, "min": 12, "max": 72, "step": 1}),
                "line_height": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "round": 0.1}),
                "padding": ("INT", {"default": 20, "min": 0, "max": 100, "step": 5}),
            },
            "optional": {
                "background_color": (["dark", "light"], {"default": "dark"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("visualization", "analysis_report", "issues_text")
    FUNCTION = "visualize"
    CATEGORY = "illumorae"
    OUTPUT_NODE = True
    DESCRIPTION = "Visualizes prompt text enclosure nesting and mismatched parenthesis, outputting an image and analysis strings."
    #endregion


#region reg
# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "illumoraeEnclosureVisualizerNode": illumoraeEnclosureVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeEnclosureVisualizerNode": "Enclosure Visualizer",
}
#endregion
