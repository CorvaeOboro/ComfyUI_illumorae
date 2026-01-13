"""
TITLE::Enclosure Visualizer
DESCRIPTIONSHORT::Visualizes prompt text enclosure nesting and mismatched parenthesis, outputting an image and analysis strings.
VERSION::20260113
IMAGE::comfyui_illumorae_text_enclosure_visualize.png
GROUP::Text
"""
import re
import random
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

class illumoraeTextEnclosureVisualizerNode:
    """
    A ComfyUI node that visualizes nested parenthesis in prompt text.
    
    Features:
    - Color-codes nested parenthesis levels with muted colors
    - Detects and highlights hanging/mismatched parenthesis in red
    - Warns about multiple layered enclosures
    - Outputs an image showing the text structure
    """

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
            "hidden": {},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("visualization", "analysis_report", "issues_text")
    FUNCTION = "visualize"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Visualizes nested parenthesis in prompt text with color coding and error detection."

    def parse_enclosures(self, text: str) -> Tuple[List[Dict], List[str]]:
        """
        Parse text to identify nested parenthesis and their depths.
        
        Returns:
            - List of character info dicts with position, char, depth, and color
            - List of warning messages
        """
        char_info = []
        depth_stack = []
        warnings = []
        max_depth = 0
        
        for i, char in enumerate(text):
            depth = len(depth_stack)
            
            if char == '(':
                depth_stack.append(i)
                depth = len(depth_stack)
                max_depth = max(max_depth, depth)
                
                # Check for multiple layered enclosures (depth > 3)
                if depth > 3:
                    warnings.append(f"Deep nesting at position {i}: depth {depth} (not preferred)")
                
                char_info.append({
                    'pos': i,
                    'char': char,
                    'depth': depth,
                    'is_error': False,
                    'is_warning': depth > 3
                })
                
            elif char == ')':
                if not depth_stack:
                    # Hanging closing parenthesis
                    warnings.append(f"Hanging closing parenthesis at position {i}")
                    char_info.append({
                        'pos': i,
                        'char': char,
                        'depth': 0,
                        'is_error': True,
                        'is_warning': False
                    })
                else:
                    depth_stack.pop()
                    depth = len(depth_stack) + 1  # Color based on the level it closes
                    char_info.append({
                        'pos': i,
                        'char': char,
                        'depth': depth,
                        'is_error': False,
                        'is_warning': False
                    })
            else:
                char_info.append({
                    'pos': i,
                    'char': char,
                    'depth': depth,
                    'is_error': False,
                    'is_warning': False
                })
        
        # Check for unclosed parenthesis
        if depth_stack:
            for pos in depth_stack:
                warnings.append(f"Unclosed opening parenthesis at position {pos}")
                # Mark as error
                for info in char_info:
                    if info['pos'] == pos:
                        info['is_error'] = True
        
        return char_info, warnings

    def get_color_for_depth(self, depth: int, is_error: bool, is_warning: bool) -> Tuple[int, int, int]:
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

    def create_visualization_image(
        self,
        text: str,
        char_info: List[Dict],
        warnings: List[str],
        font_size: int,
        line_height: float,
        padding: int,
        background_color: str
    ) -> Image.Image:
        """Create a PIL Image with color-coded text visualization."""
        
        # Set background color
        if background_color == "dark":
            bg_color = (30, 30, 35)
        else:
            bg_color = (240, 240, 245)
        
        # Try to load a monospace font, fallback to default
        try:
            font = ImageFont.truetype("consola.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("cour.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Calculate image dimensions
        lines = text.split('\n')
        max_line_length = max(len(line) for line in lines) if lines else 1
        
        # Estimate character width (monospace)
        char_width = font_size * 0.6
        line_height_px = int(font_size * line_height)
        
        img_width = int(max_line_length * char_width + padding * 2)
        img_height = int(len(lines) * line_height_px + padding * 2)
        
        # Add space for warnings at the bottom
        if warnings:
            img_height += len(warnings) * line_height_px + padding
        
        # Create image
        img = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(img)
        
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
            char_idx += 1  # Account for newline
        
        # Draw warnings at the bottom
        if warnings:
            y += padding
            warning_font_size = max(12, font_size - 4)
            try:
                warning_font = ImageFont.truetype("consola.ttf", warning_font_size)
            except:
                warning_font = font
            
            draw.text((padding, y), "WARNINGS:", fill=self.error_color, font=warning_font)
            y += line_height_px
            
            for warning in warnings:
                draw.text((padding + 20, y), f"- {warning}", fill=self.warning_color, font=warning_font)
                y += line_height_px
        
        return img

    def visualize(
        self,
        text: str,
        font_size: int = 24,
        line_height: float = 1.5,
        padding: int = 20,
        background_color: str = "dark",
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, str, str]:
        """
        Main processing function that creates the visualization.
        
        Returns:
            - Image tensor in ComfyUI format [B, H, W, C]
            - Analysis report string
            - Issues text string (hanging parenthesis and deep nesting warnings)
        """
        
        # Parse the text
        char_info, warnings = self.parse_enclosures(text)
        
        # Create analysis report
        depth_counts = {}
        for info in char_info:
            depth = info['depth']
            if depth > 0:
                depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        report_lines = [
            "=== ENCLOSURE ANALYSIS ===",
            f"Total characters: {len(text)}",
            f"Max nesting depth: {max(depth_counts.keys()) if depth_counts else 0}",
            ""
        ]
        
        if depth_counts:
            report_lines.append("Depth distribution:")
            for depth in sorted(depth_counts.keys()):
                report_lines.append(f"  Level {depth}: {depth_counts[depth]} characters")
            report_lines.append("")
        
        if warnings:
            report_lines.append(f"WARNINGS ({len(warnings)}):")
            for warning in warnings:
                report_lines.append(f"  - {warning}")
                print(f"[EnclosureVisualizer] WARNING: {warning}")
        else:
            report_lines.append("No issues found - all parenthesis properly matched!")
        
        analysis_report = "\n".join(report_lines)
        
        # Create issues text output
        issues_lines = []
        hanging_issues = []
        nesting_issues = []
        
        for warning in warnings:
            if "Hanging" in warning or "Unclosed" in warning:
                hanging_issues.append(warning)
            elif "Deep nesting" in warning or "depth" in warning:
                nesting_issues.append(warning)
        
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
        
        # Create visualization image
        pil_image = self.create_visualization_image(
            text, char_info, warnings, font_size, line_height, padding, background_color
        )
        
        # Convert PIL Image to ComfyUI tensor format [B, H, W, C]
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension
        
        return (img_tensor, analysis_report, issues_text)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "illumoraeEnclosureVisualizerNode": illumoraeTextEnclosureVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeEnclosureVisualizerNode": "Enclosure Visualizer",
}
