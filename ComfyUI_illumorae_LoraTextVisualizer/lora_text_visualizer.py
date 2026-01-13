import os
import re
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

class illumoraeLoraVisualizerWordPlotNode:
    """
    A ComfyUI node that analyzes a prompt text and creates a visual representation
    of LORA strengths, displaying them as text with varying sizes and brightness
    based on their strength values.
    """
    
    def __init__(self):
        self.output_width = 512
        self.output_height = 512
        self.bg_color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        
        # Try to find a suitable font
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]
        self.font_path = next((p for p in font_paths if os.path.exists(p)), None)
        if not self.font_path:
            print("Warning: No default font found. Text rendering may fail.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "illumorae"
    OUTPUT_NODE = True
    DESCRIPTION = "Creates a visual representation of LORA strengths from a text string."

    def extract_loras(self, text: str) -> List[Tuple[str, float]]:
        """Extract LORA names and their strengths from the prompt."""
        lora_pattern = re.compile(r'<lora:([^:]+):([^>]+)>')
        matches = lora_pattern.findall(text)
        
        # Convert strengths to floats and sort by strength descending
        loras = [(name, float(strength)) for name, strength in matches]
        return sorted(loras, key=lambda x: x[1], reverse=True)

    def create_visualization(self, loras: List[Tuple[str, float]], width: int, height: int) -> Image.Image:
        """Create a visual representation of LORA strengths."""
        if not loras:
            return self.no_lora_found_visualization(width, height)

        # Find max and min strengths for normalization
        max_strength = max(strength for _, strength in loras)
        min_strength = min(strength for _, strength in loras)
        strength_range = max_strength - min_strength

        # Compute word metrics only
        words_info = self.compute_word_infos(loras, min_strength, max_strength, strength_range)
        # Ideal packing/placement
        positions = self.ideal_pack_words(words_info, width, height)
        # Draw all words
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        for x, y, w, h, word in positions:
            font = ImageFont.truetype(self.font_path, size=max(10, int(word['font_size'] * word.get('scale_factor', 1.0))))
            text_color = word['color']
            draw.text((x, y), word['text'], fill=text_color, font=font)
        return img

    def ideal_pack_words(self, words_info, width, height):
        """Pack words into the image bounds using a greedy row-based algorithm, shrinking if needed."""
        # Sort by font size descending (largest first)
        words_info = sorted(words_info, key=lambda w: w['font_size'], reverse=True)
        positions = []
        margin = 4
        y_cursor = margin
        row_height = 0
        x_cursor = margin
        scale_factor = 1.0
        while True:
            positions.clear()
            y_cursor = margin
            row_height = 0
            x_cursor = margin
            fits = True
            for word in words_info:
                w = int(word['width'] * scale_factor)
                h = int(word['height'] * scale_factor)
                if x_cursor + w + margin > width:
                    x_cursor = margin
                    y_cursor += row_height + margin
                    row_height = 0
                if y_cursor + h + margin > height:
                    fits = False
                    break
                positions.append((x_cursor, y_cursor, w, h, {**word, 'scale_factor': scale_factor}))
                x_cursor += w + margin
                row_height = max(row_height, h)
            if fits:
                break
            scale_factor *= 0.92
            if scale_factor < 0.3:
                break
        return positions

    def no_lora_found_visualization(self, width: int, height: int) -> Image.Image:
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(self.font_path, size=36)
        text = "No LORAs found"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        draw.text((x, y), text, fill=self.text_color, font=font)
        return img

    def compute_word_infos(self, loras, min_strength, max_strength, strength_range):
        words_info = []
        for name, strength in loras:
            norm_strength = self.normalize_strength(strength, min_strength, strength_range)
            font_size = self.compute_font_size(norm_strength)
            opacity = self.compute_opacity(norm_strength)
            color = self.compute_color(norm_strength, opacity)
            font = ImageFont.truetype(self.font_path, size=font_size)
            text = name
            text_width, text_height = self.compute_word_bbox(font, text)
            words_info.append({
                'name': name,
                'font': font,
                'font_size': font_size,
                'opacity': opacity,
                'color': color,
                'text': text,
                'width': text_width,
                'height': text_height
            })
        return words_info

    def normalize_strength(self, strength, min_strength, strength_range):
        if strength_range > 0:
            return (strength - min_strength) / strength_range
        return 1.0

    def compute_font_size(self, norm_strength):
        return int(40 + norm_strength * 80)

    def compute_opacity(self, norm_strength):
        return int(128 + norm_strength * 127)

    def compute_color(self, norm_strength, opacity):
        color_val = int(128 + norm_strength * (255-128))
        return (color_val, color_val, color_val, opacity)

    def compute_word_bbox(self, font, text):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def process(self, text: str, width: int, height: int) -> tuple:
        """Process the input text and create a visualization of LORA strengths."""
        # Extract and sort LORAs
        loras = self.extract_loras(text)
        
        # Create visualization
        img = self.create_visualization(loras, width, height)
        
        # Convert to tensor format expected by ComfyUI
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0)
        
        return (img_tensor,)

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "illumoraeLoraVisualizerWordPlotNode": illumoraeLoraVisualizerWordPlotNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeLoraVisualizerWordPlotNode": "LoRA Text Strength Visualizer WordPlot",
}
