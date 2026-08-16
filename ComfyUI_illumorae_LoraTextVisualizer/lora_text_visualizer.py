"""
TITLE::LoRA Text Strength Visualizer
DESCRIPTIONSHORT::Creates an image visualization of LoRA strengths from <lora:name:strength> tags in prompt text.
VERSION::20260816
IMAGE::comfyui_illumorae_lora_strength_wordplot.png
GROUP::Lora
GROUPORDER::4
LISTORDER::3
STATUS::working
"""
#region IMPORTS
import os
import re
import math
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
#endregion


#region CLASS
class illumoraeLoRATextStrengthVisualizerWordPlotNode:
    """
    A ComfyUI node that analyzes a prompt text and creates a visual representation
    of LORA strengths, displaying them as text with varying sizes and brightness
    based on their strength values.
    """

    #region CONFIG
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
            print("Warning: No TrueType font found; falling back to PIL default font.")
    #endregion

    #region CORE
    def process(self, text: str, width: int, height: int) -> Tuple[torch.Tensor]:
        """Process the input text and create a visualization of LORA strengths.

        Any unexpected error during extraction or rendering is caught and
        surfaced as an error-image tensor so the node never raises into
        ComfyUI's queue loop.
        """
        #region C_RENDER
        # Extract LoRA tags and render the visualization; fall back to an
        # error image on any unexpected failure.
        try:
            loras = self.extract_loras(text)
            img = self.create_visualization(loras, width, height)
        except Exception as exc:  # noqa: BLE001 - last-resort guard for the queue loop
            print(f"illumoraeLoRATextStrengthVisualizerWordPlotNode error: {exc!r}")
            img = self._error_visualization(width, height, "Visualizer error")
        #endregion

        #region C_TENSOR
        # Convert the PIL image to the ComfyUI IMAGE tensor format:
        # (1, H, W, 3) float32 in [0, 1].
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0)
        return (img_tensor,)
        #endregion
    #endregion

    #region C_ORCH
    def create_visualization(self, loras: List[Tuple[str, float]], width: int, height: int) -> Image.Image:
        """Create a visual representation of LORA strengths.

        Orchestrates the pipeline: normalize strengths, compute per-word
        metrics, pack words into rows, then draw. Falls back to a notice
        image when there are no LoRAs or when packing cannot fit all words.
        """
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
        if not positions:
            # No scale factor fit all words; render an overflow notice.
            return self.too_many_loras_visualization(width, height)
        # Draw all words
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        for x, y, w, h, word in positions:
            font = self._load_font(size=max(10, int(word['font_size'] * word.get('scale_factor', 1.0))))
            text_color = word['color']
            draw.text((x, y), word['text'], fill=text_color, font=font)
        return img
    #endregion

    #region C_PARSE
    def extract_loras(self, text: str) -> List[Tuple[str, float]]:
        """Extract LORA names and their strengths from the prompt.

        Tolerates case variants (``<LORA:...>``, ``<LoRA:...>``) and
        whitespace inside the brackets, matching ComfyUI's own parsing.
        Tags with a non-finite or non-numeric strength are skipped rather
        than crashing the node.
        """
        lora_pattern = re.compile(r'<\s*lora\s*:\s*([^:]+?)\s*:\s*([^>]+?)\s*>', re.IGNORECASE)
        matches = lora_pattern.findall(text)

        # Convert strengths to floats; skip malformed / non-finite values.
        loras: List[Tuple[str, float]] = []
        for name, raw in matches:
            try:
                strength = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(strength):
                continue
            loras.append((name, strength))
        return sorted(loras, key=lambda x: x[1], reverse=True)
    #endregion

    #region C_COMPUTE
    def compute_word_infos(self, loras, min_strength, max_strength, strength_range):
        """Build per-word metric dicts (font size, color, bbox) for packing."""
        words_info = []
        for name, strength in loras:
            norm_strength = self.normalize_strength(strength, min_strength, strength_range)
            font_size = self.compute_font_size(norm_strength)
            color = self.compute_color(norm_strength)
            font = self._load_font(size=font_size)
            text = name
            text_width, text_height = self.compute_word_bbox(font, text)
            words_info.append({
                'name': name,
                'font_size': font_size,
                'color': color,
                'text': text,
                'width': text_width,
                'height': text_height
            })
        return words_info

    def normalize_strength(self, strength, min_strength, strength_range):
        """Map a strength to [0, 1] relative to the min/max of the set."""
        if strength_range > 0:
            return (strength - min_strength) / strength_range
        return 1.0

    def compute_font_size(self, norm_strength):
        """Font size in the range 40..120 pt."""
        return int(40 + norm_strength * 80)

    def compute_color(self, norm_strength):
        # Gray brightness is the only visual channel; the canvas is RGB so an
        # alpha component would be dropped. Range 128..255.
        color_val = int(128 + norm_strength * (255 - 128))
        return (color_val, color_val, color_val)

    def compute_word_bbox(self, font, text):
        """Return (width, height) of the text's bounding box for the font."""
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    #endregion

    #region C_PACK
    def ideal_pack_words(self, words_info, width, height):
        """Pack words into the image bounds using a greedy row-based algorithm, shrinking if needed.

        Returns a list of ``(x, y, w, h, word)`` tuples. If no scale factor
        in ``[0.3, 1.0]`` fits every word, returns an empty list so the
        caller can render an overflow fallback instead of a partial layout.
        """
        # Sort by font size descending (largest first)
        words_info = sorted(words_info, key=lambda w: w['font_size'], reverse=True)
        positions = []
        margin = 4
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
                # Could not fit all words even at the minimum scale; signal
                # overflow by returning an empty placement list.
                positions.clear()
                break
        return positions
    #endregion

    #region FALLBACK
    def no_lora_found_visualization(self, width: int, height: int) -> Image.Image:
        """Rendered when the input text contains no LoRA tags."""
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        font = self._load_font(size=36)
        text = "No LORAs found"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        draw.text((x, y), text, fill=self.text_color, font=font)
        return img

    def too_many_loras_visualization(self, width: int, height: int) -> Image.Image:
        """Fallback rendered when packing cannot fit all words even at min scale."""
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        font = self._load_font(size=24)
        text = "Too many LORAs to fit"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        draw.text((x, y), text, fill=self.text_color, font=font)
        return img

    def _error_visualization(self, width: int, height: int, message: str) -> Image.Image:
        """Render a single-line error message centered on the canvas."""
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        font = self._load_font(size=24)
        bbox = draw.textbbox((0, 0), message, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = max(0, (width - text_width) // 2)
        y = max(0, (height - text_height) // 2)
        draw.text((x, y), message, fill=self.text_color, font=font)
        return img
    #endregion

    #region UTIL
    def _load_font(self, size: int):
        """Return a font for the requested size.

        Uses the discovered TrueType path when available, otherwise falls
        back to PIL's bundled bitmap font (``ImageFont.load_default()``),
        which ignores ``size`` but still supports ``getbbox``/``text``.
        """
        if self.font_path:
            return ImageFont.truetype(self.font_path, size=size)
        return ImageFont.load_default()
    #endregion

    #region UI
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
    DESCRIPTION = "Creates an image visualization of LoRA strengths from <lora:name:strength> tags in prompt text."
    #endregion
#endregion


#region REGISTER
# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "illumoraeLoRATextStrengthVisualizerWordPlotNode": illumoraeLoRATextStrengthVisualizerWordPlotNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeLoRATextStrengthVisualizerWordPlotNode": "LoRA Text Strength Visualizer WordPlot",
}
#endregion
