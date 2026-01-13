import os
import sys
import torch
from comfy.utils import ProgressBar, common_upscale

# Define a maximum resolution constant for widget limits
MAX_RESOLUTION = 4096

class illumoraeImageResizeMatteVideo:
    """
    A ComfyUI node that resizes an input image to a target width and height.
    The final dimensions are determined by the following priority:
      1. If a size reference image ('get_image_size') is provided, its dimensions override the widget values.
      2. If override values for width and/or height ('width_input' and 'height_input') are provided, they override the widget values.
      3. Otherwise, the widget values (width and height) are used.

    Optionally, the node can:
      - Preserve the image's aspect ratio.
      - Adjust the output dimensions to be divisible by a specified divisor.
      - Composite the resized image onto a black matte canvas of given dimensions.
    """
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "upscale_method": (cls.upscale_methods,{"default": "lanczos"}),
                "keep_proportion": ("BOOLEAN", {"default": False}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1}),
                "apply_matte": ("BOOLEAN", {"default": False}),
                "matte_width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "matte_height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
            },
            "optional": {
                "width_input": ("INT", {"forceInput": True}),
                "height_input": ("INT", {"forceInput": True}),
                "get_image_size": ("IMAGE",),
                "crop": (["disabled", "center"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "illumorae"
    DESCRIPTION = """
Resizes the input image to a specified width and height.
The target size is determined by the following order of precedence:
  1. Use dimensions from get_image_size (if provided).
  2. Use width_input and height_input (if provided).
  3. Use the widget width and height values.

Optionally, the node can preserve the image's aspect ratio and adjust dimensions so they are divisible by a specified number.
If apply_matte is enabled, the resized image is composited onto a black matte canvas of size matte_width x matte_height.
    """

    def resize(
        self,
        image,
        width,
        height,
        upscale_method,
        keep_proportion,
        divisible_by,
        apply_matte,
        matte_width,
        matte_height,
        width_input=None,
        height_input=None,
        get_image_size=None,
        crop="disabled"
    ):
        if isinstance(image, torch.Tensor) and image.dim() == 3:
            image = image.unsqueeze(0)

        if isinstance(get_image_size, torch.Tensor) and get_image_size.dim() == 3:
            get_image_size = get_image_size.unsqueeze(0)

        # Step 1: Retrieve original image dimensions.
        # Expected input format: [batch_size, original_height, original_width, num_channels]
        batch_size, original_height, original_width, num_channels = image.shape

        size_reference_image = get_image_size
        maintain_aspect_ratio = keep_proportion
        divisor = divisible_by
        input_image = image
        upscale_method_name = upscale_method
        crop_mode = None if crop == "disabled" else crop
        compose_matte = apply_matte
        matte_canvas_width = matte_width
        matte_canvas_height = matte_height

        # Map widget and override logic
        widget_target_width = width
        widget_target_height = height

        # Use override values only if provided, otherwise fallback to widget values
        if width_input is not None:
            widget_target_width = width_input
        if height_input is not None:
            widget_target_height = height_input

        # Step 3: If a size reference image is provided, override dimensions with its size.
        if size_reference_image is not None:
            _, reference_image_height, reference_image_width, _ = size_reference_image.shape
            widget_target_width = reference_image_width
            widget_target_height = reference_image_height

        # Step 4: Adjust dimensions to preserve the aspect ratio if requested.
        if maintain_aspect_ratio and size_reference_image is None:
            # When only height is provided (width is zero)
            if widget_target_width == 0 and widget_target_height != 0:
                aspect_ratio = widget_target_height / original_height
                widget_target_width = round(original_width * aspect_ratio)
            # When only width is provided (height is zero)
            elif widget_target_height == 0 and widget_target_width != 0:
                aspect_ratio = widget_target_width / original_width
                widget_target_height = round(original_height * aspect_ratio)
            # When both dimensions are provided and non-zero
            elif widget_target_width != 0 and widget_target_height != 0:
                scaling_ratio = min(widget_target_width / original_width, widget_target_height / original_height)
                widget_target_width = round(original_width * scaling_ratio)
                widget_target_height = round(original_height * scaling_ratio)
        else:
            # Use original dimensions if any dimension is zero
            if widget_target_width == 0:
                widget_target_width = original_width
            if widget_target_height == 0:
                widget_target_height = original_height

        # Step 5: Adjust dimensions to be divisible by a specified divisor if applicable.
        if divisor > 1 and size_reference_image is None:
            widget_target_width = widget_target_width - (widget_target_width % divisor)
            widget_target_height = widget_target_height - (widget_target_height % divisor)

        # Step 6: Convert the image format from [B, H, W, C] to [B, C, H, W] for torch interpolation.
        # [Batch_size, Height, Width, num_Channels]
        image_in_tensor_format = input_image.movedim(-1, 1)

        # Step 7: Resize the image using the specified upscale method and crop mode.
        resized_tensor_image = common_upscale(image_in_tensor_format, widget_target_width, widget_target_height, upscale_method_name, crop_mode)

        # Step 8: Convert the resized image back to [B, H, W, C] format.
        # [Batch_size, Height, Width, num_Channels]
        resized_tensor_image = resized_tensor_image.movedim(1, -1)

        # Step 9: Update the final dimensions from the resized image.
        final_resized_height = resized_tensor_image.shape[1]
        final_resized_width = resized_tensor_image.shape[2]

        # Step 10: If matte composition is enabled, composite the resized image onto a black matte canvas.
        if compose_matte:
            # If the resized image is larger than the matte canvas, center-crop the resized image.
            if final_resized_width > matte_canvas_width or final_resized_height > matte_canvas_height:
                crop_start_coordinate_x = 0
                crop_start_coordinate_y = 0

                if final_resized_width > matte_canvas_width:
                    crop_start_coordinate_x = (final_resized_width - matte_canvas_width) // 2

                if final_resized_height > matte_canvas_height:
                    crop_start_coordinate_y = (final_resized_height - matte_canvas_height) // 2

                resized_tensor_image = resized_tensor_image[
                    :,
                    crop_start_coordinate_y:crop_start_coordinate_y + min(final_resized_height, matte_canvas_height),
                    crop_start_coordinate_x:crop_start_coordinate_x + min(final_resized_width, matte_canvas_width),
                    :
                ]

                final_resized_height = resized_tensor_image.shape[1]
                final_resized_width = resized_tensor_image.shape[2]

            # Create a black matte canvas image with the desired dimensions.
            matte_canvas_image = torch.zeros(
                (batch_size, matte_canvas_height, matte_canvas_width, num_channels),
                dtype=resized_tensor_image.dtype,
                device=resized_tensor_image.device
            )

            # Calculate offsets to center the resized image on the matte canvas.
            offset_coordinate_x = (matte_canvas_width - final_resized_width) // 2
            offset_coordinate_y = (matte_canvas_height - final_resized_height) // 2

            # Place the resized image onto the matte canvas.
            matte_canvas_image[
                :,
                offset_coordinate_y:offset_coordinate_y + final_resized_height,
                offset_coordinate_x:offset_coordinate_x + final_resized_width,
                :
            ] = resized_tensor_image

            # Update the resized image and its dimensions.
            resized_tensor_image = matte_canvas_image
            final_resized_width = matte_canvas_width
            final_resized_height = matte_canvas_height

        # Step 11: Return the final resized image along with its width and height.
        return (resized_tensor_image, final_resized_width, final_resized_height)

# ComfyUI custom node classes to load
NODE_CLASS_MAPPINGS = {
    "illumoraeImageResizeMatteVideoNode": illumoraeImageResizeMatteVideo,
}

# ComfyUI display name for node
NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeImageResizeMatteVideoNode": "Image Resize w Matte ",
}
