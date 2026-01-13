import cv2
import torch
import numpy as np

class illumorae_CLAHEImageNode:
    """
    A ComfyUI node that applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance the localized relative contrast of an image. 

    Default Settings:
      - clip_limit: 2.0
      - tile_grid_width: 8
      - tile_grid_height: 8

    How it works:
      1. The input image is expected in [B, H, W, C] format.
      2. For each image in the batch:
         - If the image is in float format (assumed in the range [0, 1]),
           it is converted to uint8 (range [0, 255]).
         - For RGB images (3 channels): The image is converted from RGB to LAB,
           CLAHE is applied to the L-channel, and then converted back to RGB.
         - For grayscale images (1 channel): CLAHE is applied directly.
         - The resulting image is converted back to float (range [0, 1]).
      3. The processed images are reassembled into a torch tensor with the same
         device as the input.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "clip_limit": ("FLOAT", {"default": 3.0, "min": 0.0}),
                "tile_grid_width": ("INT", {"default": 12, "min": 1}),
                "tile_grid_height": ("INT", {"default": 12, "min": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_image",)
    FUNCTION = "apply_clahe"
    CATEGORY = "illumorae"
    DESCRIPTION = (
        "Applies CLAHE Contrast Limited Adaptive Histogram Equalization to the input image. "
        "useful for locally spatial relative contrast enhancement."
    )

    def apply_clahe(self, input_image, clip_limit=2.0, tile_grid_width=8, tile_grid_height=8, **kwargs):
        """
        Applies CLAHE to each image in the input batch with configurable parameters.

        Parameters:
            input_image (torch.Tensor): A batch of images in [B, H, W, C] format.
                                         Images are expected to be in float format (range [0,1])
                                         or in uint8.
            clip_limit (float): Threshold for contrast limiting.
            tile_grid_width (int): Number of tiles in the horizontal direction.
            tile_grid_height (int): Number of tiles in the vertical direction.

        Returns:
            A tuple containing a single torch.Tensor with the processed images.
        """
        # Ensure the image is on CPU and convert to numpy array.
        input_image_cpu = input_image.cpu()
        image_batch_np = input_image_cpu.numpy()  # Shape: (B, H, W, C)

        # Create an empty list to store the processed images.
        processed_image_list = []

        # Define the CLAHE settings based on input parameters.
        clahe_tile_grid_size = (tile_grid_width, tile_grid_height)
        clahe_operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=clahe_tile_grid_size)

        # Process each image in the batch.
        for single_image in image_batch_np:
            # If the image is not in uint8, assume it's in float [0,1] and convert.
            if single_image.dtype != np.uint8:
                single_image_uint8 = (single_image * 255).clip(0, 255).astype(np.uint8)
            else:
                single_image_uint8 = single_image

            # Process RGB images.
            if single_image_uint8.shape[-1] == 3:
                # Convert from RGB to LAB.
                image_lab = cv2.cvtColor(single_image_uint8, cv2.COLOR_RGB2LAB)
                l_channel, a_channel, b_channel = cv2.split(image_lab)
                # Apply CLAHE to the L channel.
                l_channel_clahe = clahe_operator.apply(l_channel)
                # Merge channels back.
                image_lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
                # Convert from LAB back to RGB.
                processed_image_uint8 = cv2.cvtColor(image_lab_clahe, cv2.COLOR_LAB2RGB)

            # Process Grayscale images.
            elif single_image_uint8.shape[-1] == 1:
                # Apply CLAHE directly.
                image_gray = single_image_uint8[:, :, 0]
                processed_gray = clahe_operator.apply(image_gray)
                # Expand dimensions back to (H, W, 1).
                processed_image_uint8 = np.expand_dims(processed_gray, axis=-1)

            # For images with an unsupported number of channels, pass through.
            else:
                processed_image_uint8 = single_image_uint8

            # Convert the processed image back to float32 in the range [0, 1].
            processed_image_float = processed_image_uint8.astype(np.float32) / 255.0
            processed_image_list.append(processed_image_float)

        # Stack the processed images back into a numpy array.
        output_images_np = np.stack(processed_image_list, axis=0)

        # Convert the numpy array back to a torch tensor.
        output_images_tensor = torch.from_numpy(output_images_np)

        # Ensure the output tensor is on the same device as the input.
        output_images_tensor = output_images_tensor.to(input_image.device)

        return (output_images_tensor,)

# ComfyUI custom node classes to load 
NODE_CLASS_MAPPINGS = {
    "illumoraeImageCLAHENode": illumorae_CLAHEImageNode,
}

# ComfyUI display name for node
NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeImageCLAHENode": "Image CLAHE Contrast Limited Adaptive Histogram Equalization",
}
