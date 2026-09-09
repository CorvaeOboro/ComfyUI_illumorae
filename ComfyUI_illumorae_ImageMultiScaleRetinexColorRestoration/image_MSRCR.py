"""
Image Retinex Enhancement

TITLE::Image Retinex Enhancement (MSRCR)
DESCRIPTIONSHORT::Applies Multi-Scale Retinex with Color Restoration (MSRCR) to enhance dynamic range and color constancy.
VERSION::20260816
IMAGE::comfyui_illumorae_image_retinex_enhancement_msrcr.png
GROUP::Image Adjustment
GROUPORDER::2
LISTORDER::20
STATUS::working
"""
import cv2
import numpy as np
import torch


class illumoraeImageRetinexEnhancementNode:
    """
    A ComfyUI node that applies Multi-Scale Retinex with Color Restoration (MSRCR)
    to enhance the dynamic range and color constancy of input images.

    This implementation follows the MSRCR algorithm as described in the Retinex literature
    (Jobson, Rahman, and Woodell, "A Multiscale Retinex for Bridging the Gap Between 
    Color Images and the Human Observation of Scenes," IEEE Transactions on Image 
    Processing, 1997) 

    Parameters ( with corresponding original MSRCR notation):
        - gaussian_sigma_small, gaussian_sigma_medium, gaussian_sigma_large: sigma1, sigma2, sigma3 (Gaussian blur scales)
        - color_restoration_strength: alpha (alpha parameter, color restoration strength)
        - clip_percent: symmetric percentile clip for the simplest-color-balance normalization

    The classic MSRCR gain (G), output offset (b), and color-restoration gain
    (beta) are not exposed. They are scalar gain/offset terms applied before a
    linear stretch, so they cancel under any min-max or percentile stretch and
    have no effect on the output. beta is fixed at the canonical 46.0 inside the
    color restoration factor for reference; alpha is kept because its log10(alpha)
    constant shifts the balance between the MSR and color-restoration terms and
    does affect the result.

    Processing Pipeline:
        1. Input images [0,1] -> scaled to [1,256] (adding 1.0 to avoid log(0))
        2. Apply Multi-Scale Retinex using log10 operations
        3. Compute color restoration factor for color images
        4. Combine MSR and color restoration results
        5. Apply simplest color balance: percentile-clip then linear stretch per channel to [0,255]
        6. Convert back to [0,1] range for output

    The node processes images in batches (shape: [B, H, W, C]) and supports both
    color (multi-channel) and grayscale (single channel) images. For grayscale images,
    only Multi-Scale Retinex is applied without color restoration.

    This algorithm is particularly effective for challenging lighting conditions,
    enhancing details in both shadows and highlights while maintaining natural colors
    and preventing overexposure artifacts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "gaussian_sigma_small": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 500.0, "step": 0.1},
                ),
                "gaussian_sigma_medium": (
                    "FLOAT",
                    {"default": 80.0, "min": 0.1, "max": 500.0, "step": 0.1},
                ),
                "gaussian_sigma_large": (
                    "FLOAT",
                    {"default": 250.0, "min": 0.1, "max": 500.0, "step": 0.1},
                ),
                "color_restoration_strength": (
                    "FLOAT",
                    {"default": 125.0, "min": 0.1, "max": 500.0, "step": 0.1},
                ),
                "clip_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 49.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced_image",)
    FUNCTION = "apply_retinex_enhancement"
    CATEGORY = "illumorae"
    DESCRIPTION = "Applies Multi-Scale Retinex with Color Restoration (MSRCR) to enhance dynamic range and color constancy."

    def apply_retinex_enhancement(
        self, input_image, gaussian_sigma_small, gaussian_sigma_medium, gaussian_sigma_large,
        color_restoration_strength, clip_percent
    ):
        """
        Applies the MSRCR algorithm to each image in the batch.

        Parameters:
            input_image (torch.Tensor): Batch of images in [B, H, W, C] format, with pixel values in [0, 1].
            gaussian_sigma_small (float): Sigma value for the smallest scale of Gaussian blurring.
            gaussian_sigma_medium (float): Sigma value for the medium scale of Gaussian blurring.
            gaussian_sigma_large (float): Sigma value for the largest scale of Gaussian blurring.
            color_restoration_strength (float): Alpha parameter - controls the strength of the color restoration.
            clip_percent (float): Symmetric percentile clip (percent) for the simplest-color-balance normalization.

        Returns:
            Tuple containing a single torch.Tensor of enhanced images.
        """
        # Move tensor to CPU and convert to numpy array.
        input_image_cpu = input_image.cpu()
        image_batch_np = input_image_cpu.numpy()  # Expected shape: (B, H, W, C)

        # List to accumulate enhanced images.
        enhanced_images_list = []

        # Define sigma scales based on node parameters.
        gaussian_sigma_scales = [gaussian_sigma_small, gaussian_sigma_medium, gaussian_sigma_large]

        # Process each image in the batch individually.
        for single_image in image_batch_np:
            # Retrieve image dimensions.
            image_height, image_width, num_channels = single_image.shape

            # Convert image to [0, 255] range and add 1.0 to avoid log(0)
            image_scaled = single_image.astype(np.float64) * 255.0 + 1.0

            if num_channels > 1:
                # --- Process Color Images with MSRCR ---
                enhanced_image = self.apply_multi_scale_retinex_color_restoration(
                    image_scaled, gaussian_sigma_scales,
                    color_restoration_strength, clip_percent
                )
            else:
                # --- Process Grayscale Images with MSR only ---
                enhanced_image = self.apply_multi_scale_retinex_grayscale(
                    image_scaled, gaussian_sigma_scales, clip_percent
                )

            # Append the enhanced image (converted to float32) to the list.
            enhanced_images_list.append(enhanced_image.astype(np.float32))

        # Stack the enhanced images back into a single numpy array.
        output_images_np = np.stack(enhanced_images_list, axis=0)

        # Convert the numpy array back to a torch tensor and place it on the original device.
        output_images_tensor = torch.from_numpy(output_images_np).to(input_image.device)

        return (output_images_tensor,)
    
    def compute_single_scale_retinex_transformation(self, input_image, gaussian_blur_sigma):
        """Apply Single Scale Retinex transformation using logarithmic base-10 operations."""
        gaussian_blurred_image = cv2.GaussianBlur(input_image, (0, 0), gaussian_blur_sigma)
        return np.log10(input_image) - np.log10(gaussian_blurred_image)
    
    def compute_multi_scale_retinex_transformation(self, input_image, gaussian_sigma_list):
        """Apply Multi Scale Retinex by averaging Single Scale Retinex results across multiple scales."""
        accumulated_retinex_result = np.zeros_like(input_image, dtype=np.float64)
        for current_gaussian_sigma in gaussian_sigma_list:
            accumulated_retinex_result += self.compute_single_scale_retinex_transformation(input_image, current_gaussian_sigma)
        return accumulated_retinex_result / len(gaussian_sigma_list)
    
    def compute_color_restoration_factor(self, input_image, color_restoration_alpha, color_restoration_beta):
        """Apply color restoration factor computation using logarithmic channel summation."""
        channel_sum_image = np.sum(input_image, axis=2, keepdims=True)
        return color_restoration_beta * (np.log10(color_restoration_alpha * input_image) - np.log10(channel_sum_image))
    
    def apply_color_balance_enhancement(self, input_image, clip_percent=1.0):
        """Apply simplest color balance: percentile-clip then linear stretch per channel.

        Clips each channel to its [clip_percent, 100 - clip_percent] percentile
        range, then linearly stretches that range to [0, 255]. This is the IPOL
        "Simplest Color Balance" approach. A raw min-max stretch is avoided
        because the MSRCR distribution has a long outlier tail that would cram
        the bulk of pixels near 255.
        """
        is_2d = input_image.ndim == 2
        img = input_image[:, :, np.newaxis] if is_2d else input_image
        low_percentile = clip_percent
        high_percentile = 100.0 - clip_percent
        for channel_index in range(img.shape[2]):
            current_channel = img[:, :, channel_index]
            low_clip_value = np.percentile(current_channel, low_percentile)
            high_clip_value = np.percentile(current_channel, high_percentile)
            if high_clip_value > low_clip_value:
                clipped_channel = np.clip(current_channel, low_clip_value, high_clip_value)
                img[:, :, channel_index] = (clipped_channel - low_clip_value) / (high_clip_value - low_clip_value) * 255.0
            else:
                img[:, :, channel_index] = 0
        return img[:, :, 0] if is_2d else img

    def apply_multi_scale_retinex_color_restoration(self, input_image, gaussian_sigma_list, color_restoration_alpha, clip_percent):
        """Apply complete Multi-Scale Retinex with Color Restoration algorithm."""
        # Compute Multi-Scale Retinex transformation
        multi_scale_retinex_result = self.compute_multi_scale_retinex_transformation(input_image, gaussian_sigma_list)

        # Compute Color Restoration factor. beta is fixed at the canonical 46.0;
        # it is a scalar multiplier on the CRF and cancels under the percentile
        # stretch below, so it is not exposed as a node parameter.
        color_restoration_result = self.compute_color_restoration_factor(input_image, color_restoration_alpha, 46.0)

        # Combine Multi-Scale Retinex and Color Restoration multiplicatively
        msrcr_combined_result = multi_scale_retinex_result * color_restoration_result

        # Simplest color balance: percentile-clip then linear stretch to [0, 255]
        color_balanced_result = self.apply_color_balance_enhancement(msrcr_combined_result, clip_percent)

        # Convert back to [0, 1] range for final output
        return color_balanced_result.astype(np.float64) / 255.0
    
    def apply_multi_scale_retinex_grayscale(self, input_image, gaussian_sigma_list, clip_percent=1.0):
        """Apply Multi-Scale Retinex transformation to grayscale images."""
        # Extract single channel for processing
        grayscale_channel = input_image[:, :, 0] if len(input_image.shape) == 3 else input_image

        # Apply Multi-Scale Retinex transformation
        accumulated_msr_result = np.zeros_like(grayscale_channel, dtype=np.float64)
        for current_gaussian_sigma in gaussian_sigma_list:
            gaussian_blurred_channel = cv2.GaussianBlur(grayscale_channel, (0, 0), current_gaussian_sigma)
            accumulated_msr_result += np.log10(grayscale_channel) - np.log10(gaussian_blurred_channel)

        averaged_msr_result = accumulated_msr_result / len(gaussian_sigma_list)

        # Simplest color balance: percentile-clip then linear stretch to [0, 255]
        balanced_channel = self.apply_color_balance_enhancement(averaged_msr_result, clip_percent)

        # Convert back to [0, 1] range
        final_grayscale_result = balanced_channel.astype(np.float64) / 255.0

        # Expand dimensions to restore channel dimension if needed
        if len(input_image.shape) == 3:
            final_grayscale_result = np.expand_dims(final_grayscale_result, axis=2)

        return final_grayscale_result


# ComfyUI custom node classes to load 
NODE_CLASS_MAPPINGS = {
    "illumoraeImageRetinexEnhancementNode": illumoraeImageRetinexEnhancementNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeImageRetinexEnhancementNode": "Image Retinex Enhancement",
}
