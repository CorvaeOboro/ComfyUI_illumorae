"""
TITLE::Image Retinex Enhancement (MSRCR)
DESCRIPTIONSHORT::Applies Multi-Scale Retinex with Color Restoration (MSRCR) to enhance dynamic range and color constancy.
VERSION::20260113
IMAGE::comfyui_illumorae_image_retinex_enhancement_msrcr.png
GROUP::Image
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
        - gaussian_sigma_small, gaussian_sigma_medium, gaussian_sigma_large: σ₁, σ₂, σ₃ (Gaussian blur scales)
        - color_restoration_strength: α (alpha parameter, color restoration strength)
        - color_restoration_offset: β (beta parameter, color restoration offset)
        - output_gain: G (gain parameter, output multiplier)
        - output_offset: b (offset parameter, output offset)

    Processing Pipeline:
        1. Input images [0,1] → scaled to [1,256] (adding 1.0 to avoid log(0))
        2. Apply Multi-Scale Retinex using log10 operations
        3. Compute color restoration factor for color images
        4. Combine MSR and color restoration results
        5. Normalize each channel to [0,255] range
        6. Apply color balance enhancement with percentile clipping
        7. Convert back to [0,1] range for output

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
                    {"default": 12.0, "min": 0.1, "max": 500.0, "step": 0.1},
                ),
                "gaussian_sigma_medium": (
                    "FLOAT",
                    {"default": 60.0, "min": 0.1, "max": 500.0, "step": 0.1},
                ),
                "gaussian_sigma_large": (
                    "FLOAT",
                    {"default": 180.0, "min": 0.1, "max": 500.0, "step": 0.1},
                ),
                "color_restoration_strength": (
                    "FLOAT",
                    {"default": 100.0, "min": 0.1, "max": 500.0, "step": 0.1},
                ),
                "color_restoration_offset": (
                    "FLOAT",
                    {"default": 25.0, "min": 0.1, "max": 500.0, "step": 0.1},
                ),
                "output_gain": (
                    "FLOAT",
                    {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "output_offset": (
                    "FLOAT",
                    {"default": -0.5, "min": -100.0, "max": 100.0, "step": 0.1},
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
        color_restoration_strength, color_restoration_offset, output_gain, output_offset
    ):
        """
        Applies the MSRCR algorithm to each image in the batch.

        Parameters:
            input_image (torch.Tensor): Batch of images in [B, H, W, C] format, with pixel values in [0, 1].
            gaussian_sigma_small (float): Sigma value for the smallest scale of Gaussian blurring.
            gaussian_sigma_medium (float): Sigma value for the medium scale of Gaussian blurring.
            gaussian_sigma_large (float): Sigma value for the largest scale of Gaussian blurring.
            color_restoration_strength (float): Alpha parameter - controls the strength of the color restoration.
            color_restoration_offset (float): Beta parameter - controls the offset of the color restoration.
            output_gain (float): G parameter - gain multiplier applied to the final result.
            output_offset (float): b parameter - offset added to the MSRCR result.

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
                    color_restoration_strength, color_restoration_offset, 
                    output_gain, output_offset
                )
            else:
                # --- Process Grayscale Images with MSR only ---
                enhanced_image = self.apply_multi_scale_retinex_grayscale(
                    image_scaled, gaussian_sigma_scales
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
    
    def apply_color_balance_enhancement(self, input_image, low_percentile_clip=0.01, high_percentile_clip=0.99):
        """Apply color balance for contrast enhancement using percentile-based clipping."""
        total_pixel_count = input_image.shape[0] * input_image.shape[1]
        for channel_index in range(input_image.shape[2]):
            unique_values, pixel_counts = np.unique(input_image[:, :, channel_index], return_counts=True)
            cumulative_pixel_count = 0
            low_clip_value = unique_values[0]
            high_clip_value = unique_values[-1]
            
            for unique_value, pixel_count in zip(unique_values, pixel_counts):
                if float(cumulative_pixel_count) / total_pixel_count < low_percentile_clip:
                    low_clip_value = unique_value
                if float(cumulative_pixel_count) / total_pixel_count < high_percentile_clip:
                    high_clip_value = unique_value
                cumulative_pixel_count += pixel_count
            
            input_image[:, :, channel_index] = np.maximum(np.minimum(input_image[:, :, channel_index], high_clip_value), low_clip_value)
        
        return input_image
    
    def apply_multi_scale_retinex_color_restoration(self, input_image, gaussian_sigma_list, color_restoration_alpha, color_restoration_beta, output_gain_multiplier, output_offset_value):
        """Apply complete Multi-Scale Retinex with Color Restoration algorithm."""
        # Compute Multi-Scale Retinex transformation
        multi_scale_retinex_result = self.compute_multi_scale_retinex_transformation(input_image, gaussian_sigma_list)
        
        # Compute Color Restoration factor
        color_restoration_result = self.compute_color_restoration_factor(input_image, color_restoration_alpha, color_restoration_beta)
        
        # Combine Multi-Scale Retinex and Color Restoration using addition operation
        msrcr_combined_result = output_gain_multiplier * (multi_scale_retinex_result * color_restoration_result + output_offset_value)
        
        # Normalize each channel independently to [0, 255] range
        for channel_index in range(msrcr_combined_result.shape[2]):
            current_channel = msrcr_combined_result[:, :, channel_index]
            channel_minimum_value = np.min(current_channel)
            channel_maximum_value = np.max(current_channel)
            if channel_maximum_value > channel_minimum_value:  # Avoid division by zero
                msrcr_combined_result[:, :, channel_index] = (current_channel - channel_minimum_value) / (channel_maximum_value - channel_minimum_value) * 255
            else:
                msrcr_combined_result[:, :, channel_index] = 0
        
        # Clip to valid range and convert to uint8 for color balance processing
        msrcr_uint8_result = np.uint8(np.clip(msrcr_combined_result, 0, 255))
        
        # Apply simplest color balance enhancement
        color_balanced_result = self.apply_color_balance_enhancement(msrcr_uint8_result)
        
        # Convert back to [0, 1] range for final output
        return color_balanced_result.astype(np.float64) / 255.0
    
    def apply_multi_scale_retinex_grayscale(self, input_image, gaussian_sigma_list):
        """Apply Multi-Scale Retinex transformation to grayscale images."""
        # Extract single channel for processing
        grayscale_channel = input_image[:, :, 0] if len(input_image.shape) == 3 else input_image
        
        # Apply Multi-Scale Retinex transformation
        accumulated_msr_result = np.zeros_like(grayscale_channel, dtype=np.float64)
        for current_gaussian_sigma in gaussian_sigma_list:
            gaussian_blurred_channel = cv2.GaussianBlur(grayscale_channel, (0, 0), current_gaussian_sigma)
            accumulated_msr_result += np.log10(grayscale_channel) - np.log10(gaussian_blurred_channel)
        
        averaged_msr_result = accumulated_msr_result / len(gaussian_sigma_list)
        
        # Normalize to [0, 255] range
        channel_minimum_value = np.min(averaged_msr_result)
        channel_maximum_value = np.max(averaged_msr_result)
        if channel_maximum_value > channel_minimum_value:
            normalized_channel_result = (averaged_msr_result - channel_minimum_value) / (channel_maximum_value - channel_minimum_value) * 255
        else:
            normalized_channel_result = np.zeros_like(averaged_msr_result)
        
        # Convert to uint8 and back to [0, 1] range
        final_grayscale_result = np.uint8(np.clip(normalized_channel_result, 0, 255)).astype(np.float64) / 255.0
        
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
