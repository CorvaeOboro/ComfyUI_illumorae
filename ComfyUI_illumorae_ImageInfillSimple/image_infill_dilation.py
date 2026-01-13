import torch
import numpy as np
import cv2
from typing import Tuple

class illumoraeImageInfillDilationNode:
    """
    A ComfyUI node that fills in subject areas by dilating RGB values from the background.
    
    Takes a subject mask (white=subject, black=background) and fills the subject area
    by dilating background colors inward. This creates a seamless continuation of 
    background colors and textures from the edges towards the center of the subject.
    Useful for preparing images for inpainting or background extension.
    
    Features:
        - RGB dilation from background into subject area
        - Preserves background perfectly
        - Optional center blur with distance-based falloff
        - Feathering control for smooth transitions
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image (B, H, W, C) in ComfyUI format
                "mask": ("MASK",),    # Mask (B, H, W) where 1=subject (white), 0=background (black)
                "mask_mode": (["auto", "white=fill", "black=fill"],),
                "dilation_iterations": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 5000,
                    "step": 1,
                    "display": "number"
                }),
                "blur_center": ("BOOLEAN", {"default": True}),
                "blur_strength": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 10000.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "blur_falloff_distance": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 5000,
                    "step": 1,
                    "display": "number"
                }),
                "feather_amount": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "debug_prints": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("infilled_image", "distance_map_viz", "blur_weight_viz")
    FUNCTION = "infill_background"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Fills subject areas by dilating background RGB values inward. Outputs: infilled image, distance map visualization, blur weight visualization."

    def _debug_print(self, debug_prints, *args, **kwargs):
        if debug_prints:
            print(*args, **kwargs)

    def dilate_rgb(self, image_np: np.ndarray, mask_np: np.ndarray, iterations: int) -> np.ndarray:
        """
        Dilate RGB values from background into subject area.
        
        Args:
            image_np: Image as numpy array (H, W, C) in range [0, 1]
            mask_np: Mask as numpy array (H, W) where 1=subject (white), 0=background (black)
            iterations: Number of dilation iterations
            
        Returns:
            Infilled image as numpy array (H, W, C)
        """
        # Get number of channels
        num_channels = image_np.shape[2] if len(image_np.shape) == 3 else 1
        
        # Convert to uint8 for OpenCV operations
        image_uint8 = (image_np * 255).astype(np.uint8)
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        # Use mask directly as fill mask (1=subject to fill, 0=background to dilate from)
        fill_mask = mask_uint8
        
        # Create result image - start with background only, zero out subject area
        result = image_uint8.copy()
        mask_3ch = np.stack([mask_uint8] * num_channels, axis=-1)
        result = np.where(mask_3ch > 0, 0, result)  # Zero out subject area
        
        # Dilate each channel separately
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        for i in range(iterations):
            # Dilate the image (expands background colors)
            dilated = cv2.dilate(result, kernel, iterations=1)
            
            # Erode the fill mask to track which pixels to update
            fill_mask_eroded = cv2.erode(fill_mask, kernel, iterations=1)
            
            # Update only the newly exposed pixels
            update_mask = fill_mask - fill_mask_eroded
            # Stack mask to match image channels (RGB or RGBA)
            update_mask_nch = np.stack([update_mask] * num_channels, axis=-1)
            
            result = np.where(update_mask_nch > 0, dilated, result)
            
            # Update fill mask for next iteration
            fill_mask = fill_mask_eroded
            
            # Stop if no more pixels to fill
            if np.sum(fill_mask) == 0:
                break
        
        # Convert back to float [0, 1]
        result_float = result.astype(np.float32) / 255.0
        
        return result_float

    def create_distance_map(self, mask_np: np.ndarray) -> np.ndarray:
        """
        Create a distance map from mask edges.
        
        Args:
            mask_np: Mask as numpy array (H, W) where 1=subject (white), 0=background (black)
            
        Returns:
            Distance map as numpy array (H, W) with distances from edges into subject
        """
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        # Calculate distance transform into the subject area (white area)
        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        
        return dist_transform

    def apply_center_blur(
        self, 
        image_np: np.ndarray, 
        mask_np: np.ndarray, 
        blur_strength: float,
        falloff_distance: int
    ) -> np.ndarray:
        """
        Apply blur to the center of subject area where dilation fronts meet.
        
        The blur is strongest at the center (where fronts collide) and fades
        toward the edges based on distance from the mask boundary.
        
        Args:
            image_np: Image as numpy array (H, W, C) in range [0, 1]
            mask_np: Mask as numpy array (H, W) where 1=subject (white), 0=background (black)
            blur_strength: Blur kernel size (will be converted to odd integer)
            falloff_distance: Distance from edges where blur reaches full strength
            
        Returns:
            Blurred image as numpy array (H, W, C)
        """
        # Get number of channels
        num_channels = image_np.shape[2] if len(image_np.shape) == 3 else 1
        
        # Convert blur strength to odd kernel size
        kernel_size = int(blur_strength)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)
        
        # Convert to uint8 for OpenCV
        image_uint8 = (image_np * 255).astype(np.uint8)
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        # Create distance map from edges into subject area
        dist_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        
        # Normalize distance to [0, 1] based on falloff distance
        # Pixels at edges = 0 (no blur), pixels at center = 1 (full blur)
        blur_weight = np.clip(dist_map / falloff_distance, 0, 1)
        
        # Apply blur only to subject area
        subject_mask_3ch = np.stack([mask_uint8 > 0] * num_channels, axis=-1).astype(np.uint8)
        
        # Apply Gaussian blur to entire image
        blurred = cv2.GaussianBlur(image_uint8, (kernel_size, kernel_size), 0)
        
        # Stack blur weight to match channels
        blur_weight_3ch = np.stack([blur_weight] * num_channels, axis=-1)
        
        # Blend: edges=sharp, center=blurred, only in subject area
        result = image_uint8.copy()
        result = np.where(
            subject_mask_3ch > 0,
            (image_uint8 * (1 - blur_weight_3ch) + blurred * blur_weight_3ch).astype(np.uint8),
            result
        )
        
        # Convert back to float
        result_float = result.astype(np.float32) / 255.0
        
        return result_float

    def feather_mask(self, mask_np: np.ndarray, feather_amount: int) -> np.ndarray:
        """
        Feather the mask edges for smooth transitions.
        
        Args:
            mask_np: Mask as numpy array (H, W) where 1=subject (white), 0=background (black)
            feather_amount: Number of pixels to feather
            
        Returns:
            Feathered mask as numpy array (H, W)
        """
        if feather_amount == 0:
            return mask_np
        
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        # Create distance transform from both sides of mask edge
        dist_from_subject = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        dist_from_background = cv2.distanceTransform(255 - mask_uint8, cv2.DIST_L2, 5)
        
        # Create feathered transition at edges
        feather_map = np.minimum(dist_from_subject, dist_from_background)
        feather_map = np.clip(feather_map / feather_amount, 0, 1)
        
        # Apply feathering
        feathered = mask_np * feather_map
        
        return feathered

    def infill_background(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        mask_mode: str,
        dilation_iterations: int,
        blur_center: bool,
        blur_strength: float,
        blur_falloff_distance: int,
        feather_amount: int,
        debug_prints: bool = False,
    ) -> Tuple[torch.Tensor]:
        """
        Main processing function for background infill.
        
        Args:
            image: Input image tensor (B, H, W, C)
            mask: Input mask tensor (B, H, W) where 1=keep, 0=fill
            dilation_iterations: Number of dilation iterations
            blur_center: Whether to apply center blur
            blur_strength: Blur kernel size
            blur_falloff_distance: Distance for blur falloff
            feather_amount: Feathering amount in pixels
            debug_prints: Enable debug output
            
        Returns:
            Tuple containing the infilled image tensor
        """
        self._debug_print(debug_prints, f"Input image shape: {image.shape}")
        self._debug_print(debug_prints, f"Input mask shape: {mask.shape}")
        
        # Process each image in batch
        batch_size = image.shape[0]
        results = []
        distance_maps = []
        blur_weights = []
        
        for b in range(batch_size):
            # Get single image and mask
            img = image[b].cpu().numpy()  # (H, W, C)
            msk = mask[b].cpu().numpy()   # (H, W)

            msk = np.clip(msk, 0.0, 1.0)

            if mask_mode == "white=fill":
                self._debug_print(debug_prints, "Mask mode: white=fill")
            elif mask_mode == "black=fill":
                self._debug_print(debug_prints, "Mask mode: black=fill (inverting)")
                msk = 1.0 - msk
            else:
                mask_mean = float(np.mean(msk))
                self._debug_print(debug_prints, f"Mask mode: auto (mean={mask_mean:.4f})")
                if mask_mean > 0.5:
                    self._debug_print(debug_prints, "Auto mask invert triggered")
                    msk = 1.0 - msk
            
            self._debug_print(debug_prints, f"Processing batch {b+1}/{batch_size}")
            self._debug_print(debug_prints, f"Image range: [{img.min():.3f}, {img.max():.3f}]")
            self._debug_print(debug_prints, f"Mask range: [{msk.min():.3f}, {msk.max():.3f}]")
            
            # Feather mask if requested
            if feather_amount > 0:
                msk_feathered = self.feather_mask(msk, feather_amount)
                self._debug_print(debug_prints, f"Applied feathering: {feather_amount} pixels")
            else:
                msk_feathered = msk
            
            # Create distance map for visualization
            mask_uint8 = (msk_feathered * 255).astype(np.uint8)
            dist_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
            
            # Normalize distance map for visualization (0-1 range)
            if dist_map.max() > 0:
                dist_map_norm = dist_map / dist_map.max()
            else:
                dist_map_norm = dist_map
            
            # Create blur weight map for visualization
            blur_weight_map = np.clip(dist_map / blur_falloff_distance, 0, 1)
            
            # Convert to RGB for output
            dist_map_rgb = np.stack([dist_map_norm] * 3, axis=-1).astype(np.float32)
            blur_weight_rgb = np.stack([blur_weight_map] * 3, axis=-1).astype(np.float32)
            
            # Dilate RGB to fill masked areas
            self._debug_print(debug_prints, f"Dilating RGB: {dilation_iterations} iterations")
            infilled = self.dilate_rgb(img, msk_feathered, dilation_iterations)
            
            # Apply center blur if requested
            if blur_center and blur_strength > 0:
                self._debug_print(debug_prints, f"Applying center blur: strength={blur_strength}, falloff={blur_falloff_distance}")
                infilled = self.apply_center_blur(infilled, msk_feathered, blur_strength, blur_falloff_distance)
            
            # Blend with original using mask
            # Stack mask to match image channels (RGB or RGBA)
            # Mask: 1=subject (fill with infilled), 0=background (keep original)
            num_channels = img.shape[2] if len(img.shape) == 3 else 1
            msk_nch = np.stack([msk_feathered] * num_channels, axis=-1)
            final = img * (1 - msk_nch) + infilled * msk_nch
            
            self._debug_print(debug_prints, f"Final image range: [{final.min():.3f}, {final.max():.3f}]")
            self._debug_print(debug_prints, f"Distance map range: [{dist_map_norm.min():.3f}, {dist_map_norm.max():.3f}]")
            self._debug_print(debug_prints, f"Blur weight range: [{blur_weight_map.min():.3f}, {blur_weight_map.max():.3f}]")
            
            # Convert back to tensors
            result_tensor = torch.from_numpy(final).float()
            dist_map_tensor = torch.from_numpy(dist_map_rgb).float()
            blur_weight_tensor = torch.from_numpy(blur_weight_rgb).float()
            
            results.append(result_tensor)
            distance_maps.append(dist_map_tensor)
            blur_weights.append(blur_weight_tensor)
        
        # Stack batches
        output = torch.stack(results, dim=0)
        distance_output = torch.stack(distance_maps, dim=0)
        blur_weight_output = torch.stack(blur_weights, dim=0)
        
        self._debug_print(debug_prints, f"Output shape: {output.shape}")
        self._debug_print(debug_prints, f"Distance map output shape: {distance_output.shape}")
        self._debug_print(debug_prints, f"Blur weight output shape: {blur_weight_output.shape}")
        
        return (output, distance_output, blur_weight_output)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "illumoraeImageInfillDilationNode": illumoraeImageInfillDilationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeImageInfillDilationNode": "Image Infill Simple Dilation",
}
