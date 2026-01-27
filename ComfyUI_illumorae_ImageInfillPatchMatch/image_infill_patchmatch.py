"""
Image Infill PatchMatch
a rough approximation of the PatchMatch algorithm
finding nearest neighbor patches to fill a masked area ,
usage example = for the removal of objects from image then using the infilled image as guidance for inpainting

STATUS:: working , low quality 
TITLE::Image Infill PatchMatch
DESCRIPTIONSHORT::Approximate PatchMatch-style content fill to infill masked regions; useful as inpainting guidance.
VERSION::20260127
IMAGE::comfyui_illumorae_image_infill_patchmatch.png
GROUP::Image
"""
import torch
import numpy as np
import cv2
from typing import Tuple

class illumoraeImageInfillPatchMatchNode:
    """
    A ComfyUI node that fills in subject areas to approximate the PatchMatch algorithm.
    
    Infill by finding approximate nearest neighbor patches,
    this implementation is a lesser quality version of the Photoshop's Content-Aware Fill. 
    It fills masked areas by finding andcopying similar patches from the surrounding unmasked regions.
    
    - PatchMatch algorithm for patch selection
    - Iterative refinement 
    - Configurable patch size and search radius
    - Multi-scale processing support
    - Preserves background 
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image (B, H, W, C) in ComfyUI format
                "mask": ("MASK",),    # Mask (B, H, W) where 1=subject (white), 0=background (black)
                "patch_size": ("INT", {
                    "default": 7,
                    "min": 3,
                    "max": 31,
                    "step": 2,
                    "display": "number"
                }),
                "iterations": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "number"
                }),
                "search_radius": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "display": "number"
                }),
                "blend_width": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "debug_prints": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("infilled_image", "offset_map_viz")
    FUNCTION = "patchmatch_infill"
    CATEGORY = "illumorae"
    OUTPUT_NODE = False
    DESCRIPTION = "Approximate PatchMatch-style content fill to infill masked regions; useful as inpainting guidance."

    def _debug_print(self, debug_prints, *args, **kwargs):
        if debug_prints:
            print(*args, **kwargs)

    def initialize_offset_map(self, mask_np: np.ndarray, search_radius: int) -> np.ndarray:
        """
        Initialize random offset map for pixels to fill.
        
        Args:
            mask_np: Mask array (H, W) where 1=fill, 0=source
            search_radius: Maximum search distance
            
        Returns:
            Offset map (H, W, 2) with random offsets
        """
        h, w = mask_np.shape
        offset_map = np.zeros((h, w, 2), dtype=np.int32)
        
        # For each pixel to fill, assign random offset to source region
        fill_mask = mask_np > 0.5
        
        for y in range(h):
            for x in range(w):
                if fill_mask[y, x]:
                    # Random offset within search radius
                    dy = np.random.randint(-search_radius, search_radius + 1)
                    dx = np.random.randint(-search_radius, search_radius + 1)
                    
                    # Clamp to valid coordinates
                    ty = np.clip(y + dy, 0, h - 1)
                    tx = np.clip(x + dx, 0, w - 1)
                    
                    # Store offset
                    offset_map[y, x] = [ty - y, tx - x]
        
        return offset_map

    def compute_patch_distance(
        self, 
        image_np: np.ndarray, 
        mask_np: np.ndarray,
        y1: int, x1: int, 
        y2: int, x2: int, 
        patch_size: int
    ) -> float:
        """
        Compute distance between two patches.
        
        Args:
            image_np: Image array (H, W, C)
            mask_np: Mask array (H, W)
            y1, x1: First patch center
            y2, x2: Second patch center
            patch_size: Patch size (odd number)
            
        Returns:
            Distance between patches (lower is better)
        """
        h, w, c = image_np.shape
        half_patch = patch_size // 2
        
        distance = 0.0
        count = 0
        
        for dy in range(-half_patch, half_patch + 1):
            for dx in range(-half_patch, half_patch + 1):
                py1, px1 = y1 + dy, x1 + dx
                py2, px2 = y2 + dy, x2 + dx
                
                # Check bounds
                if (0 <= py1 < h and 0 <= px1 < w and 
                    0 <= py2 < h and 0 <= px2 < w):
                    
                    # Only compare if source pixel is valid (not in mask)
                    if mask_np[py2, px2] < 0.5:
                        diff = image_np[py1, px1] - image_np[py2, px2]
                        distance += np.sum(diff * diff)
                        count += 1
        
        return distance / max(count, 1)

    def propagation_step(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        offset_map: np.ndarray,
        patch_size: int,
        reverse: bool = False
    ) -> np.ndarray:
        """
        Propagation step: try offsets from neighbors.
        
        Args:
            image_np: Image array (H, W, C)
            mask_np: Mask array (H, W)
            offset_map: Current offset map (H, W, 2)
            patch_size: Patch size
            reverse: Scan in reverse order
            
        Returns:
            Updated offset map
        """
        h, w = mask_np.shape
        new_offset_map = offset_map.copy()
        fill_mask = mask_np > 0.5
        
        # Scan order
        y_range = range(h-1, -1, -1) if reverse else range(h)
        x_range = range(w-1, -1, -1) if reverse else range(w)
        
        for y in y_range:
            for x in x_range:
                if not fill_mask[y, x]:
                    continue
                
                # Current best
                current_offset = offset_map[y, x]
                ty, tx = y + current_offset[0], x + current_offset[1]
                best_distance = self.compute_patch_distance(
                    image_np, mask_np, y, x, ty, tx, patch_size
                )
                best_offset = current_offset.copy()
                
                # Try left neighbor's offset
                if x > 0 and fill_mask[y, x-1]:
                    neighbor_offset = offset_map[y, x-1]
                    ty, tx = y + neighbor_offset[0], x + neighbor_offset[1]
                    if 0 <= ty < h and 0 <= tx < w:
                        distance = self.compute_patch_distance(
                            image_np, mask_np, y, x, ty, tx, patch_size
                        )
                        if distance < best_distance:
                            best_distance = distance
                            best_offset = neighbor_offset.copy()
                
                # Try top neighbor's offset
                if y > 0 and fill_mask[y-1, x]:
                    neighbor_offset = offset_map[y-1, x]
                    ty, tx = y + neighbor_offset[0], x + neighbor_offset[1]
                    if 0 <= ty < h and 0 <= tx < w:
                        distance = self.compute_patch_distance(
                            image_np, mask_np, y, x, ty, tx, patch_size
                        )
                        if distance < best_distance:
                            best_distance = distance
                            best_offset = neighbor_offset.copy()
                
                new_offset_map[y, x] = best_offset
        
        return new_offset_map

    def random_search_step(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        offset_map: np.ndarray,
        patch_size: int,
        search_radius: int
    ) -> np.ndarray:
        """
        Random search step: try random offsets with decreasing radius.
        
        Args:
            image_np: Image array (H, W, C)
            mask_np: Mask array (H, W)
            offset_map: Current offset map (H, W, 2)
            patch_size: Patch size
            search_radius: Initial search radius
            
        Returns:
            Updated offset map
        """
        h, w = mask_np.shape
        new_offset_map = offset_map.copy()
        fill_mask = mask_np > 0.5
        
        for y in range(h):
            for x in range(w):
                if not fill_mask[y, x]:
                    continue
                
                # Current best
                current_offset = offset_map[y, x]
                ty, tx = y + current_offset[0], x + current_offset[1]
                best_distance = self.compute_patch_distance(
                    image_np, mask_np, y, x, ty, tx, patch_size
                )
                best_offset = current_offset.copy()
                
                # Try random offsets with exponentially decreasing radius
                radius = search_radius
                while radius >= 1:
                    # Random offset within current radius
                    dy = np.random.randint(-radius, radius + 1)
                    dx = np.random.randint(-radius, radius + 1)
                    
                    ty = np.clip(y + current_offset[0] + dy, 0, h - 1)
                    tx = np.clip(x + current_offset[1] + dx, 0, w - 1)
                    
                    # Only consider if target is in source region
                    if mask_np[ty, tx] < 0.5:
                        distance = self.compute_patch_distance(
                            image_np, mask_np, y, x, ty, tx, patch_size
                        )
                        if distance < best_distance:
                            best_distance = distance
                            best_offset = [ty - y, tx - x]
                    
                    radius //= 2
                
                new_offset_map[y, x] = best_offset
        
        return new_offset_map

    def reconstruct_image(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        offset_map: np.ndarray,
        blend_width: int
    ) -> np.ndarray:
        """
        Reconstruct image using offset map.
        
        Args:
            image_np: Source image (H, W, C)
            mask_np: Mask array (H, W)
            offset_map: Offset map (H, W, 2)
            blend_width: Width of blending at boundaries
            
        Returns:
            Reconstructed image (H, W, C)
        """
        h, w, c = image_np.shape
        result = image_np.copy()
        fill_mask = mask_np > 0.5
        
        # Create distance map for blending
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        dist_map = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        
        for y in range(h):
            for x in range(w):
                if fill_mask[y, x]:
                    # Get source pixel from offset
                    offset = offset_map[y, x]
                    sy, sx = y + offset[0], x + offset[1]
                    
                    if 0 <= sy < h and 0 <= sx < w:
                        source_color = image_np[sy, sx]
                        
                        # Blend near boundaries
                        if blend_width > 0 and dist_map[y, x] < blend_width:
                            blend_weight = dist_map[y, x] / blend_width
                            result[y, x] = source_color
                        else:
                            result[y, x] = source_color
        
        return result

    def patchmatch_infill(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        patch_size: int,
        iterations: int,
        search_radius: int,
        blend_width: int,
        debug_prints: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main PatchMatch infill function.
        
        Args:
            image: Input image tensor (B, H, W, C)
            mask: Input mask tensor (B, H, W) where 1=subject (fill), 0=background
            patch_size: Patch size (odd number)
            iterations: Number of PatchMatch iterations
            search_radius: Search radius for random search
            blend_width: Blending width at boundaries
            debug_prints: Enable debug output
            
        Returns:
            Tuple of (infilled_image, offset_map_visualization)
        """
        self._debug_print(debug_prints, f"Input image shape: {image.shape}")
        self._debug_print(debug_prints, f"Input mask shape: {mask.shape}")
        self._debug_print(debug_prints, f"Patch size: {patch_size}, Iterations: {iterations}")
        
        # Ensure patch_size is odd
        if patch_size % 2 == 0:
            patch_size += 1
            self._debug_print(debug_prints, f"Adjusted patch size to odd: {patch_size}")
        
        # Process each image in batch
        batch_size = image.shape[0]
        results = []
        offset_vizs = []
        
        for b in range(batch_size):
            # Get single image and mask
            img = image[b].cpu().numpy()  # (H, W, C)
            msk = mask[b].cpu().numpy()   # (H, W)
            
            self._debug_print(debug_prints, f"Processing batch {b+1}/{batch_size}")
            self._debug_print(debug_prints, f"Image range: [{img.min():.3f}, {img.max():.3f}]")
            self._debug_print(debug_prints, f"Mask range: [{msk.min():.3f}, {msk.max():.3f}]")
            
            # Initialize offset map
            self._debug_print(debug_prints, "Initializing offset map...")
            offset_map = self.initialize_offset_map(msk, search_radius)
            
            # PatchMatch iterations
            for iter_idx in range(iterations):
                self._debug_print(debug_prints, f"Iteration {iter_idx + 1}/{iterations}")
                
                # Propagation (forward)
                offset_map = self.propagation_step(
                    img, msk, offset_map, patch_size, reverse=False
                )
                
                # Propagation (backward)
                offset_map = self.propagation_step(
                    img, msk, offset_map, patch_size, reverse=True
                )
                
                # Random search
                offset_map = self.random_search_step(
                    img, msk, offset_map, patch_size, search_radius
                )
            
            # Reconstruct image
            self._debug_print(debug_prints, "Reconstructing image...")
            result = self.reconstruct_image(img, msk, offset_map, blend_width)
            
            # Create offset map visualization
            # Visualize offset magnitude
            offset_magnitude = np.sqrt(
                offset_map[:, :, 0]**2 + offset_map[:, :, 1]**2
            )
            if offset_magnitude.max() > 0:
                offset_viz = offset_magnitude / offset_magnitude.max()
            else:
                offset_viz = offset_magnitude
            
            # Convert to RGB
            offset_viz_rgb = np.stack([offset_viz] * 3, axis=-1).astype(np.float32)
            
            self._debug_print(debug_prints, f"Result range: [{result.min():.3f}, {result.max():.3f}]")
            
            # Convert to tensors
            result_tensor = torch.from_numpy(result).float()
            offset_viz_tensor = torch.from_numpy(offset_viz_rgb).float()
            
            results.append(result_tensor)
            offset_vizs.append(offset_viz_tensor)
        
        # Stack batches
        output = torch.stack(results, dim=0)
        offset_output = torch.stack(offset_vizs, dim=0)
        
        self._debug_print(debug_prints, f"Output shape: {output.shape}")
        self._debug_print(debug_prints, f"Offset viz shape: {offset_output.shape}")
        
        return (output, offset_output)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "illumoraeImageInfillPatchMatchNode": illumoraeImageInfillPatchMatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "illumoraeImageInfillPatchMatchNode": "Image Infill PatchMatch",
}
