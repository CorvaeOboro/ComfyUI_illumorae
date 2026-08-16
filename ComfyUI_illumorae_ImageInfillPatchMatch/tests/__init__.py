"""Standalone benchmarks for the Image Infill PatchMatch node.

Runs entirely without ComfyUI. Only requires numpy, opencv-python,
and (optionally) scikit-image for SSIM; falls back to a simple SSIM
approximation if not available.
"""
