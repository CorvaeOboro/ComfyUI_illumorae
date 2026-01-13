"""
illumorae Flux Kontext Image Scale Options - a ComfyUI Custom Node

Resizes the image to one that is more optimal for Flux Lontext.
a modified verison of the native node , with options to stretch instead of always cropping

Inputs:
    image: The input image to resize.
    resize_mode: The mode to use for resizing (crop or stretch).
    interpolation: The interpolation method to use (lanczos, bilinear, or nearest).
    crop_anchor: The anchor point to use for cropping (center, top, bottom, left, or right).

Outputs:
    image: The resized image.

"""
import comfy.utils

PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

class illumoraeFluxKontextImageScaleOptions:
    """
    Flux Kontext Image Scale with Options
    modded to provide options other then only "cropping" to scale the image to a prefered resolution
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "resize_mode": (["crop", "stretch"], {"default": "crop"}),
                "interpolation": (["lanczos", "bilinear", "nearest"], {"default": "lanczos"}),
                "crop_anchor": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale"

    CATEGORY = "illumorae"
    DESCRIPTION = " resizes the image to one that is more optimal for flux kontext with crop or stretch options"

    def scale(self, image, resize_mode="crop", interpolation="lanczos", crop_anchor="center"):
        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
        if resize_mode == "stretch":
            # Stretch: use interpolation, ignore crop_anchor
            # Assuming comfy.utils.common_upscale with anchor=None does plain resize
            image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, interpolation, None).movedim(1, -1)
        else:
            # Crop: use interpolation and anchor
            image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, interpolation, crop_anchor).movedim(1, -1)
        return (image, )

NODE_CLASS_MAPPINGS = {
    'illumoraeFluxKontextImageScaleOptions': illumoraeFluxKontextImageScaleOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeFluxKontextImageScaleOptions': 'Image Resize Flux Kontext Scale Options',
}
