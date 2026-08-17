<p align="center">
  <img src="https://github.com/CorvaeOboro/ComfyUI_illumorae/blob/main/docs/comfyui_illumorae_title.png?raw=true" height="200" /> 
</p>

# ComfyUI illumorae

comfyUI custom nodes focused on randomization and image variant exploration
- loading files by partial string match , randomizing lora string strength , text reordering
- load and save with external folderpath and filename outputs for project structures
- each node is self-contained and could be installed separately if prefer specific nodes

<a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/CheckpointLoaderByStringDirty.html"><img src="docs/comfyui_illumorae_load_checkpoint_text_file_basic.png" width="800" caption="workflow"/></a>

# install 
- install thru the [ComfyUI Manager](https://github.com/Comfy-Org/ComfyUI-Manager) search for "illumorae" or manually [download](https://github.com/CorvaeOboro/ComfyUI_illumorae/archive/refs/heads/main.zip) as a zip and extract as folder into the ComfyUI `custom_nodes` directory 
- OPTIONAL may install nodes individually by copying a nodes subfolder into the ComfyUI `custom_nodes` directory , each has been setup to function independently

# nodes 

<table>
  <tr>
    <td><a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/ImageInfillBoundaryPropagate.html"><img src="docs/comfyui_illumorae_image_infill_multiple.png" width="400" /></a></td>
    <td><a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/ImageMultiScaleRetinexColorRestoration.html"><img src="docs/comfyui_illumorae_image_retinex_enhancement_msrcr.png" width="400" /></a></td>
  </tr>
  <tr>
    <td><a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/LoadImageRandomVariants.html"><img src="docs/comfyui_illumorae_image_load_variant_contrast_save.png" width="400" /></a></td>
    <td><a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/GenProceduralPatterns.html"><img src="docs/comfyui_illumorae_procedural_patterns.png" width="400" /></a></td>
  </tr>
</table>

<table style="border-collapse: collapse; width: 100%;">
  <tbody style="line-height: 1.1;">
    <!-- Image -->
    <tr><th colspan="4" align="left" style="padding: 2px 6px; background: #1f2430; color: #c8d3ff;">Image</th></tr>
    <tr style="vertical-align: top;">
      <td style="padding: 1px 6px;"><a href="#image-infill-boundary-propagate">Image Infill Boundary Propagate</a></td>
      <td style="padding: 1px 6px;"><a href="#image-infill-exemplar-region-fill">Image Infill Exemplar Region Fill</a></td>
      <td style="padding: 1px 6px;"><a href="#image-infill-gaussian-mixture-layer">Image Infill Gaussian Mixture Layer</a></td>
      <td style="padding: 1px 6px;"><a href="#image-infill-patchmatch">Image Infill PatchMatch</a></td>
    </tr>
    <tr style="vertical-align: top;">
      <td style="padding: 1px 6px;"><a href="#image-clahe">Image CLAHE</a></td>
      <td style="padding: 1px 6px;"><a href="#image-multi-scale-retinex-color-restoration">Image Multi-Scale Retinex Color Restoration</a></td>
      <td style="padding: 1px 6px;"><a href="#image-face-aspect-crop">Image Face Aspect Crop</a></td>
      <td style="padding: 1px 6px;"><a href="#video-resize-matte">Video Resize Matte</a></td>
    </tr>
    <tr style="vertical-align: top;">
      <td style="padding: 1px 6px;"><a href="#image-resize-flux-kontext-options">Flux Kontext Image Scale Options</a></td>
      <td style="padding: 1px 6px;"><a href="#procedural-patterns">Procedural Patterns</a></td>
      <td style="padding: 1px 6px;"></td>
      <td style="padding: 1px 6px;"></td>
    </tr>
    <!-- LoRA -->
    <tr><th colspan="4" align="left" style="padding: 2px 6px; background: #1f2430; color: #c8d3ff;">LoRA</th></tr>
    <tr style="vertical-align: top;">
      <td style="padding: 1px 6px;"><a href="#lora-text-strength-variants">LoRA Text Strength Variants</a></td>
      <td style="padding: 1px 6px;"><a href="#lora-text-strength-multiplier">LoRA Text Strength Multiplier</a></td>
      <td style="padding: 1px 6px;"><a href="#lora-text-visualizer">LoRA Text Visualizer</a></td>
      <td style="padding: 1px 6px;"></td>
    </tr>
    <!-- Text -->
    <tr><th colspan="4" align="left" style="padding: 2px 6px; background: #1f2430; color: #c8d3ff;">Text</th></tr>
    <tr style="vertical-align: top;">
      <td style="padding: 1px 6px;"><a href="#text-to-string-safe-for-filename">Text To Filename Safe Text</a></td>
      <td style="padding: 1px 6px;"><a href="#text-token-count">Text Token Count</a></td>
      <td style="padding: 1px 6px;"><a href="#text-strength-multiplier">Text Strength Multiplier</a></td>
      <td style="padding: 1px 6px;"><a href="#text-enclosure-visualizer">Text Enclosure Visualizer</a></td>
    </tr>
    <tr style="vertical-align: top;">
      <td style="padding: 1px 6px;"><a href="#text-reorder">Text Reorder</a></td>
      <td style="padding: 1px 6px;"><a href="#text-multiline-findcolorized">Text Multiline FindColorized</a></td>
      <td style="padding: 1px 6px;"></td>
      <td style="padding: 1px 6px;"></td>
    </tr>
    <!-- Load -->
    <tr><th colspan="4" align="left" style="padding: 2px 6px; background: #1f2430; color: #c8d3ff;">Load</th></tr>
    <tr style="vertical-align: top;">
      <td style="padding: 1px 6px;"><a href="#load-image-filepath-out">Load Image FilePath Out</a></td>
      <td style="padding: 1px 6px;"><a href="#load-image-random-variants">Load Image Random Variants</a></td>
      <td style="padding: 1px 6px;"><a href="#load-text-file-graceful">Load Text File Graceful</a></td>
      <td style="padding: 1px 6px;"><a href="#select-item-by-amount-generated">Select ITEM by Amount Generated</a></td>
    </tr>
    <!-- Checkpoint -->
    <tr><th colspan="4" align="left" style="padding: 2px 6px; background: #1f2430; color: #c8d3ff;">Checkpoint</th></tr>
    <tr style="vertical-align: top;">
      <td style="padding: 1px 6px;"><a href="#checkpoint-loader-by-string-dirty">Checkpoint Loader By String Dirty</a></td>
      <td style="padding: 1px 6px;"><a href="#checkpoint-random-selector">Checkpoint Random Selector</a></td>
      <td style="padding: 1px 6px;"></td>
      <td style="padding: 1px 6px;"></td>
    </tr>
    <!-- Save -->
    <tr><th colspan="4" align="left" style="padding: 2px 6px; background: #1f2430; color: #c8d3ff;">Save</th></tr>
    <tr style="vertical-align: top;">
      <td style="padding: 1px 6px;"><a href="#save-image-extended-folderpath">Save Image Extended FolderPath</a></td>
      <td style="padding: 1px 6px;"><a href="#save-animated-webp-extended-folderpath">Save Animated WebP Extended FolderPath</a></td>
      <td style="padding: 1px 6px;"></td>
      <td style="padding: 1px 6px;"></td>
    </tr>
  </tbody>
</table>

---

## Image Infill
<a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/ImageInfillBoundaryPropagate.html"><img src="docs/comfyui_illumorae_image_infill_multiple.png" height="200" /></a>

<table style="border-collapse: collapse; width: 100%;">
  <tbody>
    <tr style="vertical-align: top;">
      <td style="padding: 4px 8px; width: 50%;">
        <a id="image-infill-boundary-propagate"></a><strong><a href="./ComfyUI_illumorae_ImageInfillBoundaryPropagate/image_infill_boundary_propagate.py">Image Infill Boundary Propagate</a></strong><br/>
        Fills the target region by propagating colors inward from the target boundary using a random-weighted onion-peel rule.<br/>
        <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/ImageInfillBoundaryPropagate.html"><img src="docs/comfyui_illumorae_image_infill_boundary.png" height="200" /></a>
      </td>
      <td style="padding: 4px 8px; width: 50%;">
        <a id="image-infill-exemplar-region-fill"></a><strong><a href="./ComfyUI_illumorae_ImageInfillExemplarRegionFill/image_infill_exemplar_regionfill.py">Image Infill Exemplar Region Fill</a></strong><br/>
        Exemplar-based region filling with isophote-driven priority (Criminisi-style), extends linear structures into the target region.<br/>
        <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/ImageInfillExemplarRegionFill.html"><img src="docs/comfyui_illumorae_image_infill_exemplar.png" height="200" /></a>
      </td>
    </tr>
    <tr style="vertical-align: top;">
      <td style="padding: 4px 8px; width: 50%;">
        <a id="image-infill-gaussian-mixture-layer"></a><strong><a href="./ComfyUI_illumorae_ImageInfillGaussianMixtureLayer/image_infill_gaussian_mixture_layer.py">Image Infill Gaussian Mixture Layer</a></strong><br/>
        Single stationary-Gaussian-field texture inpainting by FFT-based conditional simulation (Galerne-Leclaire 2017).<br/>
        <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/ImageInfillGaussianMixtureLayer.html"><img src="docs/comfyui_illumorae_image_infill_gaussian.png" height="200" /></a>
      </td>
      <td style="padding: 4px 8px; width: 50%;">
        <a id="image-infill-patchmatch"></a><strong><a href="./ComfyUI_illumorae_ImageInfillPatchMatch/image_infill_patchmatch.py">Image Infill PatchMatch</a></strong><br/>
        PatchMatch-based image infill.<br/>
        <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/ImageInfillPatchMatch.html"><img src="docs/comfyui_illumorae_image_infill_patchmatch.png" height="200" /></a>
      </td>
    </tr>
  </tbody>
</table>

## Image Adjustment
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/LoadImageRandomVariants.html"><img src="docs/comfyui_illumorae_image_load_variant_contrast_save.png" height="200" /></a>

- <a id="image-clahe"></a>**[Image CLAHE (Contrast Limited Adaptive Histogram Equalization)](./ComfyUI_illumorae_ImageContrastLimitedAdaptiveHistogramEqualization/image_CLAHE.py)**  
  Enhances local image contrast using CLAHE.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/ImageContrastLimitedAdaptiveHistogramEqualization.html"><img src="docs/comfyui_illumorae_image_clahe.png" height="200" /></a>
- <a id="image-multi-scale-retinex-color-restoration"></a>**[Image Multi-Scale Retinex Color Restoration](./ComfyUI_illumorae_ImageMultiScaleRetinexColorRestoration/image_MSRCR.py)**  
  Applies Multi-Scale Retinex with Color Restoration for dynamic range and color enhancement.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/ImageMultiScaleRetinexColorRestoration.html"><img src="docs/comfyui_illumorae_image_retinex_enhancement_msrcr.png" height="200" /></a>

## Image Resize + Crop
- <a id="image-face-aspect-crop"></a>**[Image Face Aspect Crop](./ComfyUI_illumorae_ImageFaceAspectCrop/image_face_aspect_crop.py)**  
  Crops an image to a target aspect ratio with face-detection-biased anchoring and debug overlay output.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/ImageFaceAspectCrop.html"><img src="docs/comfyui_illumorae_image_face_aspect_crop.png" height="200" /></a>
- <a id="video-resize-matte"></a>**[Video Resize Matte](./ComfyUI_illumorae_VideoResizeMatte/image_resize_matte_video.py)**  
  Resizes video frames with matte options for compositing.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/VideoResizeMatte.html"><img src="docs/comfyui_illumorae_image_resize_video_matte.png" height="200" /></a>
- <a id="image-resize-flux-kontext-options"></a>**[Image Resize Flux Kontext Options](./ComfyUI_illumorae_ImageResizeFluxKontextOptions/flux_kontext_image_scale_options.py)**  
  Resizes images for optimal Flux Kontext input, with cropping/stretching options.

## Image
- <a id="procedural-patterns"></a>**[Procedural Patterns](./ComfyUI_illumorae_GenProceduralPatterns/gen_procedural_patterns.py)**  
  Generates procedural symmetrical patterns (torus rings, radial beams, spiral rings, combined) for visual influence in image generation.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/GenProceduralPatterns.html"><img src="docs/comfyui_illumorae_procedural_patterns.png" height="200" /></a>

## LoRA 
- <a id="lora-text-strength-variants"></a>**[LoRA Text Strength Variants](./ComfyUI_illumorae_LoraTextStrengthVariants/lora_text_strength_variants.py)**  
  Parses LoRA strings and randomizes or highlights strengths within specified limits.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/LoraTextStrengthVariants.html"><img src="docs/comfyui_illumorae_lora_strength_randomize.png" height="200" /></a>
- <a id="lora-text-strength-multiplier"></a>**[LoRA Text Strength Multiplier](./ComfyUI_illumorae_LoraTextStrengthMultiplier/lora_text_strength_multiplier.py)**  
  Multiplies and caps LoRA strengths, with options for total and individual caps.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/LoraTextStrengthMultiplier.html"><img src="docs/comfyui_illumorae_lora_strength_multiplier.png" height="200" /></a>
- <a id="lora-text-visualizer"></a>**[LoRA Text Visualizer](./ComfyUI_illumorae_LoraTextVisualizer/lora_text_visualizer.py)**  
  Visualizes LoRA strengths in prompt text as a word plot image.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/LoraTextVisualizer.html"><img src="docs/comfyui_illumorae_lora_strength_wordplot.png" height="200" /></a>

## Text & Prompt Utilities
- <a id="text-to-string-safe-for-filename"></a>**[Text To String Safe For Filename](./ComfyUI_illumorae_TextToStringSafeForFilename/text_to_text_safe_for_filename.py)**  
  Converts text into a filename-safe string.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/TextToStringSafeForFilename.html"><img src="docs/comfyui_illumorae_text_to_string_filename_safe.png" height="200" /></a>
- <a id="text-token-count"></a>**[Text Token Count](./ComfyUI_illumorae_TextTokenCount/text_token_count.py)**  
  Counts tokens in a string using clip
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/TextTokenCount.html"><img src="docs/comfyui_illumorae_text_token_count.png" height="200" /></a>
- <a id="text-strength-multiplier"></a>**[Text Strength Multiplier](./ComfyUI_illumorae_TextStrengthMultiplier/text_strength_multiplier.py)**  
  Multiplies the strength of text prompt components.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/TextStrengthMultiplier.html"><img src="docs/comfyui_illumorae_text_strength_multiplier.png" height="200" /></a>
- <a id="text-enclosure-visualizer"></a>**[Text Enclosure Visualizer](./ComfyUI_illumorae_TextEnclosureVisualizer/text_enclosure_visualizer.py)**  
  Visualizes enclosed text , nested separation , and warnings on open ended enclosures
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/TextEnclosureVisualizer.html"><img src="docs/comfyui_illumorae_text_enclosure_visualize.png" height="200" /></a>
- <a id="text-reorder"></a>**[Text Reorder](./ComfyUI_illumorae_TextReorder/text_reorder.py)**  
  Reorders prompt text using different scales of rules.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/TextReorder.html"><img src="docs/comfyui_illumorae_text_reorder.png" height="200" /></a>
- <a id="text-multiline-findcolorized"></a>**[Text Multiline FindColorized](./ComfyUI_illumorae_TextMultilineFindColorized/text_multiline_findcolorized.py)**  
  Editable multiline text input with search/find highlighting and syntax-colored LoRA bracket tags.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/TextMultilineFindColorized.html"><img src="docs/comfyui_illumorae_text_multiline_findcolorized.png" height="200" /></a>

## Load
- <a id="load-image-filepath-out"></a>**[Load Image FilePath Out](./ComfyUI_illumorae_LoadImageFilePathOut/load_image_filepath_out.py)**  
  Loads an image from a file path and outputs the image, mask, file name, and folder path.
- <a id="load-image-random-variants"></a>**[Load Image Random Variants](./ComfyUI_illumorae_LoadImageRandomVariants/load_image_random_variant.py)**  
  Loads a random variant of an image from a folder, with suffix options.
- <a id="load-text-file-graceful"></a>**[Load Text File Graceful](./ComfyUI_illumorae_LoadTextFileGraceful/load_text_file_graceful.py)**  
  Loads text from a file, gracefully handling missing/invalid files.
- <a href="https://corvaeoboro.github.io/ComfyUI_illumorae/nodes/LoadTextFileGraceful.html"><img src="docs/comfyui_illumorae_load_text_graceful.png" height="200" /></a>
- <a id="select-item-by-amount-generated"></a>**[Select ITEM by Amount Generated](./ComfyUI_illumorae_SelectITEMbyAmountGenerated/select_item_by_amount_generated.py)**  
  Selects an ITEM output based on how many images/frames have been generated so far.

## Checkpoint
- <a id="checkpoint-loader-by-string-dirty"></a>**[Checkpoint Loader By String Dirty](./ComfyUI_illumorae_CheckpointLoaderByStringDirty/checkpoint_loader_by_string_dirty.py)**  
  Loads a checkpoint by matching a partial string input to closest registered checkpoint.
- <a id="checkpoint-random-selector"></a>**[Checkpoint Random Selector](./ComfyUI_illumorae_CheckpointRandomSelector/checkpoint_random_selector.py)**  
  Randomly selects a checkpoint from a category/folder at a set interval for model rotation.

## Save
- <a id="save-image-extended-folderpath"></a>**[Save Image Extended FolderPath](./ComfyUI_illumorae_SaveImageExtendedFolderPath/save_image_extended_folderpath.py)**  
  Saves images to an external folder path, supporting custom folder and filename formats.
- <a id="save-animated-webp-extended-folderpath"></a>**[Save Animated WebP Extended FolderPath](./ComfyUI_illumorae_SaveAnimatedWebPExtendedFolderPath/save_animated_webp_extended_folderpath.py)**  
  Saves animated WebP images to a user-specified folder.


# LICENSE
- free to all , [creative commons zero CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) , free to re-distribute , attribution not required