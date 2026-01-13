<p align="center">
  <img src="https://github.com/CorvaeOboro/ComfyUI_illumorae/sd_project_tools/blob/main/docs/sd_project_tools_header_long.png?raw=true" height="200" /> 
</p>

# ComfyUI illumorae

comfyUI custom nodes focused on randomization and image variant exploration
- loading random files by string , randomizing lora string strength , text order
- load and save with external folderpath and filename outputs for project structures
- each node is self-contained and could be installed separately if prefer 

<img src="docs/comfyui_illumorae_load_checkpoint_text_file_basic.png" width="800" caption="workflow"/>

# install 
- install through the the ComfyUI Manager or manually [download](https://github.com/CorvaeOboro/ComfyUI_illumorae/sd_project_tools/archive/refs/heads/main.zip) as a zip and extract as folder into the ComfyUI `custom_nodes` directory 
- OPTIONAL install nodes individual copying a nodes subfolder into the ComfyUI `custom_nodes` directory , each has been setup to function independently

# nodes 

<table>
  <thead>
    <tr>
      <th>Checkpoint</th>
      <th>LoRA</th>
      <th>Load</th>
      <th>Image</th>
      <th>Text</th>
      <th>Save</th>
    </tr>
  </thead>
 <tbody>
    <tr>
      <td>
        <ul style="list-style-type: disc; padding-left: 1.2em;">
          <li><a href="./ComfyUI_illumorae_CheckpointLoaderByStringDirty/checkpoint_loader_by_string_dirty.py">Checkpoint Loader By String Dirty</a></li>
          <li><a href="./ComfyUI_illumorae_CheckpointRandomSelector/checkpoint_random_selector.py">Checkpoint Random Selector</a></li>
        </ul>
      </td>
      <td>
        <ul style="list-style-type: disc; padding-left: 1.2em;">
          <li><a href="./ComfyUI_illumorae_LoraTextStrengthVariants/lora_text_strength_variants.py">LoRA Text Strength Variants</a></li>
          <li><a href="./ComfyUI_illumorae_LoraTextStrengthMultiplier/lora_text_strength_multiplier.py">LoRA Text Strength Multiplier</a></li>
          <li><a href="./ComfyUI_illumorae_LoraTextVisualizer/lora_text_visualizer.py">LoRA Text Visualizer</a></li>
        </ul>
      </td>
      <td>
        <ul style="list-style-type: disc; padding-left: 1.2em;">
          <li><a href="./ComfyUI_illumorae_LoadImageFilePathOut/load_image_filepath_out.py">Load Image FilePath Out</a></li>
          <li><a href="./ComfyUI_illumorae_LoadImageRandomVariants/load_image_random_variant.py">Load Image Random Variants</a></li>
          <li><a href="./ComfyUI_illumorae_LoadTextFileGraceful/load_text_file_graceful.py">Load Text File Graceful</a></li>
          <li><a href="./ComfyUI_illumorae_SelectITEMbyAmountGenerated/select_item_by_amount_generated.py">Select ITEM by Amount Generated</a></li>
        </ul>
      </td>
      <td>
        <ul style="list-style-type: disc; padding-left: 1.2em;">
          <li><a href="./ComfyUI_illumorae_ImageContrastLimitedAdaptiveHistogramEqualization/image_CLAHE.py">Image CLAHE</a></li>
          <li><a href="./ComfyUI_illumorae_ImageInfillSimple/image_infill_dilation.py">Image Infill Dilation</a></li>
          <li><a href="./ComfyUI_illumorae_ImageInfillPatchMatch/image_infill_patchmatch.py">Image Infill PatchMatch</a></li>
          <li><a href="./ComfyUI_illumorae_ImageMultiScaleRetinexColorRestoration/image_MSRCR.py">Image Multi-Scale Retinex Color Restoration</a></li>
          <li><a href="./ComfyUI_illumorae_VideoResizeMatte/image_resize_matte_video.py">Video Resize Matte</a></li>
          <li><a href="./ComfyUI_illumorae_ImageResizeFluxKontextOptions/flux_kontext_image_scale_options.py">Flux Kontext Image Scale Options</a></li>
          <li><a href="./ComfyUI_illumorae_VLM_InternVL_Local/vlm_internVL_local.py">VLM InternVL Local</a></li>
        </ul>
      </td>
      <td>
        <ul style="list-style-type: disc; padding-left: 1.2em;">
          <li><a href="./ComfyUI_illumorae_TextToStringSafeForFilename/text_to_text_safe_for_filename.py">Text To Filename Safe Text</a></li>
          <li><a href="./ComfyUI_illumorae_TextTokenCount/text_token_count.py">Text Token Count</a></li>
          <li><a href="./ComfyUI_illumorae_TextStrengthMultiplier/text_strength_multiplier.py">Text Strength Multiplier</a></li>
          <li><a href="./ComfyUI_illumorae_TextEnclosureVisualizer/text_enclosure_visualizer.py">Text Enclosure Visualizer</a></li>
          <li><a href="./ComfyUI_illumorae_TextReorder/text_reorder.py">Text Reorder</a></li>
        </ul>
      </td>
      <td>
        <ul style="list-style-type: disc; padding-left: 1.2em;">
          <li><a href="./ComfyUI_illumorae_SaveImageExtendedFolderPath/save_image_extended_folderpath.py">Save Image Extended FolderPath</a></li>
          <li><a href="./ComfyUI_illumorae_SaveAnimatedWebPExtendedFolderPath/save_animated_webp_extended_folderpath.py">Save Animated WebP Extended FolderPath</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

---

## Checkpoint
- **[Checkpoint Loader By String Dirty](./ComfyUI_illumorae_CheckpointLoaderByStringDirty/checkpoint_loader_by_string_dirty.py)**  
  Loads a Stable Diffusion checkpoint by matching a string input (full path, relative path, or filename) to any registered checkpoint.
- **[Checkpoint Random Selector](./ComfyUI_illumorae_CheckpointRandomSelector/checkpoint_random_selector.py)**  
  Randomly selects a checkpoint from a category/folder at a set interval for reproducible model rotation.

## LoRA 
- **[LoRA Text Strength Variants](./ComfyUI_illumorae_LoraTextStrengthVariants/lora_text_strength_variants.py)**  
  Parses LoRA strings and randomizes or highlights strengths within specified limits.
- <img src="docs/comfyui_illumorae_lora_strength_randomize.png" height="200" />
- **[LoRA Text Strength Multiplier](./ComfyUI_illumorae_LoraTextStrengthMultiplier/lora_text_strength_multiplier.py)**  
  Multiplies and caps LoRA strengths, with options for total and individual caps.
- <img src="docs/comfyui_illumorae_lora_strength_multiplier.png" height="200" />
- **[LoRA Text Visualizer](./ComfyUI_illumorae_LoraTextVisualizer/lora_text_visualizer.py)**  
  Visualizes LoRA strengths in prompt text as a word plot image.
- <img src="docs/comfyui_illumorae_lora_strength_wordplot.png" height="200" />
## Load
- **[Load Image FilePath Out](./ComfyUI_illumorae_LoadImageFilePathOut/load_image_filepath_out.py)**  
  Loads an image from a file path and outputs the image, mask, file name, and folder path.
- **[Load Image Random Variants](./ComfyUI_illumorae_LoadImageRandomVariants/load_image_random_variant.py)**  
  Loads a random variant of an image from a folder, with debug and suffix options.
- **[Load Text File Graceful](./ComfyUI_illumorae_LoadTextFileGraceful/load_text_file_graceful.py)**  
  Loads text from a file, gracefully handling missing/invalid files.
- <img src="docs/comfyui_illumorae_load_text_graceful.png" height="200" />
- **[Select ITEM by Amount Generated](./ComfyUI_illumorae_SelectITEMbyAmountGenerated/select_item_by_amount_generated.py)**  
  Selects an ITEM output based on how many images/frames have been generated so far.

## Image Processing
- <img src="docs/comfyui_illumorae_image_load_variant_contrast_save.png" height="200" />

- **[Image CLAHE (Contrast Limited Adaptive Histogram Equalization)](./ComfyUI_illumorae_ImageContrastLimitedAdaptiveHistogramEqualization/image_CLAHE.py)**  
  Enhances local image contrast using CLAHE.
- **[Image Infill Dilation](./ComfyUI_illumorae_ImageInfillSimple/image_infill_dilation.py)**  
  Simple image infill using dilation-based expansion.
- **[Image Infill PatchMatch](./ComfyUI_illumorae_ImageInfillPatchMatch/image_infill_patchmatch.py)**  
  PatchMatch-based image infill.
- **[Image Multi-Scale Retinex Color Restoration](./ComfyUI_illumorae_ImageMultiScaleRetinexColorRestoration/image_MSRCR.py)**  
  Applies Multi-Scale Retinex with Color Restoration for dynamic range and color enhancement.
- **[Video Resize Matte](./ComfyUI_illumorae_VideoResizeMatte/image_resize_matte_video.py)**  
  Resizes video frames with matte options for compositing.
- **[Image Resize Flux Kontext Options](./ComfyUI_illumorae_ImageResizeFluxKontextOptions/flux_kontext_image_scale_options.py)**  
  Resizes images for optimal Flux Kontext input, with cropping/stretching options.

## Text & Prompt Utilities
- **[Text To String Safe For Filename](./ComfyUI_illumorae_TextToStringSafeForFilename/text_to_text_safe_for_filename.py)**  
  Converts text into a filename-safe string.
- **[Text Token Count](./ComfyUI_illumorae_TextTokenCount/text_token_count.py)**  
  Counts tokens in a string (useful for prompt engineering).
- <img src="docs/comfyui_illumorae_text_token_count.png" height="200" />
- **[Text Strength Multiplier](./ComfyUI_illumorae_TextStrengthMultiplier/text_strength_multiplier.py)**  
  Multiplies the strength of text prompt components.
- <img src="docs/comfyui_illumorae_text_strength_multiplier.png" height="200" />
- **[Text Enclosure Visualizer](./ComfyUI_illumorae_TextEnclosureVisualizer/text_enclosure_visualizer.py)**  
  Visualizes enclosed text ranges for prompt building.
- **[Text Reorder](./ComfyUI_illumorae_TextReorder/text_reorder.py)**  
  Reorders prompt text using configurable rules.

## Save
- **[Save Image Extended FolderPath](./ComfyUI_illumorae_SaveImageExtendedFolderPath/save_image_extended_folderpath.py)**  
  Saves images to an external folder path, supporting custom folder and filename formats.
- **[Save Animated WebP Extended FolderPath](./ComfyUI_illumorae_SaveAnimatedWebPExtendedFolderPath/save_animated_webp_extended_folderpath.py)**  
  Saves animated WebP images to a user-specified folder.

## VLM
- **[VLM InternVL Local](./ComfyUI_illumorae_VLM_InternVL_Local/vlm_internVL_local.py)**  
  Runs InternVL locally for Image to Text vision model

# LICENSE
- free to all , [creative commons zero CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) , free to re-distribute , attribution not required