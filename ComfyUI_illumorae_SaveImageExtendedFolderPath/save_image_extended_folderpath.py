"""
SAVE IMAGE EXTENDED FOLDERPATH
saving images to a specified folder path with optional metadata.

TITLE::Save Image Extended FolderPath
DESCRIPTIONSHORT::Saves images to a specified folder path with customizable naming, counters, and optional metadata.
VERSION::20260816
IMAGE::comfyui_illumorae_save_image_extended_folderpath.png
GROUP::Save
GROUPORDER::7
LISTORDER::2
STATUS::working
"""

#region IMPORTS
import os
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
from datetime import datetime
from pathlib import Path
import string

import folder_paths
#endregion


#region NODECLASS
class illumoraeSaveImageExtendedFolderPathNode:

    #region NODEMETA
    RETURN_TYPES = ()
    FUNCTION = 'save_images'
    OUTPUT_NODE = True
    CATEGORY = 'illumorae'
    DESCRIPTION = "Saves images to a specified folder path with customizable naming, counters, and optional metadata."
    #endregion

    #region INIT
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = 'output'
        self.prefix_append = ''
    #endregion

    #region CORE
    def save_images(self,
        counter_digits,
        counter_position,
        delimiter,
        save_metadata,
        images,
        filename_prefix='',
        foldername_prefix='',
        folderpath_input='',
        extra_pnginfo=None,
        prompt=None
    ):
        delimiter_char = "_" if delimiter == 'underscore' else '.' if delimiter == 'dot' else ','

        #region C-PREP
        # Sanitize user-supplied name components (no dynamic keys, prefix only)
        custom_filename = self.sanitize_name(filename_prefix)
        custom_foldername = self.sanitize_name(foldername_prefix)

        # Resolve the base output folder; fall back to ComfyUI output dir when blank
        custom_folderpath = folderpath_input.strip()
        if custom_folderpath == '':
            custom_folderpath = self.output_dir
        custom_folderpath = os.path.normpath(custom_folderpath)

        # Empty-batch guard: nothing to write, return an empty UI result
        if images is None or len(images) == 0:
            return {'ui': {'images': []}}

        results = list()
        try:
            # Resolve final on-disk folder and processed filename via ComfyUI helper
            full_output_folder, filename, _, _, _ = folder_paths.get_save_image_path(
                custom_filename, custom_folderpath, images[0].shape[1], images[0].shape[0])
            filename = self.sanitize_name(filename)
            full_output_folder = os.path.normpath(full_output_folder)
            output_path = str(Path(full_output_folder) / custom_foldername)
            os.makedirs(output_path, exist_ok=True)
            # Scan existing files for the next counter value; pass the processed
            # filename and delimiter so the scan matches on-disk names
            counter = self.get_latest_counter(output_path, filename, counter_digits,
                                              counter_position, delimiter_char)
            #endregion

            #region C-META
            # Build PNG metadata once; identical across the batch
            metadata = None
            if save_metadata == 'enabled':
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text('prompt', json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            #endregion

            #region C-SAVE
            # Write each frame; subfolder is constant across the batch
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

                if counter_position == 'last':
                    file = f'{filename}{delimiter_char}{counter:0{counter_digits}}.png'
                else:
                    file = f'{counter:0{counter_digits}}{delimiter_char}{filename}.png'
                file = self.sanitize_name(file)

                image_path = str(Path(output_path) / file)
                img.save(image_path, pnginfo=metadata, compress_level=4)
                print(f"[SaveImageExtendedFolderPath] Image saved at "
                      f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {image_path}")

                subfolder = self.get_subfolder_path(image_path, custom_folderpath)
                results.append({'filename': file, 'subfolder': subfolder, 'type': self.type})
                counter += 1
            #endregion

        #region C-ERR
        except OSError as e:
            print(f'[SaveImageExtendedFolderPath] An error occurred while creating '
                  f'the subfolder or saving the image: {e}')
            raise
        except Exception as e:
            print(f'[SaveImageExtendedFolderPath] Unexpected error: {e}')
            raise
        else:
            return {'ui': {'images': results}}
        #endregion
    #endregion

    #region HELPERS
    #region H-SANITIZE
    @staticmethod
    def sanitize_name(name):
        # Remove illegal Windows path characters
        invalid_chars = '<>:"/\\|?*'
        # Replace with underscore, also strip leading/trailing whitespace
        sanitized = ''.join('_' if c in invalid_chars else c for c in name)
        sanitized = sanitized.strip()
        # Remove control characters
        sanitized = ''.join(c for c in sanitized if c in string.printable and ord(c) >= 32)
        return sanitized
    #endregion

    #region H-COUNTER
    def get_latest_counter(self, folder_path, filename_prefix, counter_digits,
                           counter_position='last', delimiter_char='_'):
        # Determine the next counter value by scanning existing PNG filenames
        counter = 1
        if not os.path.exists(folder_path):
            print(f"[SaveImageExtendedFolderPath] Folder {folder_path} does not exist, "
                  f"starting counter at 1.")
            return counter

        try:
            files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
            if files:
                if counter_position == 'last':
                    counters = [int(f[-(4 + counter_digits):-4])
                                if f[-(4 + counter_digits):-4].isdigit() else 0
                                for f in files if f.startswith(filename_prefix)]
                elif counter_position == 'first':
                    # Skip the counter digits plus the delimiter char, then match the prefix
                    skip = counter_digits + len(delimiter_char)
                    counters = [int(f[:counter_digits]) if f[:counter_digits].isdigit() else 0
                                for f in files if f[skip:].startswith(filename_prefix)]
                else:
                    print("[SaveImageExtendedFolderPath] Invalid counter_position. "
                          "Using 'last' as default.")
                    counters = [int(f[-(4 + counter_digits):-4])
                                if f[-(4 + counter_digits):-4].isdigit() else 0
                                for f in files if f.startswith(filename_prefix)]

                if counters:
                    counter = max(counters) + 1

        except Exception as e:
            print(f"[SaveImageExtendedFolderPath] An error occurred while finding "
                  f"the latest counter: {e}")

        return counter
    #endregion

    #region H-SUBFOLDER
    def get_subfolder_path(self, image_path, output_path):
        # Compute the subfolder portion of the saved path relative to the base
        try:
            image_path = Path(image_path).resolve()
            output_path = Path(output_path).resolve()
            relative_path = image_path.relative_to(output_path)
            subfolder_path = relative_path.parent
            return str(subfolder_path)
        except Exception as e:
            print(f"[SaveImageExtendedFolderPath] Error in get_subfolder_path: {e}")
            return ""
    #endregion
    #endregion

    #region UI
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'images': ('IMAGE', ),
                'folderpath_input': ('STRING', {'default': 'c:/'}),
                'foldername_prefix': ('STRING', {'default': 'gen'}),
                'filename_prefix': ('STRING', {'default': 'output'}),
                'delimiter': (['underscore', 'dot', 'comma'], {'default': 'underscore'}),
                'save_metadata': (['disabled', 'enabled'], {'default': 'enabled'}),
                'counter_digits': ([2, 3, 4, 5, 6], {'default': 3}),
                'counter_position': (['first', 'last'], {'default': 'last'}),
            },
            'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'},
        }
    #endregion
#endregion


#region REGISTRY
NODE_CLASS_MAPPINGS = {
    'illumoraeSaveImageExtendedFolderPathNode': illumoraeSaveImageExtendedFolderPathNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'illumoraeSaveImageExtendedFolderPathNode': 'Save Image Extended FolderPath',
}
#endregion
