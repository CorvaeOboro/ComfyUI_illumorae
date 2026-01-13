"""
scans all subfolders with an __init__.py and merges their
NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

was setup this way to enable each node to be independent and able to 
be simply copied out of this folder if you only want specific nodes instead of the whole node pack

"""
import os
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

base = os.path.dirname(__file__)

for name in os.listdir(base):
    subdir = os.path.join(base, name)
    if (
        os.path.isdir(subdir)
        and not name.startswith("__")
        and os.path.isfile(os.path.join(subdir, "__init__.py"))
    ):
        try:
            mod = importlib.import_module(f".{name}", package=__package__)
            NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
            NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
        except Exception as e:
            print(f"Could not import node module {name}: {e}")