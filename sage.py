# A collection of utility nodes, many of which are dealing with metadata and collecting it.

# __init__.py is the entry point for the Sage Utils package. 
#  It contains the node definitions and the display names for the nodes. 
#  It also loads the cache and styles for the nodes. 

# It also imports this file, which imports the others.

# Any files with "nodes" in the name are node definitions.

import importlib
import os
import pathlib

sage_sage_path = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).name

sage_cache_path = sage_sage_path + ".utils.cache"
sage_styles_path = sage_sage_path + ".utils.styles"

cache = importlib.import_module(sage_cache_path)
sage_styles = importlib.import_module(sage_styles_path)

from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

from .utils.helpers import *
from .nodes.metadata import *
from .nodes.basic import *
from .nodes.model import *
from .nodes.util import *
from .nodes.lora import *