# Metadata nodes.
# This includes nodes for constructing metadata, and related nodes. Saving metadata is handled in the image nodes.

import numpy as np
import os

import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfyui_version import __version__

from ..sage import *

class Sage_ConstructMetadata(ComfyNodeABC):
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_info": ('MODEL_INFO',{ "defaultInput": True}),
                "sampler_info": ('SAMPLER_INFO', { "defaultInput": True}),
                "width": ('INT', { "defaultInput": True}),
                "height": ('INT', { "defaultInput": True}),
                "positive_string": ('STRING', {"multiline": True, "defaultInput": True}),
            },
            "optional": {
                "negative_string": ('STRING', {"multiline": True, "defaultInput": True}),
                "clip_skip": ('INT', {"default": 1, "min": 1, "max": 24, "step": 1, "defaultInput": True}),
                "lora_stack": ('LORA_STACK', {"defaultInput": True}),
                "civitai_format": ('BOOLEAN', {"defaultInput": False, "default": False}),
            },
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('param_metadata',)
    FUNCTION = "construct_metadata"
    
    CATEGORY = "Sage Utils/metadata"
    DESCRIPTION = "Puts together metadata in a A1111-like format. Uses the custom sampler info node. The return value is a string, so can be manipulated by other nodes."
    
    def construct_metadata(self, model_info, sampler_info, width, height, positive_string, negative_string="", civitai_format=False, clip_skip=1, lora_stack=None):
        metadata = ''

        lora_hashes = []
        lora_hash_string = ''

        resource_hashes = []
        civitai_resources = ''
        
        sampler_name = civitai_sampler_name(sampler_info['sampler'], sampler_info['scheduler'])
        
        if lora_stack is not None:
            # We're going through generating A1111 style <lora> tags to insert in the prompt, adding the lora hashes to the resource hashes in exactly the format
            # that CivitAI's approved extension for A1111 does, and inserting the Lora hashes at the end in the way they appeared looking at the embedded metadata
            # generated by Forge.
            for lora in lora_stack:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora[0])
                lora_name = os.path.splitext(os.path.basename(lora_path))[0]
                pull_metadata(lora_path)
                lora_data = get_model_info(lora_path, lora[1])
                if lora_data != {}:
                    resource_hashes.append(lora_data)
                
                lora_hash = cache.cache.data[lora_path]["hash"]
                lora_hashes += [f"{lora_name}: {lora_hash}"]
        
        if civitai_format:
            if len(resource_hashes) >=1:
                civitai_resources = f", Civitai resources: {json.dumps(resource_hashes)}"
            metadata = f"{positive_string}\n"
        else:
            if len(lora_hashes) >= 1:
                lora_hash_string = ', Lora hashes: "' + ", ".join(lora_hashes) + '"'
            metadata = f"{positive_string}{lora_to_prompt(lora_stack)}\n"

        if negative_string != "":
            metadata += f"Negative prompt: {negative_string}\n"

        clip_skip = "" if clip_skip <= 1 else f"Clip skip: {clip_skip}, "
        metadata += f"Steps: {sampler_info['steps']}, Sampler: {sampler_name}, Scheduler type: {sampler_info['scheduler']}, CFG scale: {sampler_info['cfg']}, Seed: {sampler_info['seed']}, Size: {width}x{height}, {clip_skip}"
        metadata += f"Model: {name_from_path(model_info['path'])}, Model hash: {model_info['hash']}, Version: ComfyUI {__version__}" + civitai_resources + lora_hash_string
        return metadata,

class Sage_ConstructMetadataLite(ComfyNodeABC):
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_info": ('MODEL_INFO',{ "defaultInput": True}),
                "positive_string": ('STRING',{ "defaultInput": True}),
                "negative_string": ('STRING',{ "defaultInput": True}),
                "sampler_info": ('SAMPLER_INFO', { "defaultInput": True}),
                "width": ('INT', { "defaultInput": True}),
                "height": ('INT', { "defaultInput": True})
            },
            "optional": {
                "lora_stack": ('LORA_STACK',{ "defaultInput": True})
            },
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('param_metadata',)
    FUNCTION = "construct_metadata"
    
    CATEGORY = "Sage Utils/metadata"
    DESCRIPTION = "Puts together metadata in a A1111-like format. Uses the custom sampler info node. The return value is a string, so can be manipulated by other nodes."
    
    def construct_metadata(self, model_info, positive_string, negative_string, width, height, sampler_info, lora_stack = None):
        metadata = ''
        
        resource_hashes = []
        
        sampler_name = civitai_sampler_name(sampler_info['sampler'], sampler_info['scheduler'])
        resource_hashes.append(get_model_info(model_info['path']))
        
        if lora_stack is not None:
            # We're going through generating A1111 style prompt information, but not doing the loras and model A1111 style, rather
            # just adding the lora and model information in the resource section.
            for lora in lora_stack:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora[0])
                pull_metadata(lora_path)
                lora_data = get_model_info(lora_path, lora[1])
                if lora_data != {}:
                    resource_hashes.append(lora_data)

        metadata = f"{positive_string}" + "\n" 
        if negative_string != "": metadata += f"Negative prompt: {negative_string}" + "\n"
        metadata += f"Steps: {sampler_info['steps']}, Sampler: {sampler_name}, Scheduler type: {sampler_info['scheduler']}, CFG scale: {sampler_info['cfg']}, Seed: {sampler_info['seed']}, Size: {width}x{height},"
        metadata += f"Version: ComfyUI {__version__}, Civitai resources: {json.dumps(resource_hashes)}"
        return metadata,
