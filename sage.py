# A collection of utility nodes, many of which are dealing with metadata and collecting it.
import os
import json
import pathlib
import numpy as np
import torch

# Pieces of ComfyUI that are being brought in for one reason or another.
import comfy
import folder_paths
import cli_args
import nodes

from .sage_helpers import *
import ComfyUI_SageUtils.sage_cache as cache

class Sage_DualCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"defaultInput": True, "tooltip": "The CLIP model used for encoding the text."})
            },
            "optional": {
                "pos": ("STRING", {"defaultInput": True, "multiline": True, "dynamicPrompts": True, "tooltip": "The positive prompt's text."}), 
                "neg": ("STRING", {"defaultInput": True, "multiline": True, "dynamicPrompts": True, "tooltip": "The negative prompt's text."}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("pos_cond", "neg_cond", "pos_text", "neg_text")
    
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model. If neg is not hooked up, it'll be automatically zeroed.",)
    FUNCTION = "encode"

    CATEGORY = "Sage Utils/clip"
    DESCRIPTION = "Turns a positive and negative prompt into conditionings, and passes through the prompts. Saves space over two CLIP Text Encoders, and zeros any input not hooked up."

    def get_conditioning(self, pbar, clip, text=None):
        zero_text = text is None
        text = text or ""

        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        pbar.update(1)

        if zero_text:
            pooled_output = output.get("pooled_output")
            if pooled_output is not None:
                output["pooled_output"] = torch.zeros_like(pooled_output)
            return [[torch.zeros_like(cond), output]]
            
        return [[cond, output]]

    def encode(self, clip, pos=None, neg=None):
        pbar = comfy.utils.ProgressBar(2)
        return (
            self.get_conditioning(pbar, clip, pos),
            self.get_conditioning(pbar, clip, neg),
            pos or "",
            neg or ""
        )

class Sage_LoraStackRecent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        lora_list = get_recently_used_models("loras")
        return {
            "required": {
                "enabled": ("BOOLEAN", {"defaultInput": False, "default": True}),
                "lora_name": (lora_list, {"defaultInput": False, "tooltip": "The name of the LoRA."}),
                "model_weight": ("FLOAT", {"defaultInput": False, "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "clip_weight": ("FLOAT", {"defaultInput": False, "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                },
            "optional": {
                "lora_stack": ("LORA_STACK", {"defaultInput": True}),
            }
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    
    FUNCTION = "add_lora_to_stack"
    CATEGORY = "Sage Utils/lora"
    DESCRIPTION = "Choose a lora with weights, and add it to a lora_stack. Compatable with other node packs that have lora_stacks."
    
    def add_lora_to_stack(self, enabled, lora_name, model_weight, clip_weight, lora_stack = None):
        if enabled == True:
            stack = add_lora_to_stack(lora_name, model_weight, clip_weight, lora_stack)
        else:
            stack = lora_stack

        return (stack,)


class Sage_LoraStack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"defaultInput": False, "default": True}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"defaultInput": False, "tooltip": "The name of the LoRA."}),
                "model_weight": ("FLOAT", {"defaultInput": False, "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "clip_weight": ("FLOAT", {"defaultInput": False, "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                },
            "optional": {
                "lora_stack": ("LORA_STACK", {"defaultInput": True}),
            }
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    
    FUNCTION = "add_lora_to_stack"
    CATEGORY = "Sage Utils/lora"
    DESCRIPTION = "Choose a lora with weights, and add it to a lora_stack. Compatable with other node packs that have lora_stacks."
    
    def add_lora_to_stack(self, enabled, lora_name, model_weight, clip_weight, lora_stack = None):
        if enabled == True:
            stack = add_lora_to_stack(lora_name, model_weight, clip_weight, lora_stack)
        else:
            stack = lora_stack

        return (stack,)
