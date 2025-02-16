# Conditioning nodes
# This will include any nodes involving clip or conditioning.

import torch

import comfy
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

from ..sage import *

class Sage_ConditioningZeroOut(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "clip": ("CLIP", {"defaultInput": True, "tooltip": "The CLIP model used for encoding."}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "zero_out"

    CATEGORY = "Sage Utils/clip"
    DESCRIPTION = "Returns zeroed out conditioning."
    def zero_out(self, clip):
        tokens = clip.tokenize("")
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        output["pooled_output"] = torch.zeros_like(output.get("pooled_output", torch.tensor([])))
        conditioning = torch.zeros_like(output.pop("cond"))
        return [([conditioning, output],)]

class Sage_ConditioningOneOut(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "clip": ("CLIP", {"defaultInput": True, "tooltip": "The CLIP model used for encoding."}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "one_out"

    CATEGORY = "Sage Utils/clip"
    DESCRIPTION = "Returns oned out conditioning."
    
    EXPERIMENTAL = True
    
    def zero_out(self, clip):
        tokens = clip.tokenize("")
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        output["pooled_output"] = torch.ones_like(output.get("pooled_output", torch.tensor([])))
        conditioning = torch.ones_like(output.pop("cond"))
        return [([conditioning, output],)]
    
class Sage_ConditioningRngOut(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            "clip": ("CLIP", {"defaultInput": True, "tooltip": "The CLIP model used for encoding."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "defaultInput": True, "tooltip": "The seed used to randomize the conditioning."})
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "rng_out"

    CATEGORY = "Sage Utils/clip"
    DESCRIPTION = "Returns randomized conditioning."
    
    EXPERIMENTAL = True
    
    def rng_out(self, clip, seed):
        torch.manual_seed(seed)
        tokens = clip.tokenize("")
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        output["pooled_output"] = torch.rand_like(output.get("pooled_output", torch.tensor([])))
        conditioning = torch.rand_like(output.pop("cond"))
        return [([conditioning, output],)]  


class Sage_DualCLIPTextEncode(ComfyNodeABC):
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
        #print(f"tokens = {tokens['g'].size()}")
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