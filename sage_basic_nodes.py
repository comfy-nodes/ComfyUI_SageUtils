import torch
import comfy
import nodes

class Sage_SetBool:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bool": ("BOOLEAN", {"defaultInput": False}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    
    FUNCTION = "pass_bool"

    CATEGORY = "Sage Utils/primitives"
    DESCRIPTION = "Sets an boolean."
    
    def pass_bool(self, bool):
        return (bool,)
    
class Sage_SetInteger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int": ("INT", {"defaultInput": False}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    
    FUNCTION = "pass_int"

    CATEGORY = "Sage Utils/primitives"
    DESCRIPTION = "Sets an integer."
    
    def pass_int(self, int):
        return (int,)
    
class Sage_SetFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float": ("FLOAT", {"defaultInput": False}),
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    
    FUNCTION = "pass_float"
    
    CATEGORY = "Sage Utils/primitives"
    DESCRIPTION = "Sets an float."
    
    def pass_float(self, float):
        return (float,)

class Sage_SetText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "str": ("STRING", {"defaultInput": False, "multiline": True})
            },
            "optional": {
                "prefix": ("STRING", {"defaultInput": True, "multiline": True}),
                "suffix": ("STRING", {"defaultInput": True, "multiline": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    
    FUNCTION = "pass_str"
    
    CATEGORY = "Sage Utils/text"
    DESCRIPTION = "Sets some text."
    
    def pass_str(self, str, prefix=None, suffix=None):
        return (f"{prefix or ''}{str}{suffix or ''}",)

class Sage_JoinText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "separator": ("STRING", {"defaultInput": False, "default": ', '}),
                "str1": ("STRING", {"defaultInput": True, "multiline": True}),
                "str2": ("STRING", {"defaultInput": True, "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    
    FUNCTION = "join_str"
    
    CATEGORY = "Sage Utils/text"
    DESCRIPTION = "Joins two strings with a separator."
    
    def join_str(self, separator, str1, str2):
        return (separator.join([str1, str2]),)

class Sage_TripleJoinText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "separator": ("STRING", {"defaultInput": False, "default": ', '}),
                "str1": ("STRING", {"defaultInput": True, "multiline": True}),
                "str2": ("STRING", {"defaultInput": True, "multiline": True}),
                "str3": ("STRING", {"defaultInput": True, "multiline": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    
    FUNCTION = "join_str"
    
    CATEGORY = "Sage Utils/text"
    DESCRIPTION = "Joins three strings with a separator."
    
    def join_str(self, separator, str1, str2, str3):
        return (separator.join([str1, str2, str3]),)

class Sage_ViewText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    
    FUNCTION = "show_text"
    
    CATEGORY = "Sage Utils/text"
    DESCRIPTION = "Shows some text."
    OUTPUT_NODE = True
    
    def show_text(self, text):
        print(f"String is '{text}'")
        return { "ui": {"text": text}, "result" : (text,) }

class Sage_PonyPrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "add_score": ("BOOLEAN", {"defaultInput": False}),
                "rating": (["none", "safe", "questionable", "explicit"], {"defaultInput": False}),
                "source": (["none", "pony", "furry", "anime", "cartoon", "3d", "western", "comic", "monster"], {"defaultInput": False}),
            },
            "optional": {
                "prompt": ("STRING", {"defaultInput": True, "multiline": True})
            }
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "create_prefix"
    CATEGORY = "Sage Utils/text"

    def create_prefix(self, add_score, rating, source, prompt=None):
        prefix = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, " if add_score else ""
        prefix += f"source_{source}, " if source != "none" else ""
        prefix += f"rating_{rating}, " if rating != "none" else ""
        prefix += f"{prompt or ''}"
        return (prefix,)

class Sage_ConditioningZeroOut:
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
    
class Sage_EmptyLatentImagePassthrough:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "width": ("INT", {"defaultInput": True, "default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8, "tooltip": "The width of the latent images in pixels."}),
                "height": ("INT", {"defaultInput": True, "default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8, "tooltip": "The height of the latent images in pixels."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),
                "sd3": ("BOOLEAN", {"default": False})
            }
        }
    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    OUTPUT_TOOLTIPS = ("The empty latent image batch.", "pass through the image width", "pass through the image height")
    FUNCTION = "generate"

    CATEGORY = "Sage Utils/image"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def generate(self, width, height, batch_size=1, sd3=False):
        size = 16 if sd3 else 4
        latent = torch.zeros([batch_size, size, height // 8, width // 8], device=self.device)
        return ({"samples": latent}, width, height)
