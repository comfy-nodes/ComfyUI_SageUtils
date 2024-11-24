import torch

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
    
    CATEGORY = "Sage Utils/Primitives"
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
    
    CATEGORY = "Sage Utils/Primitives"
    DESCRIPTION = "Sets an float."
    
    def pass_float(self, float):
        return (float,)

class Sage_SetText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "str": ("STRING", {"defaultInput": False, "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    
    FUNCTION = "pass_str"
    
    CATEGORY = "Sage Utils/Primitives"
    DESCRIPTION = "Sets some text."
    
    def pass_str(self, str):
        return (str,)

class Sage_JoinText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "separator": ("STRING", {"defaultInput": False}),
                "str1": ("STRING", {"defaultInput": True, "multiline": True}),
                "str2": ("STRING", {"defaultInput": True, "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("str",)
    
    FUNCTION = "join_str"
    
    CATEGORY = "Sage Utils/Primitives"
    DESCRIPTION = "Joins two strings with a separator."
    
    def join_str(self, separator, str1, str2):
        return (separator.join([str1, str2]),)

# Commented out in __init__.py, because it doesn't currently work.
class Sage_ViewText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "str": ("STRING", {"defaultInput": True}),
            }
        }
    
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    
    FUNCTION = "show_str"
    
    CATEGORY = "Sage Utils/Primitives"
    DESCRIPTION = "Shows some text."
    OUTPUT_NODE = True
    INPUT_IS_LIST = (True,)
    
    def show_str(self, str):
       return ({"ui": {"text": [f"{str}"]}})

class Sage_PonyPrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rating": (["safe", "questionable", "explicit"], {"defaultInput": False}),
                "source": (["pony", "furry", "anime", "cartoon"], {"defaultInput": False}),
                "prompt": ("STRING", {"defaultInput": False, "multiline": True})
            }
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "create_prefix"

    CATEGORY = "Sage Utils/Util"

    def create_prefix(self, rating, source, prompt):
        return (f"score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, source_{source}, rating_{rating}, {prompt}",)

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

    CATEGORY = "Sage Utils/Primitives"
    DESCRIPTION = "Returns zeroed out conditioning."

    def zero_out(self, clip):
        c = []
        tokens = clip.tokenize("")
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        conditioning = output.pop("cond")
        pooled_output = output.get("pooled_output", None)

        if pooled_output is not None:
            output["pooled_output"] = torch.zeros_like(pooled_output)

        n = [torch.zeros_like(conditioning), output]
        c.append(n)
        return (c, )