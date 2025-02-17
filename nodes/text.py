# Text nodes.
# This contains any nodes that are dealing with text, including setting text, joining text, cleaning text, and viewing text.

from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from ..sage import *

class Sage_SetText(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "str": ("STRING", {"defaultInput": False, "dynamicPrompts": True, "multiline": True})
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

class Sage_JoinText(ComfyNodeABC):
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

class Sage_TripleJoinText(ComfyNodeABC):
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

class Sage_CleanText(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "str": ("STRING", {"defaultInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_string",)

    FUNCTION = "clean_str"

    CATEGORY = "Sage Utils/text"
    DESCRIPTION = "Cleans up the string given."

    def clean_str(self, str):
        ret_list = [x for x in str.split(" ") if x.strip()]
        ret = " ".join(ret_list)

        ret_list = [x for x in ret.split(",") if x.strip()]
        ret = ", ".join([x.strip(" ") for x in ret_list])

        ret_list = [x for x in ret.split("\n") if x.strip()]
        ret = "\n".join([x.strip(" ") for x in ret_list])
        return (ret,)

class Sage_ViewText(ComfyNodeABC):
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

class Sage_PonyPrefix(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "score": (["none", "best", "good", "normal", "all"], {"defaultInput": False},),
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

    def create_prefix(self, score, rating, source, prompt=None):
        if score == "best":
            add_score = "score_9"
        elif score == "good":
            add_score = "score_9, score_8_up, score_7_up"
        elif score == "normal":
            add_score = "score_9, score_8_up, score_7_up, score_6_up, score_5_up"
        elif score == "all":
            add_score = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
        prefix = f"{add_score}, " if score != "none" else ""
        prefix += f"rating_{rating}, " if rating != "none" else ""
        prefix += f"source_{source}, " if source != "none" else ""
        prefix += f"{prompt or ''}"
        return (prefix,)
