# For nodes that involve pulling or working with model info.

import json
import torch

from ..sage import *
import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict


# Copied from https://github.com/lzyhha/ComfyUI-Lumina2/blob/master/comfy_extras/nodes_model_advanced.py#L298
# Temporary, until this issue is resolved: https://github.com/comfyanonymous/ComfyUI/issues/6741
class Sage_RenormCFG():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "cfg_trunc": ("FLOAT", {"default": 100, "min": 0.0, "max": 100.0, "step": 0.01}),
                              "renorm_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"
    def patch(self, model, cfg_trunc, renorm_cfg):
        def renorm_cfg_func(args):
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            timestep = args["timestep"]
            x_orig = args["input"]
            in_channels = model.model.diffusion_model.in_channels
            if timestep[0] < cfg_trunc:
                cond_eps, uncond_eps = cond_denoised[:, :in_channels], uncond_denoised[:, :in_channels]
                cond_rest, uncond_rest = cond_denoised[:, in_channels:], uncond_denoised[:, in_channels:]
                half_eps = uncond_eps + cond_scale * (cond_eps - uncond_eps)
                half_rest = cond_rest
                if float(renorm_cfg) > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(cond_eps
                            , dim=tuple(range(1, len(cond_eps.shape))), keepdim=True
                    )
                    max_new_norm = ori_pos_norm * float(renorm_cfg)
                    new_pos_norm = torch.linalg.vector_norm(
                            half_eps, dim=tuple(range(1, len(half_eps.shape))), keepdim=True
                        )
                    if new_pos_norm >= max_new_norm:
                        half_eps = half_eps * (max_new_norm / new_pos_norm)
            else:
                cond_eps, uncond_eps = cond_denoised[:, :in_channels], uncond_denoised[:, :in_channels]
                cond_rest, uncond_rest = cond_denoised[:, in_channels:], uncond_denoised[:, in_channels:]
                half_eps = cond_eps
                half_rest = cond_rest

            cfg_result = torch.cat([half_eps, half_rest], dim=1)
            # cfg_result = uncond_denoised + (cond_denoised - uncond_denoised) * cond_scale
            return x_orig - cfg_result
        m = model.clone()
        m.set_model_sampler_cfg_function(renorm_cfg_func)
        return (m, )

class Sage_ModelInfo(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO", {"defaultInput": True})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("base_model", "name", "url", "latest_url", "image")

    FUNCTION = "get_last_info"

    CATEGORY = "Sage Utils/model"
    DESCRIPTION = "Pull the civitai model info, and return what the base model is, the name with version, the url, the url for the latest version, and a preview image. Note that last model in the stack is not necessarily the one this node is hooked to, since that node may be disabled."

    def get_last_info(self, model_info):
        if model_info is None:
            return ("", "", "", "", None)

        image = blank_image()
        try:
            json_data = get_civitai_model_version_json(model_info["hash"])
            if "modelId" in json_data:
                url = f"https://civitai.com/models/{json_data['modelId']}?modelVersionId={json_data['id']}"
                latest_version = get_latest_model_version(json_data["modelId"])
                latest_url = f"https://civitai.com/models/{json_data['modelId']}?modelVersionId={latest_version}"
                image_urls = pull_lora_image_urls(model_info["hash"], True)
                image = url_to_torch_image(image_urls[0])
            else:
                url = ""
                latest_url = ""

            return (
                json_data.get("baseModel", ""),
                json_data.get("model", {}).get("name", "") + " " + json_data.get("name", ""),
                url,
                latest_url,
                image)
        except:
            print("Exception when getting json data.")
            return ("", "", "", "", image)

class Sage_LastLoraInfo(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_stack": ("LORA_STACK", {"defaultInput": True})
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("base_model", "name", "url", "latest_url", "image")

    FUNCTION = "get_last_info"

    CATEGORY = "Sage Utils/lora"
    DESCRIPTION = "Take the last lora in the stack, pull the civitai model info, and return what the base model is, the name with version, the url, the url for the latest version, and a preview image. Note that last model in the stack is not necessarily the one this node is hooked to, since that node may be disabled."

    def get_last_info(self, lora_stack):
        if lora_stack is None:
            return ("", "", "", "", None)

        last_lora = lora_stack[-1]
        image = blank_image()
        try:
            hash = get_lora_hash(last_lora[0])
            json_data = get_civitai_model_version_json(hash)
            if "modelId" in json_data:
                url = f"https://civitai.com/models/{json_data['modelId']}?modelVersionId={json_data['id']}"
                latest_version = get_latest_model_version(json_data["modelId"])
                latest_url = f"https://civitai.com/models/{json_data['modelId']}?modelVersionId={latest_version}"
                image_urls = pull_lora_image_urls(hash, True)
                image = url_to_torch_image(image_urls[0])
            else:
                url = ""
                latest_url = ""

            return (
                json_data.get("baseModel", ""),
                json_data.get("model", {}).get("name", "") + " " + json_data.get("name", ""),
                url,
                latest_url,
                image)
        except:
            print("Exception when getting json data.")
            return ("", "", "", "", image)

class Sage_GetFileHash(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_dir": (list(folder_paths.folder_names_and_paths.keys()), {"defaultInput": False}),
                "filename": ("STRING", {"defaultInput": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hash",)

    FUNCTION = "get_hash"

    CATEGORY = "Sage Utils/util"
    DESCRIPTION = "Get an sha256 hash of a file."

    def get_hash(self, base_dir, filename):
        the_hash = ""
        try:
            file_path = folder_paths.get_full_path_or_raise(base_dir, filename)
            pull_metadata(file_path)
            the_hash = cache.cache.data[file_path]["hash"]
        except:
            print(f"Unable to hash file '{file_path}'. \n")
            the_hash = ""

        print(f"Hash for '{file_path}': {the_hash}")
        return (str(the_hash),)
