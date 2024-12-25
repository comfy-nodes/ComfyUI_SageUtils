# File for any loader nodes. Separating out as the number is likely to grow.

from .sage_helpers import *
import ComfyUI_SageUtils.sage_cache as cache

import torch
import pathlib
import numpy as np
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo

# Pieces of ComfyUI that are being brought in for one reason or another.
import comfy
import folder_paths
import node_helpers


class Sage_CheckpointLoaderRecent:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        ckpt_list = get_recently_used_models("checkpoints")
            
        return {
            "required": {
                "ckpt_name": (ckpt_list, {"tooltip": "The name of the checkpoint (model) to load."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "MODEL_INFO")
    RETURN_NAMES = ("model", "clip", "vae", "model_info")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.", 
                    "The CLIP model used for encoding text prompts.", 
                    "The VAE model used for encoding and decoding images to and from latent space.",
                    "The model path and hash, all in one output.")
    FUNCTION = "load_checkpoint"

    CATEGORY  =  "Sage Utils/model"
    DESCRIPTION = "Loads a diffusion model checkpoint. Also returns a model_info output to pass to the construct metadata node, and the hash. (And hashes and pulls civitai info for the file.)"

    def load_checkpoint(self, ckpt_name):
        model_info = { "path": folder_paths.get_full_path_or_raise("checkpoints", ckpt_name) }
        pull_metadata(model_info["path"], True)

        model_info["hash"] = cache.cache_data[model_info["path"]]["hash"]
    
        out = comfy.sd.load_checkpoint_guess_config(model_info["path"], output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        result = (*out[:3], model_info)
        return (result)
    
class Sage_CheckpointLoaderSimple:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "MODEL_INFO")
    RETURN_NAMES = ("model", "clip", "vae", "model_info")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.", 
                    "The CLIP model used for encoding text prompts.", 
                    "The VAE model used for encoding and decoding images to and from latent space.",
                    "The model path and hash, all in one output.")
    FUNCTION = "load_checkpoint"

    CATEGORY  =  "Sage Utils/model"
    DESCRIPTION = "Loads a diffusion model checkpoint. Also returns a model_info output to pass to the construct metadata node, and the hash. (And hashes and pulls civitai info for the file.)"

    def load_checkpoint(self, ckpt_name):
        model_info = { "path": folder_paths.get_full_path_or_raise("checkpoints", ckpt_name) }
        pull_metadata(model_info["path"], True)

        model_info["hash"] = cache.cache_data[model_info["path"]]["hash"]
    
        out = comfy.sd.load_checkpoint_guess_config(model_info["path"], output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        result = (*out[:3], model_info)
        return (result)

class Sage_UNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
                            }}
    RETURN_TYPES = ("MODEL", "MODEL_INFO")
    RETURN_NAMES = ("model", "model_info")

    FUNCTION = "load_unet"
    CATEGORY  =  "Sage Utils/model"

    def load_unet(self, unet_name, weight_dtype):
        dtype_map = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e4m3fn_fast": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2
        }
        model_options = {"dtype": dtype_map.get(weight_dtype)}
        if weight_dtype == "fp8_e4m3fn_fast":
            model_options["fp8_optimizations"] = True

        model_info = {
            "name": pathlib.Path(unet_name).name,
            "path": folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        }
        pull_metadata(model_info["path"], True)
        model_info["hash"] = cache.cache_data[model_info["path"]]["hash"]

        model = comfy.sd.load_diffusion_model(model_info["path"], model_options=model_options)
        return model, model_info

# Modified version of the main lora loader.
class Sage_LoraStackLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."})
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"defaultInput": True})
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.", "The stack of loras.")
    FUNCTION = "load_loras"

    CATEGORY = "Sage Utils/lora"
    DESCRIPTION = "Accept a lora_stack with Model and Clip, and apply all the loras in the stack at once."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if not (strength_model or strength_clip):
            return model, clip

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        
        if self.loaded_lora and self.loaded_lora[0] == lora_path:
            lora = self.loaded_lora[1]
        else:
            pull_metadata(lora_path, True)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
    
    def load_loras(self, model, clip, lora_stack=None):
        if not lora_stack:
            print("No lora stacks found. Warning: Passing 'None' to lora_stack output.")
            return model, clip, None
        pbar = comfy.utils.ProgressBar(len(lora_stack))

        for lora in lora_stack:
            if lora:
                model, clip = self.load_lora(model, clip, *lora)
            pbar.update(1)
        return model, clip, lora_stack
    
class Sage_LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        files = sorted(str(x.relative_to(folder_paths.get_input_directory())) 
                        for x in pathlib.Path(folder_paths.get_input_directory()).rglob('*') if x.is_file())
        return {"required": {"image": (files, {"image_upload": True})}}

    CATEGORY = "Sage Utils/image"

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "width", "height", "metadata")

    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images, output_masks = [], []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == 'I':
                i = i.point(lambda x: x * (1 / 255))
            image = i.convert("RGB")

            if not output_images:
                w, h = image.size
            
            if image.size != (w, h):
                continue
            
            image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
            mask = (1. - torch.from_numpy(np.array(i.getchannel('A')).astype(np.float32) / 255.0)).unsqueeze(0) if 'A' in i.getbands() else torch.zeros((1, 64, 64), dtype=torch.float32)
            output_images.append(image)
            output_masks.append(mask)

        output_image = torch.cat(output_images, dim=0) if len(output_images) > 1 and img.format != 'MPO' else output_images[0]
        output_mask = torch.cat(output_masks, dim=0) if len(output_masks) > 1 and img.format != 'MPO' else output_masks[0]

        return output_image, output_mask, w, h, f"{img.info}"

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True