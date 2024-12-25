# File for any loader nodes. Separating out as the number is likely to grow.

from .sage_helpers import *
import ComfyUI_SageUtils.sage_cache as cache

import torch
import pathlib
import numpy as np
import json
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

class Sage_CacheMaintenance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "remove_ghost_entries": ("BOOLEAN", {"defaultInput": True})
            }
        }
        
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("ghost_entries", "dup_hash","dup_model")
    
    FUNCTION = "cache_maintenance"
    CATEGORY = "Sage Utils/model"
    DESCRIPTION = "Lets you remove entries for models that are no longer there. dup_hash returns a list of files with the same hash, and dup_model returns ones with the same civitai model id (but not neccessarily the same version)."

    def cache_maintenance(self, remove_ghost_entries):
        ghost_entries = [path for path in cache.cache_data if not pathlib.Path(path).is_file()]
        cache_by_hash = {}
        cache_by_id = {}
        dup_hash = {}
        dup_id = {}

        for model_path, data in cache.cache_data.items():
            if 'hash' in data:
                cache_by_hash.setdefault(data['hash'], []).append(model_path)
            if 'modelId' in data:
                cache_by_id.setdefault(data['modelId'], []).append(model_path)

        if remove_ghost_entries:
            for ghost in ghost_entries:
                cache.cache_data.pop(ghost)
            cache.save_cache()

        dup_hash = {h: paths for h, paths in cache_by_hash.items() if len(paths) > 1}
        dup_id = {i: paths for i, paths in cache_by_id.items() if len(paths) > 1}

        return (", ".join(ghost_entries), json.dumps(dup_hash, separators=(",", ":"), sort_keys=True, indent=4), json.dumps(dup_id, separators=(",", ":"), sort_keys=True, indent=4))

class Sage_ModelReport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scan_models": (("none", "loras", "checkpoints", "all"), {"defaultInput": False, "default": "none"}),
            }
        }
        
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_list", "lora_list")
    
    FUNCTION = "pull_list"
    CATEGORY = "Sage Utils/model"
    DESCRIPTION = "Calculates the hash of models & checkpoints & pulls civitai info if chosen. Returns a list of models in the cache of the specified type, by base model type."

    def get_files(self, scan_models):
        the_paths = []
        if scan_models == "loras":
            the_paths = folder_paths.get_folder_paths("loras")
        elif scan_models == "checkpoints":
            the_paths = folder_paths.get_folder_paths("checkpoints")
        elif scan_models == "all":
            the_lora_paths = folder_paths.get_folder_paths("loras")
            the_checkpoint_paths = folder_paths.get_folder_paths("checkpoints")
            the_paths = [*the_lora_paths, *the_checkpoint_paths]
        
        print(f"Scanning {len(the_paths)} paths.")
        print(f"the_paths == {the_paths}")
        if the_paths != []: model_scan(the_paths)
    
    def pull_list(self, scan_models):
        sorted_models = {}
        sorted_loras = {}
        model_list = ""
        lora_list = ""
        
        self.get_files(scan_models)
        
        for model_path in cache.cache_data.keys():
            cur = cache.cache_data.get(model_path, {})
            baseModel = cur.get('baseModel', None)
            if cur.get('model', {}).get('type', None) == "Checkpoint":
                if baseModel not in sorted_models: sorted_models[baseModel] = []
                sorted_models[baseModel].append(model_path)
            if cur.get('model', {}).get('type', None) == "LORA":
                if baseModel not in sorted_loras: sorted_loras[baseModel] = []
                sorted_loras[baseModel].append(model_path)

        if sorted_models != {}: model_list = json.dumps(sorted_models, separators=(",", ":"), sort_keys=True, indent=4)
        if sorted_loras != {}: lora_list = json.dumps(sorted_loras, separators=(",", ":"), sort_keys=True, indent=4)
        
        return (model_list, lora_list)
