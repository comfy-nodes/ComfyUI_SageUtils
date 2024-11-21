# A collection of utility nodes, many of which are dealing with metadata and collecting it.
import os
import json
import pathlib
import numpy as np

from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Pieces of ComfyUI that are being brought in for one reason or another.
import comfy
import folder_paths
import cli_args
import nodes

from .sage_utils import *

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

class Sage_GetFileHash:
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
    
    CATEGORY = "Sage Utils/Util"
    DESCRIPTION = "Get an sha256 hash of a file. Can be used for detecting models, civitai calls and such."
    
    def get_hash(self, base_dir, filename):
        the_hash = ""
        try:
            file_path = folder_paths.get_full_path_or_raise(base_dir, filename)
            the_hash = get_file_sha256(file_path)
        except:
            print(f"Unable to hash file '{file_path}'. \n")
            the_hash = ""
        
        print(f"Hash for '{file_path}': {the_hash}")
        return (str(the_hash),)
    
class Sage_DualCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"defaultInput": True, "tooltip": "The CLIP model used for encoding the text."}),
                "pos": ("STRING", {"defaultInput": True, "default": "score_9, score_8_up, scoure_7_up, score_6_up, score_5_up, score_4_up", "multiline": True, "dynamicPrompts": True, "tooltip": "The positive prompt's text."}), 
                "neg": ("STRING", {"defaultInput": True, "default": "signature, watermark", "multiline": True, "dynamicPrompts": True, "tooltip": "The negative prompt's text."})
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("pos_cond", "neg_cond", "pos_text", "neg_text")
    
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "Sage Utils"
    DESCRIPTION = "Turns a positive and negative prompt into conditionings, to save space."

    def encode(self, clip, pos, neg):
        pos_tokens = clip.tokenize(pos)
        pos_output = clip.encode_from_tokens(pos_tokens, return_pooled=True, return_dict=True)
        pos_cond = pos_output.pop("cond")
        
        neg_tokens = clip.tokenize(neg)
        neg_output = clip.encode_from_tokens(neg_tokens, return_pooled=True, return_dict=True)
        neg_cond = neg_output.pop("cond")
        
        return ([[pos_cond, pos_output]], [[neg_cond, neg_output]], pos, neg)

class Sage_SamplerInfo:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m", "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta", "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
            }
        }

    RETURN_TYPES = ("SAMPLER_INFO",)
    OUTPUT_TOOLTIPS = ("To be piped to the Construct Metadata node.",)
    FUNCTION = "pass_info"

    CATEGORY = "Sage Utils"
    DESCRIPTION = "Grabs most of the sampler info and passes it along."

    def pass_info(self, seed, steps, cfg, sampler_name, scheduler):
        s_info = {}
        s_info["seed"] = seed
        s_info["steps"] = steps
        s_info["cfg"] = cfg
        s_info["sampler"] = sampler_name
        s_info["scheduler"] = scheduler
        return s_info,

class Sage_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "sampler_info": ('SAMPLER_INFO', { "defaultInput": True}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "Sage Utils"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, sampler_info, positive, negative, latent_image, denoise=1.0):
        return nodes.common_ksampler(model, sampler_info["seed"], sampler_info["steps"], sampler_info["cfg"], sampler_info["sampler"], sampler_info["scheduler"], positive, negative, latent_image, denoise=denoise)

class Sage_ConstructMetadata:
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
    
    CATEGORY = "Sage Utils"
    DESCRIPTION = "Puts together metadata from data provided by other nodes in Sage Utils."
    
    def construct_metadata(self, model_info, positive_string, negative_string, width, height, sampler_info, lora_stack = None):
        metadata = ''
        lora_info = ''
        resource_hashes = {}
        lora_hashes = ', Lora hashes: '
        
        resource_hashes['model'] = model_info['hash']
        sampler_name = civitai_sampler_name(sampler_info['sampler'], sampler_info['scheduler'])
        
        if lora_stack is None:
            lora_info = ""
        else:
            # We're going through generating A1111 style <lora> tags to insert in the prompt, adding the lora hashes to the resource hashes in exactly the format
            # that CivitAI's approved extension for A1111 does, and inserting the Loar hashes at the end in the way they appeared looking at the embedded metadata
            # generated by Forge.
            for lora in lora_stack:
                lora_info += lora_to_string(lora[0], lora[1], lora[2])
                lora_path = folder_paths.get_full_path_or_raise("loras", lora[0])
                lora_name = str(pathlib.Path(lora_path).name)
                lora_hash = get_file_sha256(lora_path)
                resource_hashes[f"lora:{lora_name}"] = lora_hash
                lora_hashes += f"{lora_name}: {lora_hash},"
        
        metadata = f"{positive_string} {lora_info}" + "\n" 
        metadata += f"Negative prompt: {negative_string}" + "\n"
        metadata += f"Steps: {sampler_info['steps']}, Sampler: {sampler_name}, Scheduler type: {sampler_info['scheduler']}, CFG scale: {sampler_info['cfg']}, Seed: {sampler_info['seed']}, Size: {width}x{height},"
        metadata += f"Model: {model_info['name']}, Model hash: {model_info['hash']}, Version: v1.10-RC-6-actually-totally-comfyui, Hashes: {json.dumps(resource_hashes)}{lora_hashes}"
        return metadata,

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
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "MODEL_INFO", )
    RETURN_NAMES = ("model", "clip", "vae", "model_info")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.", 
                       "The CLIP model used for encoding text prompts.", 
                       "The VAE model used for encoding and decoding images to and from latent space.",
                       "The name of the model.")
    FUNCTION = "load_checkpoint"

    CATEGORY  =  "Sage Utils"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_name):
        model_info = { "full_name": ckpt_name }
        model_info["path"] = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        model_info["name"] = pathlib.Path(model_info["full_name"]).name
        model_info["hash"] = get_file_sha256(model_info["path"])
    
        out = comfy.sd.load_checkpoint_guess_config(model_info["path"], output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        result = (*out[:3], model_info)
        return (result)
 
class Sage_LoraStackDebugString:
    def __init__(self):
         pass
     
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_stack": ("LORA_STACK", {"defaultInput": True}),
                },
        }
    
    RETURN_TYPES = ("STRING",)
    
    FUNCTION = "output_value"
    CATEGORY = "Sage Utils/Debug"
    
    def output_value(self, lora_stack):
        return (f"{lora_stack}",)
    
     
 
class Sage_LoraStack:
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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
    CATEGORY = "Sage Utils"
    
    def add_lora_to_stack(self, lora_name, model_weight, clip_weight, lora_stack = None):
        lora = (lora_name, model_weight, clip_weight)
        if lora_stack is None:
            stack = [lora]
            return(stack,)
        
        stack = []
        for the_name, m_weight, c_weight in lora_stack:
            stack.append((the_name, m_weight, c_weight))
        stack.append((lora_name, model_weight, clip_weight))

        return (stack,)

# Modified version of the main lora loader.
class Sage_LoraStackLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_stack": ("LORA_STACK", {"defaultInput": True})
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.", "The stack of loras.")
    FUNCTION = "load_loras"

    CATEGORY = "Sage Utils"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)
    
    def load_loras(self, model, clip, lora_stack):
        for lora in lora_stack:
            model, clip = self.load_lora(model, clip, lora[0], lora[1], lora[2]) if lora else None
        return (model, clip, lora_stack)
    
# An altered version of Save Image
class Sage_SaveImageWithMetadata:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI_Meta", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "include_node_metadata": ("BOOLEAN", {"default": True, "defaultInput": False}),
                "include_extra_pnginfo_metadata": ("BOOLEAN", {"default": True, "defaultInput": False})
            },
            "optional": {
                "param_metadata": ("STRING",{ "defaultInput": True}),
                "extra_metadata": ("STRING",{ "defaultInput": True})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Sage Utils"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory with added metadata. The param_metadata input should come from Construct Metadata, and the extra_metadata is anything you want."

    def set_metadata(self, include_node_metadata, include_extra_pnginfo_metadata, param_metadata = None, extra_metadata=None, prompt=None, extra_pnginfo=None):
        result = None
        if not cli_args.args.disable_metadata:
            result = PngInfo()
            if param_metadata is not None:
                result.add_text("parameters", param_metadata)
            if include_node_metadata == True:
                if prompt is not None:
                    result.add_text("prompt", json.dumps(prompt))
            if include_extra_pnginfo_metadata == True:
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        result.add_text(x, json.dumps(extra_pnginfo[x]))
            if extra_metadata is not None:
                result.add_text("Extra", extra_metadata)
        return result
        

    def save_images(self, images, filename_prefix, include_node_metadata, include_extra_pnginfo_metadata, param_metadata = None, extra_metadata=None, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            final_metadata = self.set_metadata(include_node_metadata, include_extra_pnginfo_metadata, param_metadata, extra_metadata, prompt, extra_pnginfo)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            
            img.save(os.path.join(full_output_folder, file), pnginfo=final_metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
