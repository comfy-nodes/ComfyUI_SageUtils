# A collection of utility nodes, many of which are dealing with metadata and collecting it.

import os
import json
import pathlib
import hashlib
import numpy as np

from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Pieces of ComfyUI that are being brought in for one reason or another.
import comfy
import folder_paths
import cli_args
import nodes
        
def lora_to_string(lora_name, model_weight, clip_weight):
    lora_string = ' <lora:' + str(pathlib.Path(lora_name).name) + ":" + str(model_weight) + ":"  + str(clip_weight) +  ">"
        
    return lora_string

def lora_stack_to_string(stack):
    lora_string = ''
        
    for lora in stack:
        lora_string += lora_to_string(lora[0], lora[1], lora[2])
        
    return lora_string

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
        if lora_stack is None:
            lora_info = ""
        else: 
            lora_info = lora_stack_to_string(lora_stack)
        
        resource_hashes['model'] = model_info['hash']
        
        metadata = f"{positive_string} {lora_info}" + "\n" 
        metadata += f"Negative prompt: {negative_string}" + "\n"
        metadata += f"Steps: {sampler_info['steps']}, Sampler: {sampler_info['sampler']}, Scheduler type: {sampler_info['scheduler']}, CFG scale: {sampler_info['cfg']}, Seed: {sampler_info['seed']}, Size: {width}x{height},"
        metadata += f"Model: {model_info['name']},Model hash: {model_info['hash']},  Version: v1.10-RC-6-actually-totally-comfyui, Hashes: {json.dumps(resource_hashes)}"
        print(metadata)
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
        
        m = hashlib.sha256()
        with open(model_info["path"], 'rb') as f:
            m.update(f.read())
        model_info["hash"] = str(m.digest().hex()[:10])
    
        out = comfy.sd.load_checkpoint_guess_config(model_info["path"], output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        result = (*out[:3], model_info)
        return (result)

class Sage_MultiModelNameChooser:
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
            }
        }
 
    RETURN_TYPES  = ('COMBO', 'STRING',)
    RETURN_NAMES  = ('ckpt_name', 'string')
 
    FUNCTION  =  "pick_name"
    OUTPUT_NODE   = False
    CATEGORY  =  "Sage Utils"
    DESCRIPTION = "Returns both a combo box with the name, and the name as a string."
 
    def test(self, ckpt_name):
        result = str(ckpt_name)
        return (ckpt_name, result,)

class Sage_ExamineNode:
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",)
                },
            }
 
    RETURN_TYPES  = ('STRING',)
 
    FUNCTION  =  "test"
    OUTPUT_NODE   = False
    CATEGORY  =  "Sage Utils"
 
    def test(self, model):
        results  = []
        result = vars(model.model)
        # model.model.__class__.__name__ is model type.
        return (str(result),)
 
 
class Sage_LoraNameToString:
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": ("STRING",{ "defaultInput": True}),
                "model_weight": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "defaultInput": True, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "clip_weight": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "defaultInput": True, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                },
             }
 
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ("Lora String",)
    
    FUNCTION = "lora_to_string"
    CATEGORY = "Sage Utils"
    
    def lora_to_string(self, lora_name, model_weight, clip_weight):
        return (lora_to_string(lora_name, model_weight, clip_weight),)
    
class Sage_LoraStackToString:
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stack": ("LORA_STACK",)
                },
            }
        
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ("Lora String",)
    
    FUNCTION = "lora_stack_to_string"
    CATEGORY = "Sage Utils"
    
    def lora_stack_to_string(self, stack):
        return (lora_stack_to_string(stack),)

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
                "filename_prefix": ("STRING", {"default": "ComfyUI_Meta", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
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

    def set_metadata(self, param_metadata = None, extra_metadata=None, prompt=None, extra_pnginfo=None):
        result = None
        if not cli_args.args.disable_metadata:
            result = PngInfo()
            if param_metadata is not None:
                result.add_text("parameters", param_metadata)
            if prompt is not None:
                result.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                        result.add_text(x, json.dumps(extra_pnginfo[x]))
            if extra_metadata is not None:
                result.add_text("Extra", extra_metadata)
        return result
        

    def save_images(self, images, filename_prefix="ComfyUI", param_metadata = None, extra_metadata=None, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            final_metadata = self.set_metadata(param_metadata, extra_metadata, prompt, extra_pnginfo)

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

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Sage_ExamineNode": Sage_ExamineNode,
    #"Sage_MultiModelNameChooser": Sage_MultiModelNameChooser,
    "Sage_LoraStackToString": Sage_LoraStackToString,
    "Sage_LoraNameToString": Sage_LoraNameToString,
    "Sage_SamplerInfo": Sage_SamplerInfo,
    "Sage_KSampler": Sage_KSampler,
    "Sage_ConstructMetadata": Sage_ConstructMetadata,
    "Sage_CheckpointLoaderSimple": Sage_CheckpointLoaderSimple,
    "Sage_SaveImageWithMetadata": Sage_SaveImageWithMetadata
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS  = {
    "Sage_ExamineNode":  "Examine Node",
    #"Sage_MultiModelNameChooser": "Multi Model Name Chooser",
    "Sage_LoraStackToString":   "Lora Stack to String",
    "Sage_LoraNameToString":  "Lora Name to String",
    "Sage_SamplerInfo": "Sampler Info",
    "Sage_KSampler": "KSampler w/ Sampler Info",
    "Sage_ConstructMetadata": "Construct Metadata",
    "Sage_CheckpointLoaderSimple": "Load Checkpoint With Name",
    "Sage_SaveImageWithMetadata": "Save Image with Added Metadata"
}