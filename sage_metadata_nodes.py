import comfy
import folder_paths
import nodes
from comfy.cli_args import args
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfyui_version import __version__

from .sage import *

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import os
import re
from datetime import datetime

class Sage_SamplerInfo(ComfyNodeABC):
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
    OUTPUT_TOOLTIPS = ("To be piped to the Construct Metadata node and the KSampler with Metadata node.",)
    FUNCTION = "pass_info"

    CATEGORY = "Sage Utils/metadata"
    DESCRIPTION = "Grabs most of the sampler info. Should be routed both to the Construct Metadata node and the KSampler w/ Sampler Info node."

    def pass_info(self, seed, steps, cfg, sampler_name, scheduler):
        return {"seed": seed, "steps": steps, "cfg": cfg, "sampler": sampler_name, "scheduler": scheduler},

class Sage_AdvSamplerInfo(ComfyNodeABC):
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "add_noise": ("BOOLEAN", {"default": True}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("ADV_SAMPLER_INFO",)
    OUTPUT_TOOLTIPS = ("To be piped to the KSampler.",)
    FUNCTION = "pass_adv_info"

    CATEGORY = "Sage Utils/metadata"
    DESCRIPTION = "Adds more optional values to the KSampler."

    def pass_adv_info(self, add_noise, start_at_step, end_at_step, return_with_leftover_noise):
        s_info = {}
        s_info["add_noise"] = add_noise
        s_info["start_at_step"] = start_at_step
        s_info["end_at_step"] = end_at_step
        s_info["return_with_leftover_noise"] = return_with_leftover_noise
        return s_info,

class Sage_KSampler(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "sampler_info": ('SAMPLER_INFO', { "defaultInput": True, "tooltip": "Adds in most of the KSampler options. Should be piped both here and to the Construct Metadata node."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."})
            },
            "optional": {
                "advanced_info": ('ADV_SAMPLER_INFO', {"defaultInput": True, "tooltip": "Optional. Adds in the options an advanced KSampler would have."})
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "Sage Utils/sampler"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image. Designed to work with the Sampler info node."

    def sample(self, model, sampler_info, positive, negative, latent_image, denoise=1.0, advanced_info = None):
        if advanced_info is None:
            return nodes.common_ksampler(model, sampler_info["seed"], sampler_info["steps"], sampler_info["cfg"], sampler_info["sampler"], sampler_info["scheduler"], positive, negative, latent_image, denoise=denoise)
        
        force_full_denoise = True
        if advanced_info["return_with_leftover_noise"] == True:
            force_full_denoise = False

        disable_noise = False
        if advanced_info["add_noise"] == False:
            disable_noise = True
        return nodes.common_ksampler(model, sampler_info["seed"], sampler_info["steps"], sampler_info["cfg"], sampler_info["sampler"],  sampler_info["scheduler"], positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=advanced_info['start_at_step'], last_step=advanced_info['end_at_step'], force_full_denoise=force_full_denoise)

class Sage_ConstructMetadata(ComfyNodeABC):
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model_info": ('MODEL_INFO',{ "defaultInput": True}),
                "sampler_info": ('SAMPLER_INFO', { "defaultInput": True}),
                "width": ('INT', { "defaultInput": True}),
                "height": ('INT', { "defaultInput": True}),
                "positive_string": ('STRING', {"multiline": True, "defaultInput": True}),
            },
            "optional": {
                "negative_string": ('STRING', {"multiline": True, "defaultInput": True}),
                "clip_skip": ('INT', {"default": 1, "min": 1, "max": 24, "step": 1, "defaultInput": True}),
                "lora_stack": ('LORA_STACK', {"defaultInput": True}),
                "civitai_format": ('BOOLEAN', {"defaultInput": False, "default": False}),
            },
        }

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('param_metadata',)
    FUNCTION = "construct_metadata"
    
    CATEGORY = "Sage Utils/metadata"
    DESCRIPTION = "Puts together metadata in a A1111-like format. Uses the custom sampler info node. The return value is a string, so can be manipulated by other nodes."
    
    def construct_metadata(self, model_info, sampler_info, width, height, positive_string, negative_string="", civitai_format=False, clip_skip=1, lora_stack=None):
        metadata = ''

        lora_hashes = []
        lora_hash_string = ''

        resource_hashes = []
        civitai_resources = ''
        
        sampler_name = civitai_sampler_name(sampler_info['sampler'], sampler_info['scheduler'])
        
        if lora_stack is not None:
            # We're going through generating A1111 style <lora> tags to insert in the prompt, adding the lora hashes to the resource hashes in exactly the format
            # that CivitAI's approved extension for A1111 does, and inserting the Lora hashes at the end in the way they appeared looking at the embedded metadata
            # generated by Forge.
            for lora in lora_stack:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora[0])
                lora_name = os.path.splitext(os.path.basename(lora_path))[0]
                pull_metadata(lora_path)
                lora_data = get_model_info(lora_path, lora[1])
                if lora_data != {}:
                    resource_hashes.append(lora_data)
                
                lora_hash = cache.cache.data[lora_path]["hash"]
                lora_hashes += [f"{lora_name}: {lora_hash}"]
        
        if civitai_format:
            if len(resource_hashes) >=1:
                civitai_resources = f", Civitai resources: {json.dumps(resource_hashes)}"
            metadata = f"{positive_string}\n"
        else:
            if len(lora_hashes) >= 1:
                lora_hash_string = ', Lora hashes: "' + ", ".join(lora_hashes) + '"'
            metadata = f"{positive_string}{lora_to_prompt(lora_stack)}\n"

        if negative_string != "":
            metadata += f"Negative prompt: {negative_string}\n"

        clip_skip = "" if clip_skip <= 1 else f"Clip skip: {clip_skip}, "

        metadata += f"Steps: {sampler_info['steps']}, Sampler: {sampler_name}, Scheduler type: {sampler_info['scheduler']}, CFG scale: {sampler_info['cfg']}, Seed: {sampler_info['seed']}, Size: {width}x{height}, {clip_skip}"
        metadata += f"Model: {name_from_path(model_info['path'])}, Model hash: {model_info['hash']}, Version: ComfyUI {__version__}" + civitai_resources + lora_hash_string
        return metadata,


class Sage_ConstructMetadataLite(ComfyNodeABC):
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
    
    CATEGORY = "Sage Utils/metadata"
    DESCRIPTION = "Puts together metadata in a A1111-like format. Uses the custom sampler info node. The return value is a string, so can be manipulated by other nodes."
    
    def construct_metadata(self, model_info, positive_string, negative_string, width, height, sampler_info, lora_stack = None):
        metadata = ''
        
        resource_hashes = []
        
        sampler_name = civitai_sampler_name(sampler_info['sampler'], sampler_info['scheduler'])
        resource_hashes.append(get_model_info(model_info['path']))

        
        if lora_stack is not None:
            # We're going through generating A1111 style prompt information, but not doing the loras and model A1111 style, rather
            # just adding the lora and model information in the resource section.
            for lora in lora_stack:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora[0])
                pull_metadata(lora_path)
                lora_data = get_model_info(lora_path, lora[1])
                if lora_data != {}:
                    resource_hashes.append(lora_data)

        metadata = f"{positive_string}" + "\n" 
        if negative_string != "": metadata += f"Negative prompt: {negative_string}" + "\n"
        metadata += f"Steps: {sampler_info['steps']}, Sampler: {sampler_name}, Scheduler type: {sampler_info['scheduler']}, CFG scale: {sampler_info['cfg']}, Seed: {sampler_info['seed']}, Size: {width}x{height}, "
        metadata += f"Version: ComfyUI {__version__}, Civitai resources: {json.dumps(resource_hashes)}"
        return metadata,

# An altered version of Save Image
class Sage_SaveImageWithMetadata(ComfyNodeABC):
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
            },
            "optional": {
                "param_metadata": ("STRING", {"defaultInput": True}),
                "extra_param_metadata": ("STRING", {}),
                "include_workflow_metadata": ("BOOLEAN", {"default": True}),
                "save_workflow_json": ("BOOLEAN", {"default": False})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "Sage Utils/image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory with added metadata. The param_metadata input should come from Construct Metadata, and the extra_param_metadata is anything you want. Both are just strings, though, with the difference being that the first has a keyword of parameters, and the second, extra, so technically you could pass in your own metadata, or even type it in in a Set Text node and hook that to this node."

    pattern_format = re.compile(r"(%[^%]+%)")

    def set_metadata(self, include_workflow_metadata, param_metadata=None, extra_param_metadata=None, prompt=None, extra_pnginfo=None):
        result = None
        if not args.disable_metadata:
            result = PngInfo()
            if param_metadata is not None:
                if extra_param_metadata is not None:
                    param_metadata += (f", {extra_param_metadata}")
                result.add_text("parameters", param_metadata)
            if include_workflow_metadata == True:
                if prompt is not None:
                    result.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        result.add_text(x, json.dumps(extra_pnginfo[x]))
        return result
        

    def save_images(self, images, filename_prefix, param_metadata = None, extra_param_metadata=None, include_workflow_metadata=True, save_workflow_json=False, prompt=None, extra_pnginfo=None):
        filename_prefix = self.format_filename(filename_prefix) + self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            final_metadata = self.set_metadata(include_workflow_metadata, param_metadata, extra_param_metadata, prompt, extra_pnginfo)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            filename_base = f"{filename_with_batch_num}_{counter:05}_"
            file = f"{filename_base}.png"
            
            img.save(os.path.join(full_output_folder, file), pnginfo=final_metadata, compress_level=self.compress_level)

            if save_workflow_json and extra_pnginfo is not None:
                file_path_workflow = os.path.join(
                    full_output_folder, f"{filename_base}.json"
                )
                with open(file_path_workflow, "w", encoding="utf-8") as f:
                    json.dump(extra_pnginfo["workflow"], f)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }


    @classmethod
    def format_filename(cls, filename):
        result = re.findall(cls.pattern_format, filename)
        
        now = datetime.now()
        date_table = {
            "yyyy": str(now.year),
            "yy": str(now.year)[-2:],
            "MM": str(now.month).zfill(2),
            "dd": str(now.day).zfill(2),
            "hh": str(now.hour).zfill(2),
            "mm": str(now.minute).zfill(2),
            "ss": str(now.second).zfill(2),
        }

        for segment in result:
            parts = segment.replace("%", "").split(":")
            key = parts[0]

            if key == "date":
                date_format = parts[1] if len(parts) >= 2 else "yyyyMMddhhmmss"
                for k, v in date_table.items():
                    date_format = date_format.replace(k, v)
                filename = filename.replace(segment, date_format)

        return filename
