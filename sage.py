# A collection of utility nodes, many of which are dealing with metadata and collecting it.
import os
import json
import pathlib
import numpy as np
import torch

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import requests

# Pieces of ComfyUI that are being brought in for one reason or another.
import comfy
import folder_paths
import cli_args
import nodes

from .sage_utils import *
import ComfyUI_SageUtils.sage_cache as cache

class Sage_CollectKeywordsFromLoraStack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_stack": ("LORA_STACK", {"defaultInput": True})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("keywords",)

    FUNCTION = "get_keywords"

    CATEGORY = "Sage Utils/lora"
    DESCRIPTION = "Go through each model in the lora stack, grab any keywords from civitai, and combine them into one string. Place at the end of a lora_stack, or you won't get keywords for the entire stack."

    def get_keywords(self, lora_stack):
        lora_keywords = []
        if lora_stack is None:
            return ("",)
        
        for lora in lora_stack:
            try:
                hash = get_lora_hash(lora[0])

                json = get_civitai_json(hash)
                keywords = json["trainedWords"]
                if keywords != []:
                    lora_keywords.extend(keywords)
            except:
                print("Exception getting keywords!")
                continue

        ret = ", ".join(lora_keywords)
        ret = ' '.join(ret.split('\n'))
        return (ret,)

class Sage_GetInfoFromHash:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hash": ("STRING", {"defaultInput": True})
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("type", "base_model", "version_id", "model_id", "name", "version", "trained_words", "url")

    FUNCTION = "get_info"
    
    CATEGORY = "Sage Utils/util"
    DESCRIPTION = "Pull out various useful pieces of information from a hash, such as the model and version id, the model name and version, what model it's based on, and what keywords it has."

    def get_info(self, hash):
        ret = []
        path = ""

        try:
            json = get_civitai_json(hash)
            ret.append(json["model"]["type"])
            ret.append(json["baseModel"])
            ret.append(str(json["id"]))
            ret.append(str(json["modelId"]))
            ret.append(json["model"]["name"])
            ret.append(json["name"])
            words = json["trainedWords"]

            if words == []:
                ret.append("")
            else:
                ret.append(", ".join(words))
            ret.append(json["downloadUrl"])
        except:
            print("Exception when getting json data.")
            ret = ["", "", "", "", "", "", "", ""]
        
        return (ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], ret[6], ret[7],)

class Sage_GetPicturesFromHash:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hash": ("STRING", {"defaultInput": True}),
                "explicit": ("BOOLEAN", {"defaultInput": False})
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )

    FUNCTION = "get_pics"
    
    CATEGORY = "Sage Utils/util"
    DESCRIPTION = "Pull pics from civitai."

    def get_pics(self, hash, explicit):
        ret_urls = []
        #ret = []
        path = ""

        try:
            ret_urls = pull_lora_image_urls(hash, explicit)
        except:
            print("Exception when getting json data.")
            return([],)
        
        #for url in ret_urls:
        img = Image.open(requests.get(ret_urls[0], stream=True).raw)
        img = ImageOps.exif_transpose(img)
        img = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        ret = torch.from_numpy(img)[None,]

        return (ret,)

class Sage_IterOverFiles:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_dir": (list(folder_paths.folder_names_and_paths.keys()), {"defaultInput": False}),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("list",)
    
    FUNCTION = "get_files"
    
    CATEGORY = "Sage Utils/util"
    DESCRIPTION = "Calculates the hash of every model in the chosen directory and pulls civitai information. Takes forever. Returns the filenames."
    
    def get_files(self, base_dir):
        ret = pull_all_loras(folder_paths.folder_names_and_paths[base_dir])
        return (f"{ret}",)


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
    
    CATEGORY = "Sage Utils/util"
    DESCRIPTION = "Get an sha256 hash of a file. Can be used for detecting models, civitai calls and such."
    
    def get_hash(self, base_dir, filename):
        the_hash = ""
        try:
            file_path = folder_paths.get_full_path_or_raise(base_dir, filename)
            pull_metadata(file_path)
            the_hash = cache.cache_data[file_path]["hash"]
        except:
            print(f"Unable to hash file '{file_path}'. \n")
            the_hash = ""
        
        print(f"Hash for '{file_path}': {the_hash}")
        return (str(the_hash),)

class Sage_GetModelJSONFromHash:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hash": ("STRING", {"defaultInput": True})
            }
        }
        
    RETURN_TYPES = ("STRING",)
    
    FUNCTION = "pull_json"
    CATEGORY = "Sage Utils/util"
    DESCRIPTION = "Returns the JSON that civitai will give you, based on a hash. Useful if you want to see all the information, just what I'm using. This is the specific version hash."

    def pull_json(self, hash):
        the_json = {}
        try:
            the_json = get_civitai_json(hash)
        except:
            the_json = {}
        return(f"{json.dumps(the_json)}",)

class Sage_DualCLIPTextEncode:
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

    def get_conditioning(self, clip, text = None):
        zero_text = False
        if text == None:
            zero_text = True
            text = ""

        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")

        if zero_text == True:
            pooled_output = output.get("pooled_output", None)
            if pooled_output is not None:
                output["pooled_output"] = torch.zeros_like(pooled_output)
            return ([[torch.zeros_like(cond), output]])
            
        return ([[cond, output]])

    def encode(self, clip, pos = None, neg = None):
        ret_pos = ""
        ret_neg = ""

        if pos != None:
            ret_pos = pos
        
        if neg != None:
            ret_neg = neg

        return (self.get_conditioning(clip, pos), self.get_conditioning(clip, neg), ret_pos, ret_neg)

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
    OUTPUT_TOOLTIPS = ("To be piped to the Construct Metadata node and the KSampler with Metadata node.",)
    FUNCTION = "pass_info"

    CATEGORY = "Sage Utils/metadata"
    DESCRIPTION = "Grabs most of the sampler info. Should be routed both to the Construct Metadata node and the KSampler w/ Sampler Info node."

    def pass_info(self, seed, steps, cfg, sampler_name, scheduler):
        s_info = {}
        s_info["seed"] = seed
        s_info["steps"] = steps
        s_info["cfg"] = cfg
        s_info["sampler"] = sampler_name
        s_info["scheduler"] = scheduler
        return s_info,

class Sage_AdvSamplerInfo:
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

class Sage_KSampler:
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

    CATEGORY = "Sage Utils"
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
    
    CATEGORY = "Sage Utils/metadata"
    DESCRIPTION = "Puts together metadata in a A1111-like format. Uses the custom sampler info node. The return value is a string, so can be manipulated by other nodes."
    
    def construct_metadata(self, model_info, positive_string, negative_string, width, height, sampler_info, lora_stack = None):
        metadata = ''

        lora_hashes = []
        lora_hash_string = ''

        resource_hashes = []
        civitai_string = ''
        
        sampler_name = civitai_sampler_name(sampler_info['sampler'], sampler_info['scheduler'])
        
        if lora_stack is not None:
            # We're going through generating A1111 style <lora> tags to insert in the prompt, adding the lora hashes to the resource hashes in exactly the format
            # that CivitAI's approved extension for A1111 does, and inserting the Lora hashes at the end in the way they appeared looking at the embedded metadata
            # generated by Forge.
            for lora in lora_stack:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora[0])
                lora_name = str(pathlib.Path(lora_path).name)
                pull_metadata(lora_path)
                lora_data = get_model_info(lora_path, lora[1])
                if lora_data != {}:
                    resource_hashes.append(lora_data)
                
                lora_hash = cache.cache_data[lora_path]["hash"]
                lora_hashes += [f"{lora_name}: {lora_hash}"]
        
        lora_hash_string = "Lora hashes: " + ",".join(lora_hashes)
        civitai_string = f"Civitai resources: {json.dumps(resource_hashes)}"

        metadata = f"{positive_string} {lora_to_prompt(lora_stack)}" + "\n" 
        if negative_string != "":
            metadata += f"Negative prompt: {negative_string}" + "\n"
        metadata += f"Steps: {sampler_info['steps']}, Sampler: {sampler_name}, Scheduler type: {sampler_info['scheduler']}, CFG scale: {sampler_info['cfg']}, Seed: {sampler_info['seed']}, Size: {width}x{height},"
        metadata += f"Model: {model_info['name']}, Model hash: {model_info['hash']}, Version: v1.10-RC-6-comfyui, {civitai_string}, {lora_hash_string}"
        return metadata,


class Sage_ConstructMetadataLite:
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
        metadata += f"Steps: {sampler_info['steps']}, Sampler: {sampler_name}, Scheduler type: {sampler_info['scheduler']}, CFG scale: {sampler_info['cfg']}, Seed: {sampler_info['seed']}, Size: {width}x{height},"
        metadata += f"Version: v1.10-RC-6-comfyui, Civitai resources: {json.dumps(resource_hashes)}"
        return metadata,

class Sage_LoraStack:
    def __init__(self):
        pass
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"defaultInput": False, "default": True}),
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
    CATEGORY = "Sage Utils/lora"
    DESCRIPTION = "Choose a lora with weights, and add it to a lora_stack. Compatable with other node packs that have lora_stacks."
    
    def add_lora_to_stack(self, enabled, lora_name, model_weight, clip_weight, lora_stack = None):
        if enabled == True:
            stack = add_lora_to_stack(lora_name, model_weight, clip_weight, lora_stack)
        else:
            stack = lora_stack

        return (stack,)

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
    DESCRIPTION = "Saves the input images to your ComfyUI output directory with added metadata. The param_metadata input should come from Construct Metadata, and the extra_metadata is anything you want. Both are just strings, though, with the difference being that the first has a keyword of parameters, and the second, extra, so technically you could pass in your own metadata, or even type it in in a Set Text node and hook that to this node."

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
