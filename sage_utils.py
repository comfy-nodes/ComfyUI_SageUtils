#Utility functions for use in the nodes.

import pathlib
import hashlib
import requests
import folder_paths
import json

def get_civitai_json(hash):
    r = requests.get("https://civitai.com/api/v1/model-versions/by-hash/" + hash)
    return r.json()

def lora_to_string(lora_name, model_weight, clip_weight):
    lora_string = ' <lora:' + str(pathlib.Path(lora_name).name) + ":" + str(model_weight) +  ">" #  + ":" + str(clip_weight)
        
    return lora_string

def lora_stack_to_string(stack):
    lora_string = ''
        
    for lora in stack:
        lora_string += lora_to_string(lora[0], lora[1], lora[2])
        
    return lora_string

def get_file_sha256(path):
    cache_data = {}
    cache_path = pathlib.Path.cwd() / "custom_nodes" / "ComfyUI_SageUtils" / "sage_cache.json"
    if cache_path.is_file():
        with cache_path.open("r") as read_file:
            cache_data = json.load(read_file)
    
    if path in cache_data:
        if 'hash' in cache_data[path]:
            return cache_data[path]["hash"]
        
    m = hashlib.sha256()
    with open(path, 'rb') as f:
        m.update(f.read())
    result = str(m.digest().hex()[:10])

    cache_data[path] = {}
    cache_data[path]["hash"] = result

    with cache_path.open("w") as output_file:
        output_file.write(json.dumps(cache_data, separators=(",", ":"), sort_keys=True, indent=4))

    return result

def get_lora_hash(lora_name):
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    hash = get_file_sha256(lora_path)

    return hash

def civitai_sampler_name(sampler_name, scheduler_name):
    comfy_to_auto = {
        'ddim': 'DDIM',
        'dpm_2': 'DPM2',
        'dpm_2_ancestral': 'DPM2 a',
        'dpmpp_2s_ancestral': 'DPM++ 2S a',
        'dpmpp_2m': 'DPM++ 2M',
        'dpmpp_sde': 'DPM++ SDE',
        'dpmpp_2m_sde': 'DPM++ 2M SDE',
        'dpmpp_2m_sde_gpu': 'DPM++ 2M SDE',
        'dpmpp_3m_sde': 'DPM++ 3M SDE',
        'dpmpp_3m_sde_gpu': 'DPM++ 3M SDE',
        'dpm_fast': 'DPM fast',
        'dpm_adaptive': 'DPM adaptive',
        'euler_ancestral': 'Euler a',
        'euler': 'Euler',
        'heun': 'Heun',
        'lcm': 'LCM',
        'lms': 'LMS',
        'plms': 'PLMS',
        'uni_pc': 'UniPC',
        'uni_pc_bh2': 'UniPC'
    }
    result = ""
    try:
        result = comfy_to_auto[sampler_name]
    except:
        result = f"{sampler_name}"
    
    if (scheduler_name == "karras"):
        result += " Karras"
    elif (scheduler_name == "exponential"):
        result += " Exponential"
    
    return result
