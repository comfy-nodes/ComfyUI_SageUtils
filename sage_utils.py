#Utility functions for use in the nodes.

import pathlib
import hashlib
import requests

def get_civitai_json(hash):
    r = requests.get("https://civitai.com/api/v1/model-versions/by-hash/" + hash)
    return r.json()

def lora_to_string(lora_name, model_weight, clip_weight):
    lora_string = ' <lora:' + str(pathlib.Path(lora_name).name) + ":" + str(model_weight) + ":"  + str(clip_weight) +  ">"
        
    return lora_string

def lora_stack_to_string(stack):
    lora_string = ''
        
    for lora in stack:
        lora_string += lora_to_string(lora[0], lora[1], lora[2])
        
    return lora_string

def get_file_sha256(path):
    m = hashlib.sha256()
    with open(path, 'rb') as f:
        m.update(f.read())
    result = str(m.digest().hex()[:10])
    return result

def civitai_sampler_name(sampler_name, scheduler_name):
    comfy_to_auto = {
        'ddim': 'DDIM',
        'dpm_2': 'DPM2',
        'dpm_2_ancestral': 'DPM2 a',
        'dpmpp_2s_ancestral': 'DPM++ 2S a',
        'dpmpp_2m': 'DPM++ 2M',
        'dpmpp_sde': 'DPM++ SDE',
        'dpmpp_2m_sde': 'DPM++ 2M SDE',
        'dpmpp_3m_sde': 'DPM++ 3M SDE',
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
    result = comfy_to_auto[sampler_name]
    
    if (scheduler_name == "karras"):
        result += " Karras"
    elif (scheduler_name == "exponential"):
        result += " Exponential"
    
    return result
