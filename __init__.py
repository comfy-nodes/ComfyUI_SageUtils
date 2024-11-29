import ComfyUI_SageUtils.sage_cache
import ComfyUI_SageUtils.sage_utils

from .sage import *
from .sage_basic import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Sage_SetInteger": Sage_SetInteger,
    "Sage_SetFloat": Sage_SetFloat,
    "Sage_SetText": Sage_SetText,
    "Sage_JoinText": Sage_JoinText,
    #"Sage_ViewText": Sage_ViewText,
    "Sage_ConditioningZeroOut": Sage_ConditioningZeroOut,
    "Sage_GetFileHash": Sage_GetFileHash,
    "Sage_GetInfoFromHash": Sage_GetInfoFromHash,
    "Sage_GetModelJSONFromHash": Sage_GetModelJSONFromHash,
    "Sage_LoraStackDebugString": Sage_LoraStackDebugString,
    "Sage_CollectKeywordsFromLoraStack": Sage_CollectKeywordsFromLoraStack,
    
    "Sage_CheckpointLoaderSimple": Sage_CheckpointLoaderSimple,
    "Sage_LoraStack": Sage_LoraStack,
    "Sage_LoraStackLoader": Sage_LoraStackLoader,
    "Sage_DualCLIPTextEncode": Sage_DualCLIPTextEncode,
    "Sage_SamplerInfo": Sage_SamplerInfo,
    "Sage_KSampler": Sage_KSampler,
    "Sage_ConstructMetadata": Sage_ConstructMetadata,
    "Sage_SaveImageWithMetadata": Sage_SaveImageWithMetadata,
    "Sage_PonyPrefix": Sage_PonyPrefix,
    "Sage_IterOverFiles": Sage_IterOverFiles
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS  = {
    "Sage_SetInteger": "Set Integer",
    "Sage_SetFloat": "Set Float",
    "Sage_SetText": "Set Text",
    "Sage_JoinText": "Join Text",
    #"Sage_ViewText": "View Text",
    "Sage_ConditioningZeroOut": "Zero Conditioning",
    "Sage_GetFileHash": "Get Sha256 Hash",
    "Sage_GetInfoFromHash": "Get Model Info From Hash",
    "Sage_GetModelJSONFromHash": "Get Model JSON From Hash",
    "Sage_LoraStackDebugString": "Lora Stack Debug String",
    "Sage_CollectKeywordsFromLoraStack": "Lora Stack -> Keywords" ,   
    "Sage_CheckpointLoaderSimple": "Load Checkpoint With Name",
    "Sage_LoraStack":   "Simple Lora Stack",
    "Sage_LoraStackLoader": "Lora Stack Loader",
    "Sage_DualCLIPTextEncode": "Prompts to CLIP",
    "Sage_SamplerInfo": "Sampler Info",
    "Sage_KSampler": "KSampler w/ Sampler Info",
    "Sage_ConstructMetadata": "Construct Metadata",
    "Sage_SaveImageWithMetadata": "Save Image with Added Metadata",
    "Sage_PonyPrefix": "Add Pony v6 Prefixes",
    "Sage_IterOverFiles": "Scan for Metadata and Calculate Hashes"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 