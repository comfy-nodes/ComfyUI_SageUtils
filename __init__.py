import ComfyUI_SageUtils.sage_cache
import ComfyUI_SageUtils.sage_utils
import ComfyUI_SageUtils.sage_styles

from .sage import *
from .sage_basic import *
from .sage_loaders import *
from .sage_info import *

sage_cache.load_cache()
sage_styles.load_styles()
WEB_DIRECTORY = "./js"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Sage_SetBool": Sage_SetBool,
    "Sage_SetInteger": Sage_SetInteger,
    "Sage_SetFloat": Sage_SetFloat,
    "Sage_SetText": Sage_SetText,
    "Sage_JoinText": Sage_JoinText,
    "Sage_TripleJoinText": Sage_TripleJoinText,
    "Sage_ViewText": Sage_ViewText,
    "Sage_ConditioningZeroOut": Sage_ConditioningZeroOut,
    "Sage_GetFileHash": Sage_GetFileHash,
    "Sage_GetInfoFromHash": Sage_GetInfoFromHash,
    "Sage_LastLoraInfo": Sage_LastLoraInfo,
    "Sage_ModelInfo": Sage_ModelInfo,
    "Sage_ModelInfoBreakout": Sage_ModelInfoBreakout,
    "Sage_GetPicturesFromHash": Sage_GetPicturesFromHash,
    "Sage_GetModelJSONFromHash": Sage_GetModelJSONFromHash,
    "Sage_CollectKeywordsFromLoraStack": Sage_CollectKeywordsFromLoraStack,
    "Sage_LoadImage": Sage_LoadImage,
    "Sage_EmptyLatentImagePassthrough": Sage_EmptyLatentImagePassthrough,
    "Sage_UNETLoader": Sage_UNETLoader,
    "Sage_CheckpointLoaderSimple": Sage_CheckpointLoaderSimple,
    "Sage_CheckpointLoaderRecent": Sage_CheckpointLoaderRecent,
    "Sage_LoraStack": Sage_LoraStack,
    "Sage_LoraStackRecent": Sage_LoraStackRecent,
    "Sage_LoraStackLoader": Sage_LoraStackLoader,
    "Sage_DualCLIPTextEncode": Sage_DualCLIPTextEncode,
    "Sage_SamplerInfo": Sage_SamplerInfo,
    "Sage_AdvSamplerInfo": Sage_AdvSamplerInfo,
    "Sage_KSampler": Sage_KSampler,
    "Sage_ConstructMetadata": Sage_ConstructMetadata,
    "Sage_ConstructMetadataLite": Sage_ConstructMetadataLite,
    "Sage_SaveImageWithMetadata": Sage_SaveImageWithMetadata,
    "Sage_PonyPrefix": Sage_PonyPrefix,
    "Sage_PopulateCache": Sage_PopulateCache,
    "Sage_CacheMaintenance": Sage_CacheMaintenance,
    "Sage_ModelReport": Sage_ModelReport,
    "Sage_ModelInfoFromModelId": Sage_ModelInfoFromModelId
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS  = {
    "Sage_SetBool": "Set Bool",
    "Sage_SetInteger": "Set Integer",
    "Sage_SetFloat": "Set Float",
    "Sage_SetText": "Set Text",
    "Sage_JoinText": "Join Text",
    "Sage_TripleJoinText": "Join Text x3",
    "Sage_ViewText": "View Text",
    "Sage_ConditioningZeroOut": "Zero Conditioning",
    "Sage_GetFileHash": "Get Sha256 Hash",
    "Sage_GetInfoFromHash": "Get Model Info From Hash",
    "Sage_LastLoraInfo": "Last Lora Info",
    "Sage_ModelInfo": "Model Info",
    "Sage_ModelInfoBreakout": "model_info -> Path and Hash",
    "Sage_GetPicturesFromHash": "Get Model Images From Hash",
    "Sage_GetModelJSONFromHash": "Get Model JSON From Hash",
    "Sage_CollectKeywordsFromLoraStack": "Lora Stack -> Keywords",
    "Sage_EmptyLatentImagePassthrough": "Empty Latent Passthrough",
    "Sage_LoadImage": "Load Image w/ Size & Metadata",
    "Sage_UNETLoader": "Load Diffusion Model w/ Metadata",
    "Sage_CheckpointLoaderSimple": "Load Checkpoint w/ Metadata",
    "Sage_CheckpointLoaderRecent": "Load Recently Used Checkpoint",
    "Sage_LoraStack": "Simple Lora Stack",
    "Sage_LoraStackRecent": "Recent Lora Stack",
    "Sage_LoraStackLoader": "Lora Stack Loader",
    "Sage_DualCLIPTextEncode": "Prompts to CLIP",
    "Sage_SamplerInfo": "Sampler Info",
    "Sage_AdvSamplerInfo": "Adv Sampler Info",
    "Sage_KSampler": "KSampler w/ Sampler Info",
    "Sage_ConstructMetadata": "Construct Metadata",
    "Sage_ConstructMetadataLite": "Construct Metadata Lite",
    "Sage_SaveImageWithMetadata": "Save Image w/ Added Metadata",
    "Sage_PonyPrefix": "Add Pony v6 Prefixes",
    "Sage_PopulateCache": "Scan for Metadata & Hash",
    "Sage_CacheMaintenance": "Cache Maintenance",
    "Sage_ModelReport": "Model Report",
    "Sage_ModelInfoFromModelId": "Get Model Info from Model Id"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY'] 