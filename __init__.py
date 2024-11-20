from .sage import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Sage_LoraStackDebugString": Sage_LoraStackDebugString,
    "Sage_LoraStack": Sage_LoraStack,
    "Sage_LoraStackLoader": Sage_LoraStackLoader,
    "Sage_DualCLIPTextEncode": Sage_DualCLIPTextEncode,
    "Sage_SamplerInfo": Sage_SamplerInfo,
    "Sage_KSampler": Sage_KSampler,
    "Sage_ConstructMetadata": Sage_ConstructMetadata,
    "Sage_CheckpointLoaderSimple": Sage_CheckpointLoaderSimple,
    "Sage_SaveImageWithMetadata": Sage_SaveImageWithMetadata
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS  = {
    "Sage_LoraStackDebugString": "Lora Stack Debug String",
    "Sage_LoraStackLoader": "Lora Stack Loader",
    "Sage_LoraStack":   "Simple Lora Stack",
    "Sage_DualCLIPTextEncode": "Prompts to CLIP",
    "Sage_SamplerInfo": "Sampler Info",
    "Sage_KSampler": "KSampler w/ Sampler Info",
    "Sage_ConstructMetadata": "Construct Metadata",
    "Sage_CheckpointLoaderSimple": "Load Checkpoint With Name",
    "Sage_SaveImageWithMetadata": "Save Image with Added Metadata"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 