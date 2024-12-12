# For nodes that involve pulling or working with model info.

import json

from .sage_utils import *
import ComfyUI_SageUtils.sage_cache as cache
import folder_paths

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

        try:
            ret_urls = pull_lora_image_urls(hash, explicit)
        except:
            print("Exception when getting json data.")
            return([],)
        
        ret = url_to_torch_image(ret_urls[0])

        return (ret,)
    

class Sage_PopulateCache:
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


class Sage_ModelInfoBreakout:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_info": ("MODEL_INFO", {"defaultInput": True})
            }
        }
        
    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("path", "hash")
    
    FUNCTION = "model_breakout"
    CATEGORY = "Sage Utils/util"
    DESCRIPTION = "Breaks down the model info output into the path and hash."

    def model_breakout(self, model_info):
        return(model_info['path'], model_info['hash'])
    
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
    CATEGORY = "Sage Utils/util"
    DESCRIPTION = "Lets you remove entries for models that are no longer there. dup_hash returns a list of files with the same hash, and dup_model returns ones with the same civitai model id (but not neccessarily the same version)."

    def cache_maintenance(self, remove_ghost_entries):
        ghost_entries = []
        cache_by_hash = {}
        cache_by_id = {}
        dup_hash = {}
        dup_id = {}

        for model_path in cache.cache_data.keys():
            if pathlib.Path(model_path).is_file() == False:
                ghost_entries.append(model_path)
            else:
                if 'hash' in cache.cache_data[model_path]:
                    cur_hash = cache.cache_data[model_path]['hash']
                    if cur_hash not in cache_by_hash:
                        cache_by_hash[cur_hash] = []
                    cache_by_hash[cur_hash].append(model_path)
                if 'modelId' in cache.cache_data[model_path]:
                    cur_id = cache.cache_data[model_path]['modelId']
                    if cur_id not in cache_by_id:
                        cache_by_id[cur_id] = []
                    cache_by_id[cur_id].append(model_path)

        if remove_ghost_entries == True:
            for ghost in ghost_entries:
                cache.cache_data.pop(ghost)
            cache.save_cache()

        for cur_hash in cache_by_hash:
            if len(cache_by_hash[cur_hash]) > 1:
                dup_hash[cur_hash] = cache_by_hash[cur_hash]
        
        for cur_id in cache_by_id:
            if len(cache_by_id[cur_id]) > 1:
                dup_id[cur_id] = cache_by_id[cur_id]

        dup_hashes = json.dumps(dup_hash, separators=(",", ":"), sort_keys=True, indent=4)
        dup_ids = json.dumps(dup_id, separators=(",", ":"), sort_keys=True, indent=4)
        return(", ".join(ghost_entries), dup_hashes, dup_ids)


class Sage_ModelReport:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (("LORA", "Checkpoint"), {"defaultInput": True})
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_list",)
    
    FUNCTION = "pull_list"
    CATEGORY = "Sage Utils/util"
    DESCRIPTION = "Returns a list of models in the cache of the specified type, by basel model type."
    
    def pull_list(self, type):
        sorted_models = {}
        
        for model_path in cache.cache_data.keys():
            if 'model' in cache.cache_data[model_path]:
                if 'type' in cache.cache_data[model_path]['model']:
                    if cache.cache_data[model_path]['model']['type'] == type:
                        if 'baseModel' in cache.cache_data[model_path]:
                            baseModel = cache.cache_data[model_path]['baseModel']
                            if baseModel not in sorted_models:
                                sorted_models[baseModel] = []
                            
                            sorted_models[baseModel].append(model_path)
                        
        ret = json.dumps(sorted_models, separators=(",", ":"), sort_keys=True, indent=4)
        
        return (ret,)