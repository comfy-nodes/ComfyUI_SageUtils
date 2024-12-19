import json
import pathlib
import folder_paths

cache_path = pathlib.Path(folder_paths.base_path) / "custom_nodes" / "ComfyUI_SageUtils" / "sage_cache.json"
cache_data = {}

def load_cache():
    global cache_data
    try:
        if cache_path.is_file():
            with cache_path.open("r") as read_file:
                cache_data = json.load(read_file)
    except Exception as e:
        print(f"Unable to load cache: {e}")

def save_cache():
    try:
        if cache_data:
            with cache_path.open("w") as output_file:
                json.dump(cache_data, output_file, separators=(",", ":"), sort_keys=True, indent=4)
        else:
            print("Skipping saving cache, as the cache is empty.")
    except Exception as e:
        print(f"Unable to save cache: {e}")
