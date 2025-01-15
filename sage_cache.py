import json
import pathlib

cache_path = pathlib.Path.cwd()
cache_data = {}

def init_cache(base_path):
    global cache_path
    cache_path = pathlib.Path(base_path) / "sage_cache.json"
    cache_data = {}
    load_cache()
    
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
