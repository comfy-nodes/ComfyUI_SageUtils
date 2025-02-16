import os
import json
import pathlib

from ..sage import base_path

class SageCache:
    def __init__(self, path):
        self.path = pathlib.Path(path) / "sage_cache.json"
        self.data = {}

    def load(self):
        try:
            if self.path.is_file():
                with self.path.open("r") as read_file:
                    self.data = json.load(read_file)
        except Exception as e:
            print(f"Unable to load cache: {e}")

    def save(self):
        try:
            if self.data:
                with self.path.open("w") as output_file:
                    json.dump(self.data, output_file, separators=(",", ":"), sort_keys=True, indent=4)
            else:
                print("Skipping saving cache, as the cache is empty.")
        except Exception as e:
            print(f"Unable to save cache: {e}")

cache = SageCache(base_path)
