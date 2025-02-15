import os
import json
import pathlib

cache_realpath = os.path.dirname(os.path.realpath(__file__))

sage_styles = {}
style_path = pathlib.Path(cache_realpath) / "sage_styles.json"
style_user_path = pathlib.Path(cache_realpath) / "sage_styles_user.json"

def load_styles():
    global sage_styles
    global style_path, style_user_path
    sage_styles = []

    for path in [style_path, style_user_path]:
        if path.is_file():
            try:
                with path.open(mode="r") as read_file:
                    sage_styles.append(json.load(read_file))
            except Exception as e:
                print(f"Unable to load styles from {path}: {e}")

