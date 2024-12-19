import json
import pathlib
import folder_paths

sage_styles = {}
style_path = pathlib.Path(folder_paths.base_path) / "custom_nodes" / "ComfyUI_SageUtils" / "sage_styles.json"
style_user_path = pathlib.Path(folder_paths.base_path) / "custom_nodes" / "ComfyUI_SageUtils" / "sage_styles_user.json"


def load_styles():
    global sage_styles
    sage_styles = []

    for path in [style_path, style_user_path]:
        if path.is_file():
            try:
                with path.open(mode="r") as read_file:
                    sage_styles.append(json.load(read_file))
            except Exception as e:
                print(f"Unable to load styles from {path}: {e}")

