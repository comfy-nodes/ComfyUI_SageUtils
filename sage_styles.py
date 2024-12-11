import json
import pathlib
import folder_paths

sage_styles = {}
style_path = pathlib.Path(folder_paths.base_path) / "custom_nodes" / "ComfyUI_SageUtils" / "sage_styles.json"
style_user_path = pathlib.Path(folder_paths.base_path) / "custom_nodes" / "ComfyUI_SageUtils" / "sage_styles_user.json"


def load_styles():
    global sage_styles
    global style_path
    
    try:
        sage_styles = []
        if style_path.is_file():
            with style_path.open(mode = "r") as read_file:
                temp_styles = json.load(read_file)
                sage_styles.append(temp_styles)
        if style_user_path.is_file():
            with style_user_path.open(mode = "r") as read_file:
                temp_styles = json.load(read_file)
                sage_styles.append(temp_styles)
        #print(f"{json.dumps(sage_styles, separators=(",", ":"), sort_keys=True, indent=4)}")
    except:
        print("Unable to load styles.")

