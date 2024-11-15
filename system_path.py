import platform
import os
system = platform.system()
model_path_pre = ""
if system == 'Linux':
    model_path_pre = "~/.cahce/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14-336px/deploy/"
elif system == 'Windows':
    model_path_pre = os.environ.get("USERPROFILE") + "/.cache/huggingface/hub/models--OFA-Sys--chinese-clip-vit-large-patch14-336px/deploy/"