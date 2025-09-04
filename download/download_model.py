from huggingface_hub import snapshot_download
import os

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

# HF_HOME = "/run/determined/NAS1/public/HuggingFace/"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
local_model_dir = "/run/determined/NAS1/public/HuggingFace/BAGEL-7B-MoT"
os.makedirs(local_model_dir, exist_ok=True)
snapshot_download(
    # cache_dir=HF_HOME,
    local_dir = local_model_dir,
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
