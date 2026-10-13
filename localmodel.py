from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv


HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT


model_id = "Qwen/Qwen2.5-1.5B-Instruct"
local_dir = "./models/Qwen2.5"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True, # 断点续传
)