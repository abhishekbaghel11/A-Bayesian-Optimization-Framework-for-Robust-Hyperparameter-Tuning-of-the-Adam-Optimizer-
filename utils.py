import torch
import subprocess
import os

def setup_device():
    """Setup device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    return device

def run_cmd(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def ensure_repo(repo_dir, git_url):
    if not os.path.exists(repo_dir):
        run_cmd(["git", "clone", git_url, repo_dir])