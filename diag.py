import torch
import sys
import subprocess
import os

def check_gpu_status():
    print("## PyTorch GPU Configuration")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA memory allocation
        try:
            test_tensor = torch.ones(1000, 1000).cuda()
            print("Successfully allocated test tensor on GPU")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error allocating tensor on GPU: {e}")
    
    print("\n## Environment Variables")
    cuda_vars = {k: v for k, v in os.environ.items() if 'CUDA' in k}
    for k, v in cuda_vars.items():
        print(f"{k}: {v}")
    
    print("\n## NVIDIA-SMI Output")
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode()
        print(nvidia_smi)
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

if __name__ == "__main__":
    check_gpu_status()