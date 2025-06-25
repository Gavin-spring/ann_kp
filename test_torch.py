import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version used by PyTorch: {torch.version.cuda}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")