import torch

def check_device():
    if torch.cuda.is_available():
        print("GPU available - using CUDA")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("GPU available - using Metal (MPS)")
    else:
        print("No GPU detected - using CPU")

    # Create a sample tensor - it will automatically use the best available device
    device = torch.device("cuda" if torch.cuda.is_available() 
                         else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                         else "cpu")
    tensor = torch.rand(3, 3, device=device)
    print(f"\nCreated tensor on {device}:\n{tensor}")

if __name__ == "__main__":
    check_device() 