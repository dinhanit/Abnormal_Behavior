
import torch
print("Torch CUDA Available: ", torch.cuda.is_available())
print("Current GPU Device: ", torch.cuda.get_device_name(0))
