# mnist_training.py
import torch

print("Is GPU available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Using GPU", torch.cuda.get_device_name(0))
