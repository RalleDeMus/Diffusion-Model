import sys
import os

# Add the directory containing the module to the Python path
external_module_path = "/zhome/1a/a/156609/project/path"
sys.path.append(external_module_path)


import torch
import numpy as np
from configs.config import config_CIFAR10, config_CELEBA, config_MNIST
from UNetResBlock.model import UNet, generate_samples

# Configurations
config = config_CELEBA
model_filename = "ResNetCELEBA_518406_ema.pt"
batch_size = 200  # Number of samples to generate per batch
num_samples = 10000  # Number of samples to generate

# idk
image_shape = config["image_shape"]  # Channels, Height, Width

# Define the output folder and ensure it exists
output_folder = "generateAndRead/binSamples"
os.makedirs(output_folder, exist_ok=True)
dataset_name = config["dataset_name"]
output_binary_file = os.path.join(output_folder, f"model_{dataset_name}_{num_samples}samples_ema.bin")

# Load the saved model
model_folder = "savedModels"
model_path = os.path.join(model_folder, model_filename)


channels = config["image_shape"][0]
dim = config["image_shape"][1]
# Initialize and load the model
u_net = UNet(in_channels=channels, dim = config["image_shape"][1], out_channels=channels)
state_dict = torch.load(model_path, weights_only=True)
u_net.load_state_dict(state_dict)
u_net.eval()

# Generate samples in parallel batches
print("Generating samples in parallel...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
u_net.to(device)

# Adjust the shape of `all_samples` to include channels
all_samples = np.zeros((num_samples, *image_shape), dtype=np.uint8)

for i in range(0, num_samples, batch_size):
    current_batch_size = min(batch_size, num_samples - i)
    print(f"Generating batch {i // batch_size + 1}: {current_batch_size} samples...")
    
    # Generate samples for the current batch
    samples = generate_samples(u_net, current_batch_size, image_shape).detach().cpu().numpy()
    
    # Normalize the samples
    samples = (samples - samples.min(axis=(1, 2, 3), keepdims=True)) / \
              (samples.max(axis=(1, 2, 3), keepdims=True) - samples.min(axis=(1, 2, 3), keepdims=True)) * 255
    samples = samples.astype(np.uint8)

    # Store samples directly without squeezing the channel dimension
    all_samples[i:i + current_batch_size] = samples

# Save all samples to a binary file
with open(output_binary_file, "wb") as f:
    f.write(all_samples.tobytes())

print(f"All samples saved to {output_binary_file}.")

