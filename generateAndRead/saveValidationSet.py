import os
import sys

# Add the directory containing the module to the Python path
external_module_path = "/zhome/1a/a/156609/project/path"
sys.path.append(external_module_path)

import numpy as np
from utils.dataset_loader import load_dataset
from configs.config import config_MNIST, config_CIFAR10, config_CELEBA
import torch

# Configurations
config = config_MNIST  # Change this to config_MNIST or config_CIFAR10 as needed
batch_size = 128  # Batch size for DataLoader
output_folder = "generateAndRead/binSamples"
os.makedirs(output_folder, exist_ok=True)
dataset_name = config["dataset_name"]
output_binary_file = os.path.join(output_folder, f"{dataset_name}_validation_samples.bin")

# Load the dataset
print(f"Loading validation dataset: {dataset_name}")
dataset = load_dataset(config, validation=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Determine the shape from the config
image_shape = config["image_shape"]  # (channels, height, width)
num_channels = image_shape[0]
image_size = image_shape[1] * image_shape[2]

# Function to unnormalize images from [-1, 1] to [0, 255]
def unnormalize(images):
    return ((images * 0.5) + 0.5) * 255

# Open a binary file for writing
with open(output_binary_file, "wb") as f:
    print("Saving validation samples to binary file...")
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Unnormalize images
        images = unnormalize(images).byte().numpy()  # Scale back to [0, 255]
        
        for image, label in zip(images, labels):
            # Save label (1 byte) + image data (channels * height * width bytes)
            f.write(label.numpy().astype(np.uint8).tobytes())  # Write label
            f.write(image.tobytes())  # Write image data

print(f"All validation samples saved to {output_binary_file}.")
