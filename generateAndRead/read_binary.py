import sys
import os

# Add the directory containing the module to the Python path
external_module_path = "/zhome/1a/a/156609/project/path"
sys.path.append(external_module_path)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from configs.config import config_CIFAR10, config_CELEBA, config_MNIST


config = config_CELEBA
num_images_to_plot = 500


# Configurations
image_shape = config["image_shape"]  # Channels, Height, Width
dataset_name = config["dataset_name"]

# input_binary_file = f"generateAndRead/binSamples/model_{dataset_name}_10000samples_int.bin"
input_binary_file = f"generateAndRead/binSamples/model_CELEBA_10000samples_ema.bin"

output_folder = "generateAndRead/plots"
output_plot_file = os.path.join(output_folder, f"first_{num_images_to_plot}_images_in_{dataset_name}_ema.png")

# Function to read the binary file
def read_binary_file(file_path, image_shape, num_images, val):
    """
    Reads a binary file containing labeled images.

    Args:
        file_path (str): Path to the binary file.
        image_shape (tuple): Shape of each image (channels, height, width).
        num_images (int): Number of images to read.

    Returns:
        np.ndarray: Array of images of shape (num_images, channels, height, width).
    """
    
    # Bytes per image: 1 byte for label + image data
    bytes_per_image =  np.prod(image_shape) + (1 if val else 0)
    with open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Ensure enough data is available for the requested number of images
    total_bytes = num_images * bytes_per_image
    assert len(data) >= total_bytes, "Not enough data in the file for the requested number of images."
    
    # Reshape image data
    images = []
    for i in range(num_images):
        start = i * bytes_per_image + 1  # Skip the label
        end = start + np.prod(image_shape)
        image = data[start:end].reshape(image_shape)
        images.append(image)
    return np.array(images)

# Read the first images
images = read_binary_file(input_binary_file, image_shape, num_images_to_plot, False)

# Prepare output directory
os.makedirs(output_folder, exist_ok=True)

# Calculate the grid dimensions
max_images_per_row = 20
num_rows = (num_images_to_plot + max_images_per_row - 1) // max_images_per_row  # Round up
row_width = max_images_per_row * image_shape[2]
grid_height = num_rows * image_shape[1]

# Create the grid
if image_shape[0] == 1:  # Grayscale
    grid = np.zeros((grid_height, row_width), dtype=np.uint8)
else:  # RGB
    grid = np.zeros((grid_height, row_width, 3), dtype=np.uint8)

# Fill the grid with images
for idx, image in enumerate(images):
    row = idx // max_images_per_row
    col = idx % max_images_per_row
    start_y = row * image_shape[1]
    end_y = start_y + image_shape[1]
    start_x = col * image_shape[2]
    end_x = start_x + image_shape[2]
    if image_shape[0] == 1:  # Grayscale
        grid[start_y:end_y, start_x:end_x] = image[0]
    else:  # RGB
        grid[start_y:end_y, start_x:end_x] = image.transpose(1, 2, 0)

# Save the final grid as a single image
if image_shape[0] == 1:  # Grayscale
    Image.fromarray(grid, mode="L").save(output_plot_file)
else:  # RGB
    Image.fromarray(grid).save(output_plot_file)

print(f"Plot saved to {output_plot_file}.")

