import os
import numpy as np
import matplotlib.pyplot as plt

# Configurations
image_shape = (3, 32, 32)  # Define as (channels, height, width)
num_images_to_plot = 20
dataset_name = "CIFAR10"

input_binary_file = f"generateAndRead/binSamples/CIFAR10_validation_samples.bin"
output_folder = "generateAndRead/plots"
output_plot_file = os.path.join(output_folder, f"val_first_{num_images_to_plot}_images_in_{dataset_name}.png")

# Function to read the binary file
def read_binary_file(file_path, image_shape, num_images):
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
    bytes_per_image = 1 + np.prod(image_shape)
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
images = read_binary_file(input_binary_file, image_shape, num_images_to_plot)

# Prepare output directory
os.makedirs(output_folder, exist_ok=True)

# Plot the first images
fig, axes = plt.subplots(1, num_images_to_plot, figsize=(15, 5))

for i, ax in enumerate(axes):
    if image_shape[0] == 1:
        ax.imshow(images[i, 0], cmap="gray")
    elif image_shape[0] == 3:
        ax.imshow(images[i].transpose(1, 2, 0))
    ax.axis("off")
    ax.set_title(f"Image {i+1}")

plt.tight_layout()
plt.savefig(output_plot_file)
plt.close(fig)

print(f"Plot saved to {output_plot_file}.")
