import os
import numpy as np
import matplotlib.pyplot as plt

# Configurations
image_shape = (28, 28)
num_images_to_plot = 15

input_binary_file = "generateAndRead/binSamples/generated_samples_10000.bin"
output_folder = "generateAndRead/plots"
output_plot_file = os.path.join(output_folder, f"first_{num_images_to_plot}_images.png")


# Function to read the binary file
def read_binary_file(file_path, image_shape, num_images):
    """
    Reads a binary file containing grayscale images.

    Args:
        file_path (str): Path to the binary file.
        image_shape (tuple): Shape of each image (height, width).
        num_images (int): Number of images to read.

    Returns:
        np.ndarray: Array of images of shape (num_images, height, width).
    """
    bytes_per_image = np.prod(image_shape)
    with open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data[:num_images * bytes_per_image].reshape((num_images, *image_shape))
    return images

# Read the first 10 images
images = read_binary_file(input_binary_file, image_shape, num_images_to_plot)

# Plot the first 10 images
os.makedirs(output_folder, exist_ok=True)
fig, axes = plt.subplots(1, num_images_to_plot, figsize=(15, 5))

for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap="gray")
    ax.axis("off")
    ax.set_title(f"Image {i+1}")

plt.tight_layout()
plt.savefig(output_plot_file)
plt.close(fig)

print(f"Plot saved to {output_plot_file}.")
