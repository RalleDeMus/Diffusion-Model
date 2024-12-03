# -*- coding: utf-8 -*-
"""FID scorer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xUJML29cjxFWm60Ai05BB8iYgN9CY3sq
"""

# Install the required versions of TensorFlow, TensorFlow Probability, and TensorFlow GAN
#!pip install tensorflow==2.15 tensorflow-probability==0.23 tensorflow-gan matplotlib tqdm

# Import necessary libraries
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_gan as tfgan
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained MNIST classifier
MNIST_MODULE = "https://tfhub.dev/tensorflow/tfgan/eval/mnist/logits/1"
mnist_classifier_fn = tfhub.load(MNIST_MODULE)

IMAGE_MODULE = "https://www.kaggle.com/models/tensorflow/inception/tensorFlow1/tfgan-eval-inception/1"
image_classifier_fn = tfhub.load(IMAGE_MODULE)

def wrapped_image_classifier_fn(input_tensor):
    # Ensure input_tensor has 3 channels (convert grayscale to RGB)
    if input_tensor.shape[-1] == 1:  # Check if it's grayscale
        input_tensor = tf.image.grayscale_to_rgb(input_tensor)

    # Resize images to 299x299 for Inception
    # input_tensor = tf.image.resize(input_tensor, [299, 299])

    # Pass the processed input to the classifier
    output = image_classifier_fn(input_tensor)
    
    return output['pool_3']  # Use 'pool_3' for FID, or adjust based on requirements


def pack_images_to_tensor(path, img_size=None):
    """
    Given a path, pack all images into a tensor of shape (nb_images, height, width, channels)
    """
    nb_images = len(list(path.rglob("*.png")))
    logger.info(f"Computing statistics for {nb_images} images")
    images = np.empty((nb_images, 28, 28, 1))  # TODO: Consider the RGB case
    for idx, f in enumerate(tqdm(path.rglob("*.png"))):
        img = Image.open(f)
        # resize if not the right size
        if img_size is not None and img.size[:2] != img_size:
            img = img.resize(
                size=(img_size[0], img_size[1]),
                resample=Image.BILINEAR,
            )
        img = np.array(img) / 255.0
        images[idx] = img[..., None]
    images_tf = tf.convert_to_tensor(images, dtype=tf.float32)
    return images_tf



def compute_activations(tensors, num_batches, classifier_fn):
    """
    Given a tensor of of shape (batch_size, height, width, channels), computes
    the activiations given by classifier_fn.
    """
    tensors_list = tf.split(tensors, num_or_size_splits=num_batches)
    stack = tf.stack(tensors_list)
    activation = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(classifier_fn, stack, parallel_iterations=1, swap_memory=True),
    )
    return tf.concat(tf.unstack(activation), 0)



def save_activations(activations, path):
    np.save(path, activations.numpy())

def read_binary_file(file_path, image_shape, isValidation):
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
    if isValidation:
        bytes_per_image = np.prod(image_shape) + 1
    else:
        bytes_per_image =  np.prod(image_shape)

    with open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    total_bytes = os.path.getsize(file_path)

    num_images =  total_bytes // bytes_per_image - 1

    images = []
    for i in range(num_images):
        start = i * bytes_per_image + 1  # Skip the label
        end = start + np.prod(image_shape)
        image = data[start:end].reshape(image_shape)
        images.append(image)
    return np.array(images)

# Updated function to load validation activations

def compute_fid_for_validation(file_path, validation_path, output_dir, mnist_classifier_fn, image_shape=(1, 32, 32)):
    """
    Computes FID score for generated samples from a binary file against validation samples.

    Args:
        file_path (str): Path to the binary file for generated images.
        validation_path (str): Path to the binary file for validation images.
        output_dir (str): Directory to save intermediate results.
        mnist_classifier_fn: Pre-trained MNIST classifier function.
        num_images (int): Number of images to process.
        image_shape (tuple): Shape of images in the binary file (channels, height, width).
        is_validation (bool): Whether the binary file includes labels.

    Returns:
        float: FID score.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load generated images
    logger.info(f"Reading generated images from binary file...")
    generated_images = read_binary_file(file_path, image_shape, isValidation=False)
    generated_images = generated_images / 255.0  # Normalize to [0, 1]
    generated_images_tf = tf.convert_to_tensor(generated_images, dtype=tf.float32)
    generated_images_tf = tf.transpose(generated_images_tf, [0, 2, 3, 1])  # Convert to NHWC format

    # Resize to (28, 28)
    logger.info("Resizing generated images to (28, 28)...")
    generated_images_tf = tf.image.resize(generated_images_tf, [28, 28])

    # Compute activations for generated images
    logger.info("Computing activations for generated images...")
    activations_fake = compute_activations(generated_images_tf, num_batches=1, classifier_fn=mnist_classifier_fn)

    # Load validation images
    logger.info(f"Reading validation images from binary file...")
    validation_images = read_binary_file(validation_path, image_shape, isValidation=True)
    validation_images = validation_images / 255.0  # Normalize to [0, 1]
    validation_images_tf = tf.convert_to_tensor(validation_images, dtype=tf.float32)
    validation_images_tf = tf.transpose(validation_images_tf, [0, 2, 3, 1])  # Convert to NHWC format

    # Resize to (28, 28)
    logger.info("Resizing validation images to (28, 28)...")
    validation_images_tf = tf.image.resize(validation_images_tf, [28, 28])

    # Compute activations for validation images
    logger.info("Computing activations for validation images...")
    activations_real = compute_activations(validation_images_tf, num_batches=1, classifier_fn=mnist_classifier_fn)

    # Compute FID
    logger.info("Computing FID score...")
    fid_score = tfgan.eval.frechet_classifier_distance_from_activations(activations_real, activations_fake)
    logger.info(f"FID score: {fid_score}")

    # Save activations and results
    np.save(os.path.join(output_dir, "activations_fake.npy"), activations_fake.numpy())
    np.save(os.path.join(output_dir, "activations_real.npy"), activations_real.numpy())

    return fid_score.numpy()

# # Paths to binary files
# binary_file_path = "/content/drive/MyDrive/binSamples/model_MNIST_10000samples.bin"
# validation_file_path = "/content/drive/MyDrive/binSamples/MNIST_validation_samples.bin"
# output_directory = "./output"
# image_dimensions = (1, 32, 32)  # Channel, height, width

# images = read_binary_file(
#     file_path=binary_file_path,
#     image_shape=image_dimensions,
#     isValidation=False
# )

import matplotlib.pyplot as plt

def plot_images(images, num_images=10, title="Images from Binary File"):
    """
    Plots images from the binary file.

    Args:
        images (np.ndarray): Array of images with shape (num_images, channels, height, width).
        num_images (int): Number of images to display.
        title (str): Title for the plot.
    """
    # If the images are in CHW format, transpose to HWC for plotting
    images = images.transpose(0, 2, 3, 1)  # Convert (N, C, H, W) to (N, H, W, C)

    # If images are grayscale, squeeze the last dimension
    if images.shape[-1] == 1:
        images = images.squeeze(-1)

    # Create a grid for plotting
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

    for i in range(num_images):
        axes[i].imshow(images[i], cmap="gray" if images.ndim == 3 else None)
        axes[i].axis("off")
        axes[i].set_title(f"Image {i+1}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Plot the images from the binary file
#plot_images(images, num_images=20, title="Generated Images from Binary File")

# # Compute FID score
# fid_mnist = compute_fid_for_validation(
#     file_path=binary_file_path,
#     validation_path=validation_file_path,
#     output_dir=output_directory,
#     mnist_classifier_fn=mnist_classifier_fn,
#     image_shape=image_dimensions,
# )

# print(f"Computed FID: {fid_mnist}")

def compute_fid_for_CIFAR_or_CELEBA(dataset, image_shape):
    """
    Computes FID for CIFAR10 or CELEBA dataset using TensorFlow GAN utilities.

    Args:
        dataset (str): Name of the dataset ('CIFAR10' or 'CELEBA').
        image_shape (tuple): Shape of each image (channels, height, width).
        batch_size (int): Number of images to process in each batch.

    Returns:
        float: Computed FID score.
    """
    # File paths
    if dataset == "CIFAR10":
        generated_path = "generateAndRead/binSamples/model_CIFAR10_10000samples.bin"
        validation_path = "generateAndRead/binSamples/CIFAR10_validation_samples.bin"
    elif dataset == "CELEBA":
        generated_path = "generateAndRead/binSamples/model_CELEBA_10000samples_int.bin"
        validation_path = "generateAndRead/binSamples/CELEBA_validation_samples.bin"
    elif dataset == "MNIST":
        generated_path = "generateAndRead/binSamples/model_MNIST_10000samples.bin"
        validation_path = "generateAndRead/binSamples/MNIST_validation_samples.bin"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    output_dir = "./output"

    os.makedirs(output_dir, exist_ok=True)

    # Load generated images
    logger.info(f"Reading generated images from binary file...")
    generated_images = read_binary_file(generated_path, image_shape, isValidation=False)
    generated_images = generated_images / 255.0  # Normalize to [0, 1]
    generated_images_tf = tf.convert_to_tensor(generated_images, dtype=tf.float32)
    generated_images_tf = tf.transpose(generated_images_tf, [0, 2, 3, 1])  # Convert to NHWC format

    # Compute activations for generated images
    logger.info("Computing activations for generated images...")
    activations_fake = compute_activations(generated_images_tf, num_batches=1, classifier_fn=wrapped_image_classifier_fn)

    # Load validation images
    logger.info(f"Reading validation images from binary file...")
    validation_images = read_binary_file(validation_path, image_shape, isValidation=True)
    validation_images = validation_images / 255.0  # Normalize to [0, 1]
    validation_images_tf = tf.convert_to_tensor(validation_images, dtype=tf.float32)
    validation_images_tf = tf.transpose(validation_images_tf, [0, 2, 3, 1])  # Convert to NHWC format

    # Compute activations for validation images
    logger.info("Computing activations for validation images...")
    activations_real = compute_activations(validation_images_tf, num_batches=1, classifier_fn=wrapped_image_classifier_fn)

    # Compute FID
    logger.info("Computing FID score...")
    # Reshape activations to rank-2 tensors
    activations_fake = tf.reshape(activations_fake, [activations_fake.shape[0], activations_fake.shape[-1]])
    activations_real = tf.reshape(activations_real, [activations_real.shape[0], activations_real.shape[-1]])

    # Compute FID
    fid_score = tfgan.eval.frechet_classifier_distance_from_activations(activations_real, activations_fake)

    logger.info(f"FID score: {fid_score}")

    # Save activations and results
    np.save(os.path.join(output_dir, "activations_fake.npy"), activations_fake.numpy())
    np.save(os.path.join(output_dir, "activations_real.npy"), activations_real.numpy())

    return fid_score.numpy()

# Example usage
fid_cifar10 = compute_fid_for_CIFAR_or_CELEBA(
    dataset="CELEBA",
    image_shape=(3, 64, 64)
)

print(f"Computed FID for CIFAR10: {fid_cifar10}")
