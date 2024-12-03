# Import necessary libraries
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_gan as tfgan
import numpy as np
import logging
import os
import math

def read_binary_file(file_path, image_shape, isValidation):
    """
    Reads a binary file containing labeled images.

    Args:
        file_path (str): Path to the binary file.
        image_shape (tuple): Shape of each image (channels, height, width).

    Returns:
        np.ndarray: Array of images of shape (num_images, channels, height, width).
    """
    # Bytes per image: 1 byte for label + image data
    if isValidation:
        bytes_per_image = np.prod(image_shape) + 1
    else:
        bytes_per_image = np.prod(image_shape)

    with open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    total_bytes = os.path.getsize(file_path)
    num_images = total_bytes // bytes_per_image - 1

    images = []
    for i in range(num_images):
        start = i * bytes_per_image + 1  # Skip the label
        end = start + np.prod(image_shape)
        image = data[start:end].reshape(image_shape)
        images.append(image)
    return np.array(images)

def compute_activations(images, model):
    """
    Compute activations for the given images using the provided model.

    Args:
        images (tf.Tensor): Tensor of images (NHWC format).
        model: Preloaded Inception model.

    Returns:
        tf.Tensor: Activations from the model.
    """
    images = tf.image.resize(images, [299, 299])  # Resize to Inception input size
    return model(images, training=False)

def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Compute the Frechet Distance between two distributions.

    Args:
        mu1, sigma1: Mean and covariance of distribution 1.
        mu2, sigma2: Mean and covariance of distribution 2.

    Returns:
        float: Frechet distance.
    """
    diff = mu1 - mu2
    covmean = tf.linalg.sqrtm(tf.matmul(sigma1, sigma2))
    if tf.math.reduce_any(tf.math.is_nan(covmean)):
        covmean = tf.linalg.eye(sigma1.shape[0])  # Fallback for numerical issues
    return tf.reduce_sum(diff ** 2) + tf.linalg.trace(sigma1 + sigma2 - 2 * covmean)

def compute_fid_for_CIFAR_or_CELEBA(dataset, image_shape=(3, 32, 32), max_batches=50, batch_size=100):
    """
    Computes the FID score for CIFAR10 or CELEBA dataset.

    Args:
        dataset (str): Dataset name ("CIFAR10" or "CELEBA").
        image_shape (tuple): Shape of each image (channels, height, width).
        max_batches (int): Maximum number of batches to limit computation.
        batch_size (int): Number of images to process in each batch.

    Returns:
        float: FID score.
    """
    # Preload the Inception model from TensorFlow Hub
    logger.info("Loading Inception model...")
    inception_model = hub.load("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5")

    # Define file paths for datasets
    if dataset == "CIFAR10":
        file_path = "generateAndRead/binSamples/model_CIFAR10_10000samples.bin"
        validation_path = "generateAndRead/binSamples/CIFAR10_validation_samples.bin"
    elif dataset == "CELEBA":
        file_path = "generateAndRead/binSamples/model_CELEBA_10000samples.bin"
        validation_path = "generateAndRead/binSamples/CELEBA_validation_samples.bin"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load generated images
    logger.info(f"Reading generated images from binary file for {dataset}...")
    generated_images = read_binary_file(file_path, image_shape, isValidation=False)
    generated_images = generated_images / 255.0  # Normalize to [0, 1]
    generated_images_tf = tf.convert_to_tensor(generated_images, dtype=tf.float32)
    generated_images_tf = tf.transpose(generated_images_tf, [0, 2, 3, 1])  # Convert to NHWC format

    # Load validation images
    logger.info(f"Reading validation images from binary file for {dataset}...")
    validation_images = read_binary_file(validation_path, image_shape, isValidation=True)
    validation_images = validation_images / 255.0  # Normalize to [0, 1]
    validation_images_tf = tf.convert_to_tensor(validation_images, dtype=tf.float32)
    validation_images_tf = tf.transpose(validation_images_tf, [0, 2, 3, 1])  # Convert to NHWC format

    # Compute activations in batches
    def compute_stats(images):
        activations = []
        for start in range(0, images.shape[0], batch_size):
            end = min(start + batch_size, images.shape[0])
            batch_activations = compute_activations(images[start:end], inception_model)
            activations.append(batch_activations)
        activations = tf.concat(activations, axis=0)
        mu = tf.reduce_mean(activations, axis=0)
        sigma = tfp.stats.covariance(activations)
        return mu, sigma

    logger.info("Computing statistics for generated images...")
    mu_gen, sigma_gen = compute_stats(generated_images_tf)
    logger.info("Computing statistics for validation images...")
    mu_val, sigma_val = compute_stats(validation_images_tf)

    # Compute Frechet Distance
    logger.info("Computing FID score...")
    fid = compute_frechet_distance(mu_gen, sigma_gen, mu_val, sigma_val)
    logger.info(f"Computed FID score for {dataset}: {fid}")
    return fid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example usage
fid_cifar10 = compute_fid_for_CIFAR_or_CELEBA(
    dataset="CIFAR10",
    image_shape=(3, 32, 32)
)

print(f"Computed FID for CIFAR10: {fid_cifar10}")
