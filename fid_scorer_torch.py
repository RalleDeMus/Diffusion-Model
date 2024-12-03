import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import Resize, Normalize, ToTensor, Compose
from scipy.linalg import sqrtm
import os
from PIL import Image



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


# Define the image shape
image_shape = (3, 32, 32)

# Paths to the binary files
file_path = "generateAndRead/binSamples/model_CIFAR10_10000samples.bin"
validation_path = "generateAndRead/binSamples/CIFAR10_validation_samples.bin"

# Load the data
generated_images = read_binary_file(file_path, image_shape, isValidation=False)
validation_images = read_binary_file(validation_path, image_shape, isValidation=True)


def preprocess_images(images):
    preprocess = Compose([
        Resize((299, 299)),  # Resize to InceptionV3 input size
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for InceptionV3
    ])
    # Convert NumPy array to PIL Image and apply transformations
    preprocessed_images = torch.stack([
        preprocess(Image.fromarray(image.transpose(1, 2, 0))) for image in images
    ])
    return preprocessed_images

# Preprocess the datasets
generated_images = preprocess_images(generated_images)
validation_images = preprocess_images(validation_images)

# Load the InceptionV3 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inception_v3(pretrained=True, transform_input=False).eval().to(device)

# Extract features function
def extract_features(images, model):
    with torch.no_grad():
        features = model(images.to(device))
    return features.cpu().numpy()

# Extract features for both datasets
generated_features = extract_features(generated_images, model)
validation_features = extract_features(validation_images, model)

# Compute FID score
def calculate_fid(features1, features2):
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

fid_score = calculate_fid(validation_features, generated_features)
print("FID Score:", fid_score)
