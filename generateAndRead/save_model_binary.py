import sys
import os

# Add the directory containing the module to the Python path
external_module_path = "/zhome/1a/a/156609/project/path"
sys.path.append(external_module_path)


import torch
import numpy as np
from configs.MNIST import config
from models.ModelLayers.MNIST import GetEncDecLayers
from models.attention_model.attention_model import generate_samples
from models.attention_model.attention_model import ScoreNetwork

# Configurations
num_samples = 10000  # Number of samples to generate
image_shape = (1, 28, 28)  # Channels, Height, Width
batch_size = 500  # Number of samples to generate per batch

# Define the output folder and ensure it exists
output_folder = "output_samples"
os.makedirs(output_folder, exist_ok=True)
output_binary_file = os.path.join(output_folder, f"generated_samples_{num_samples}.bin")

# Load the saved model
model_folder = "savedModels"
model_filename = "MNISTNetwork_500epochs2.pt"
model_path = os.path.join(model_folder, model_filename)

# Initialize and load the model
score_network = ScoreNetwork(in_channels=config["image_shape"][0], layers=GetEncDecLayers())
state_dict = torch.load(model_path, weights_only=True)
score_network.load_state_dict(state_dict)
score_network.eval()

# Allocate memory for all samples
all_samples = np.zeros((num_samples, 28, 28), dtype=np.uint8)

# Generate samples in parallel batches
print("Generating samples in parallel...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
score_network.to(device)

for i in range(0, num_samples, batch_size):
    current_batch_size = min(batch_size, num_samples - i)
    print(f"Generating batch {i // batch_size + 1}: {current_batch_size} samples...")
    
    # Generate samples for the current batch
    samples = generate_samples(score_network, current_batch_size, image_shape).detach().cpu().numpy()
    
    # Normalize and store the samples
    samples = (samples - samples.min(axis=(1, 2, 3), keepdims=True)) / \
              (samples.max(axis=(1, 2, 3), keepdims=True) - samples.min(axis=(1, 2, 3), keepdims=True)) * 255
    samples = samples.astype(np.uint8).squeeze(1)  # Remove channel dimension
    all_samples[i:i + current_batch_size] = samples

# Save all samples to a binary file
with open(output_binary_file, "wb") as f:
    f.write(all_samples.tobytes())

print(f"All samples saved to {output_binary_file}.")
