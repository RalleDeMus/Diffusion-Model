import sys
import os

# Add the directory containing the module to the Python path
external_module_path = "/zhome/1a/a/156609/project/path"
sys.path.append(external_module_path)


import torch
import numpy as np
from configs.config import config_CIFAR10, config_CELEBA, config_MNIST
from UNetResBlock.model import UNet, generate_samples
import matplotlib.pyplot as plt

# Configurations
config = config_CELEBA
model_filename = "ResNetCELEBA_414974ckpt.pt"

# idk
image_shape = config["image_shape"]  # Channels, Height, Width

# Define the output folder and ensure it exists
output_folder = "generateAndRead/noising"
os.makedirs(output_folder, exist_ok=True)
dataset_name = config["dataset_name"]

# Load the saved model
model_folder = "savedModels"
model_path = os.path.join(model_folder, model_filename)


channels = config["image_shape"][0]
dim = config["image_shape"][1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize and load the model
u_net = UNet(in_channels=channels, dim = config["image_shape"][1], out_channels=channels)
u_net = u_net.to(device)
state_dict = torch.load(model_path, weights_only=True)
u_net.load_state_dict(state_dict)
u_net.eval()

def linear_beta_schedule(timesteps):
    beta_start = 0.0001  # Small noise variance at start
    beta_end = 0.02  # Larger noise variance at end
    return torch.linspace(beta_start, beta_end, timesteps)

def compute_alpha_and_alpha_bar(betas):
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar


# Updated generate_samples_noise to ensure CUDA
def generate_samples_noise(u_net, nsamples, image_shape, timesteps=1000, interval=100):
    # Define the linear beta schedule
    betas = linear_beta_schedule(timesteps).to(device)  # Move to device
    alphas, alpha_bar = compute_alpha_and_alpha_bar(betas)
    alphas = alphas.to(device)
    alpha_bar = alpha_bar.to(device)

    x_t = torch.randn((nsamples, *image_shape), device=device)  # Start from pure noise

    sampled_images = []  # To store every 100th sample

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((x_t.size(0),), t, device=device).long()  # Current time step
        
        # Compute the variance terms
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]
        alpha_bar_prev = alpha_bar[t - 1] if t > 0 else 1.0
        
        # Predict the noise
        predicted_noise = u_net(x_t, t_tensor / timesteps).detach()
        
        # Reconstruct the mean (mu) of x_{t-1}
        mu = (1 / torch.sqrt(alpha_t)) * (
            x_t - (betas[t] / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )
        
        # Add noise for all steps except the final one
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * betas[t])
            x_t = mu + sigma_t * noise
        else:
            x_t = mu  # No noise added at the final step
        
        # Save every 100th sample
        if t % interval == 0:
            sampled_images.append(x_t.clone().cpu())  # Clone to avoid overwriting
    
    return sampled_images  # Return the list of sampled images

import matplotlib.pyplot as plt
import numpy as np
import os

# Set the number of samples to 10
num_samples = 50
timesteps = 1000
interval = 100  # Save every 100th timestep

# Generate samples using the updated generate_samples function
samples = generate_samples_noise(u_net, num_samples, image_shape, timesteps, interval)

# Convert the samples into a grid format (rows: samples, columns: timesteps)
combined_image = []
for i in range(num_samples):  # Iterate over the number of samples
    images_row = []
    for sample in samples:  # Iterate over timesteps for this sample
        single_sample = sample[i].squeeze().detach().cpu().numpy()
        single_sample = (single_sample - single_sample.min()) / (single_sample.max() - single_sample.min())  # Normalize to [0, 1]

        if image_shape[0] == 1:  # Grayscale image
            images_row.append(single_sample)  # Append grayscale image
        else:  # RGB image
            images_row.append(np.transpose(single_sample, (1, 2, 0)))  # Transpose to HxWxC

    # Concatenate all 10 images for this row horizontally
    if image_shape[0] == 1:  # Grayscale
        combined_row = np.hstack(images_row)
    else:  # RGB
        combined_row = np.hstack(images_row)

    combined_image.append(combined_row)  # Append this row to the grid

# Concatenate all rows vertically to create the grid
final_combined_image = np.vstack(combined_image)

# Save the final grid image as a single file
output_combined_image_path = os.path.join(output_folder, f"combined_samples_grid_{dataset_name}.png")
if image_shape[0] == 1:  # Grayscale
    plt.imsave(output_combined_image_path, final_combined_image, cmap="gray")
else:  # RGB
    plt.imsave(output_combined_image_path, final_combined_image)

print(f"Combined image saved to {output_combined_image_path}")

