import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FFMpegWriter
from UNetResBlock.model import UNet, compute_alpha_and_alpha_bar, linear_beta_schedule

# Function to load the U-Net model
def load_unet_model(checkpoint_path, device):
    model = UNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.to(device)  # Ensure the model is on the correct device
    model.eval()
    return model

# Function to generate samples
def generate_samples_with_timesteps(u_net, nsamples, image_shape, timesteps=1000):
    device = next(u_net.parameters()).device  # Ensure we use the same device as the model

    # Define the linear beta schedule
    betas = linear_beta_schedule(timesteps)
    alphas, alpha_bar = compute_alpha_and_alpha_bar(betas)

    # Create initial noise and allocate memory for samples
    x_t = torch.randn((nsamples, *image_shape), device=device)
    samples = torch.zeros((image_shape[0], image_shape[1], image_shape[2], timesteps, nsamples), device=device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((x_t.size(0),), t, device=device).long()
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]
        alpha_bar_prev = alpha_bar[t - 1] if t > 0 else 1.0
        predicted_noise = u_net(x_t, t_tensor / timesteps).detach()
        mu = (1 / torch.sqrt(alpha_t)) * (
            x_t - (betas[t] / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * betas[t])
            x_t = mu + sigma_t * noise
        else:
            x_t = mu
        samples[:, :, :, t, :] = x_t.permute(1, 2, 3, 0)

    return samples

# Function to render samples to video
def render_samples_video(output_path, u_net, nsamples=5, image_shape=(3, 32, 32), timesteps=1000, fps=30):
    
    # Generate samples
    samples = generate_samples_with_timesteps(u_net, nsamples, image_shape, timesteps)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Denormalize from [-1, 1] to [0, 1] and clip
    samples = torch.clamp((samples + 1) / 2.0, 0, 1).cpu().numpy()

    # Metadata and writer setup
    metadata = dict(title='Generated Samples', artist='Matplotlib', comment='Diffusion Process')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig, axes = plt.subplots(1, nsamples, figsize=(nsamples * 3, 3))
    plt.tight_layout()

    # Render video
    with writer.saving(fig, output_path, dpi=100):
        for t in range(timesteps):
            if t % 10 != 0:  # Skip frames not divisible by 10
                continue
            for i in range(nsamples):
                ax = axes[i] if nsamples > 1 else axes
                ax.clear()
                ax.imshow(samples[:, :, :, t, i].transpose(1, 2, 0), interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'Sample {i+1}, Step {t+1}')
            writer.grab_frame()
            print(f"Rendering frame {t} / {timesteps}")

    print(f"Video saved to {output_path}")


def save_noise_image(output_path, image_shape=(3, 32, 32)):
    """
    Save a single pure noise image as a PNG file.

    Args:
        output_path (str): Path to save the noise image.
        image_shape (tuple): Shape of the noise image (channels, height, width).
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate pure noise
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.randn(image_shape, device=device).cpu().numpy()
    noise = (noise - noise.min()) / (noise.max() - noise.min())  # Normalize to [0, 1]

    # Convert to HWC format for visualization
    noise = noise.transpose(1, 2, 0)

    # Save the noise image
    plt.figure(figsize=(6, 6))
    plt.imshow(noise, cmap='viridis', interpolation='nearest')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Noise image saved at {output_path}")

# output_file = "videos/pure_noise.png"
# save_noise_image(output_file, image_shape=(3, 32, 32))

model_path = "UNetResBlock/models/ResNetCIFAR10_401843ckpt.pt"
output_file = "videos/generated_samples.mp4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
u_net = load_unet_model(model_path, device)
render_samples_video(output_file, u_net, nsamples=5, image_shape=(3, 32, 32), timesteps=1000)
