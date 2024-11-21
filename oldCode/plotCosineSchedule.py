import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Configuration
image_shape = (3, 32, 32)
num_timesteps = 1000
output_file = "cosine_noise_progression.png"

# Load a single CIFAR-10 image
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = torchvision.datasets.CIFAR10("cifar10", download=True, transform=transform)
image, _ = dataset[0]  # Take the first image
image = image.unsqueeze(0)  # Add batch dimension
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image = image.to(device)

# Cosine beta schedule (from 2020 DDPM paper)
def cosine_schedule(t, s=0.008):
    """
    Implements the cosine schedule for alpha_bar as described in the DDPM paper.
    Args:
        t (float): Time step in [0, 1].
        s (float): Small offset for numerical stability.
    Returns:
        torch.Tensor: Alpha_bar at timestep t.
    """
    return torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2

# Generate alpha_bar values
time_points = torch.linspace(0, 1, num_timesteps, device=device)
alpha_bars = cosine_schedule(time_points)

# Generate noisy images
noisy_images = []

for t in range(num_timesteps):
    alpha_bar_t = alpha_bars[t]
    noise = torch.randn_like(image)
    noisy_image = torch.sqrt(alpha_bar_t) * image + torch.sqrt(1 - alpha_bar_t) * noise
    noisy_images.append(noisy_image.squeeze().cpu())

# Select 100 evenly spaced timesteps for visualization
indices = np.linspace(0, num_timesteps - 1, 100, dtype=int)

# Plot and save
fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i, ax in enumerate(axes.flatten()):
    idx = indices[i]
    img = noisy_images[idx].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.savefig(output_file)
print(f"Noise progression saved to {output_file}")
