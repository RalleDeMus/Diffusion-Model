import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Configuration
image_shape = (3, 32, 32)
num_timesteps = 1000
output_file = "linear_noise_progression.png"

# Load a single CIFAR-10 image
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = torchvision.datasets.CIFAR10("cifar10", download=True, transform=transform)
image, _ = dataset[0]  # Take the first image
image = image.unsqueeze(0)  # Add batch dimension
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image = image.to(device)

# Linear beta schedule
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

# Calculate alpha and alpha_bar
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

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