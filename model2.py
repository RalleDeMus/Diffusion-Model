import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, Resize, Normalize
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn

# Configuration for the dataset and training
dataset_name = "CIFAR10"
image_shape = (3, 32, 32)
project_name = f"{dataset_name} diffusion"
batch_size = 128
epochs = 1000
learning_rate = 3e-4

# Initialize W&B
wandb.init(
    project=project_name,
    name="Cifar10 - Self attention, cosine schedule, higher lr",
    config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    }
)

config = wandb.config

# Generate the CIFAR-10 dataset with augmentation
transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
])
dataset = torchvision.datasets.CIFAR10("cifar10", download=True, transform=transforms)
print(f"Image shape: {dataset[0][0].shape}")

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Transformation for Inception input
resize_transform = Compose([
    Resize((299, 299)),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        attn = torch.bmm(query, key)
        attn = F.softmax(attn, dim=-1)
        
        value = self.value(x).view(batch_size, -1, width * height)
        attn_output = torch.bmm(value, attn.permute(0, 2, 1))
        attn_output = attn_output.view(batch_size, C, width, height)
        
        return self.gamma * attn_output + x

# Cosine Noise Schedule
class CosineSchedule:
    def __init__(self, timesteps, scale=20.0):
        self.timesteps = timesteps
        self.scale = scale

    def beta_t(self, t):
        return self.scale * (1 - torch.cos((t * torch.pi / 2)))

# U-Net with Self-Attention and Cosine Schedule
class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        chs = [128, 256, 512, 512]

        # Encoder layers
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels + 1, chs[0], kernel_size=3, padding=1),
                nn.GroupNorm(8, chs[0]),
                nn.SiLU(),
                SelfAttention(chs[0])
            ),
            nn.Sequential(
                nn.Conv2d(chs[0], chs[1], kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, chs[1]),
                nn.SiLU(),
                SelfAttention(chs[1])
            ),
            nn.Sequential(
                nn.Conv2d(chs[1], chs[2], kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, chs[2]),
                nn.SiLU(),
                SelfAttention(chs[2])
            ),
            nn.Sequential(
                nn.Conv2d(chs[2], chs[3], kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, chs[3]),
                nn.SiLU(),
                SelfAttention(chs[3])
            )
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(chs[3], chs[3], kernel_size=3, padding=1),
            nn.GroupNorm(8, chs[3]),
            nn.SiLU(),
            SelfAttention(chs[3])
        )

        # Decoder layers
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, chs[2]),
                nn.SiLU(),
                SelfAttention(chs[2])
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, chs[1]),
                nn.SiLU(),
                SelfAttention(chs[1])
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, chs[0]),
                nn.SiLU(),
                SelfAttention(chs[0])
            )
        ])

        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(chs[0], in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, t):
        # Append time embedding
        t = t.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        x = torch.cat([x, t], dim=1)

        # Encoder
        skips = []
        for idx, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for idx, decoder in enumerate(self.decoders):
            if idx < len(self.decoders):  # Add skip connections for all but the final layer
                x = torch.cat([x, skips[-(idx + 1)]], dim=1)
            x = decoder(x)

        # Final layer
        x = self.final(x)
        return x


# Define the loss calculation function
def calc_loss(unet: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    timesteps = 1000
    cosine_schedule = CosineSchedule(timesteps)

    beta_t = cosine_schedule.beta_t(t).view(-1, 1, 1, 1)
    int_beta = torch.cumsum(beta_t, dim=0)
    alpha_t = torch.exp(-int_beta)

    noise = torch.randn_like(x)
    x_t = x * alpha_t.sqrt() + noise * (1 - alpha_t).sqrt()

    predicted_score = unet(x_t, t)

    target_score = -noise
    loss = F.mse_loss(predicted_score, target_score)
    return loss

# Define sample generation function
def generate_samples(unet: nn.Module, nsamples: int) -> torch.Tensor:
    device = next(unet.parameters()).device
    x_t = torch.randn((nsamples, *image_shape), device=device)
    timesteps = 1000
    time_pts = torch.linspace(1, 0, timesteps, device=device)
    cosine_schedule = CosineSchedule(timesteps)

    for t in time_pts:
        beta_t = cosine_schedule.beta_t(t).view(1, 1, 1, 1)
        noise = torch.randn_like(x_t) if t > 0 else 0
        score = unet(x_t, t.expand(x_t.shape[0], 1)).detach()
        x_t = x_t + beta_t * score + (beta_t.sqrt() * noise)

    return x_t


# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = UNet(in_channels=image_shape[0]).to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=config.learning_rate)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# Training loop
for epoch in range(config.epochs):
    total_loss = 0
    for data, _ in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        t = torch.rand(data.size(0), device=device)
        loss = calc_loss(unet, data, t)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    wandb.log({"loss": avg_loss})

    if epoch % 5 == 0:  # Save samples and calculate FID every 5 epochs
        print(f"Epoch {epoch}, Loss: {avg_loss}")

        # Generate fake samples
        samples = generate_samples(unet, 500).detach()

        

        # Log generated samples
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for ax, img in zip(axes.flatten(), samples[:16]):
            img = img.permute(1, 2, 0).cpu().numpy()
            ax.imshow((img - img.min()) / (img.max() - img.min()))
            ax.axis('off')
        wandb.log({"Generated samples": wandb.Image(fig)})
        plt.close(fig)

wandb.finish()
