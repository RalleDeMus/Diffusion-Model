import torch.nn as nn
import torch
import torch.nn.functional as F
from UNetResBlock.blocks import ResBlock, DownBlock, UpBlock, SelfAttention, TimeEmbedding
from UNetResBlock.alphabeta import compute_linear_beta_schedule, compute_alpha_schedule, compute_alpha_cumulative_product
import math
import os
from PIL import Image
import torchvision.utils as vutils

class UNet(nn.Module):
    def __init__(self, dim=32, in_channels=3, out_channels=3, time_dim=256, device="cuda"):
        super(UNet, self).__init__()

        self.device = device

        self.time_embedding_dim = 128  # Or some other dimension
        self.time_projection_dim = 512
        self.time_embedder = TimeEmbedding(self.time_embedding_dim, self.time_projection_dim)

        # Encoder (Downsampling path)
        self.inc = ResBlock(in_channels, 64, self.time_projection_dim)  # (b, 64, 32, 32)
        self.down1 = DownBlock(64, 128, self.time_projection_dim)       # (b, 128, 16, 16)
        self.attn1 = SelfAttention(128, dim // 2)
        self.down2 = DownBlock(128, 256, self.time_projection_dim)      # (b, 256, 8, 8)
        self.attn2 = SelfAttention(256, dim // 4)
        self.down3 = DownBlock(256, 512, self.time_projection_dim)      # (b, 512, 4, 4)
        self.attn3 = SelfAttention(512, dim // 8)
        self.down4 = DownBlock(512, 1024, self.time_projection_dim)     # (b, 1024, 2, 2)

        # Bottleneck
        self.bot1 = ResBlock(1024, 1024, self.time_projection_dim)      # (b, 1024, 2, 2)
        self.bot2 = ResBlock(1024, 1024, self.time_projection_dim)      # (b, 1024, 2, 2)
        self.attn_bot = SelfAttention(1024, dim // 16)
        self.bot3 = ResBlock(1024, 512, self.time_projection_dim)       # (b, 512, 2, 2)

        # Decoder (Upsampling path)
        self.up1 = UpBlock(512, 256, self.time_projection_dim)          # (b, 256, 4, 4)
        self.attn4 = SelfAttention(256, dim // 8)
        self.up2 = UpBlock(256, 128, self.time_projection_dim)          # (b, 128, 8, 8)
        self.attn5 = SelfAttention(128, dim // 4)
        self.up3 = UpBlock(128, 64, self.time_projection_dim)           # (b, 64, 16, 16)
        self.attn6 = SelfAttention(64, dim // 2)
        self.up4 = UpBlock(64, 64, self.time_projection_dim)            # (b, 64, 32, 32)

        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)          # Final output (b, c_out, 32, 32)

    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Sinusoidal embeddings for discrete timesteps.
        """
        assert len(timesteps.shape) == 1, "Timesteps should be a 1D tensor"
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))  # Zero pad to match dimensions
        return emb
        
    def forward(self, x, t):
        t_emb = self.get_timestep_embedding(t, self.time_embedding_dim)
        t_emb = self.time_embedder(t_emb)  # Project embedding

        # Encoder
        x1 = self.inc(x, t_emb)            # (b, 64, 32, 32)
        x2 = self.down1(x1, t_emb)         # (b, 128, 16, 16)
        x2 = self.attn1(x2)
        x3 = self.down2(x2, t_emb)         # (b, 256, 8, 8)
        x3 = self.attn2(x3)
        x4 = self.down3(x3, t_emb)         # (b, 512, 4, 4)
        x4 = self.attn3(x4)
        x5 = self.down4(x4, t_emb)         # (b, 1024, 2, 2)
        
        # Bottleneck
        x = self.bot1(x5, t_emb)           # (b, 1024, 2, 2)
        x = self.bot2(x, t_emb)           # (b, 1024, 2, 2)
        x = self.attn_bot(x)
        x = self.bot3(x, t_emb)           # (b, 512, 2, 2)

        # Decoder path (Upsampling)
        x = self.up1(x, x4, t_emb)         # (b, 256, 4, 4)
        x = self.attn4(x)
        x = self.up2(x, x3, t_emb)         # (b, 128, 8, 8)
        x = self.attn5(x)
        x = self.up3(x, x2, t_emb)         # (b, 64, 16, 16)
        x = self.attn6(x)
        x = self.up4(x, x1, t_emb)         # (b, 64, 32, 32)

        # Final output
        output = self.outc(x)              # (b, c_out, 32, 32)
        return output



def linear_beta_schedule(timesteps):
    beta_start = 0.0001  # Small noise variance at start
    beta_end = 0.02  # Larger noise variance at end
    return torch.linspace(beta_start, beta_end, timesteps)

def compute_alpha_and_alpha_bar(betas):
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar

def calc_loss(u_net, x, timesteps=1000):
    # Define the linear beta schedule
    betas = linear_beta_schedule(timesteps)
    _, alpha_bar = compute_alpha_and_alpha_bar(betas)
    
    # Sample random time steps
    t = torch.randint(0, timesteps, (x.size(0),), device=x.device).long()
    
    # Get alpha_bar_t values for chosen time steps
    alpha_bar_t = alpha_bar.to(x.device)[t].view(-1, 1, 1, 1)    
    
    # Add noise according to the forward process
    noise = torch.randn_like(x)
    x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
    
    # Predict the noise using the score network
    predicted_noise = u_net(x_t, t/timesteps)
    
    # Compute the loss as the difference between predicted and actual noise
    loss = torch.mean((predicted_noise - noise) ** 2)
    return loss

def generate_samples(u_net, nsamples, image_shape, timesteps=1000):
    # Define the linear beta schedule
    betas = linear_beta_schedule(timesteps)
    alphas, alpha_bar = compute_alpha_and_alpha_bar(betas)

    device = next(u_net.parameters()).device
    x_t = torch.randn((nsamples, *image_shape), device=device)  # Start from pure noise
    
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((x_t.size(0),), t, device=device).long()  # Current time step
        
        # Compute the variance terms
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]
        alpha_bar_prev = alpha_bar[t - 1] if t > 0 else 1.0
        
        # Predict the noise
        predicted_noise = u_net(x_t, t_tensor/timesteps).detach()
        
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
    
    return x_t


def save_intermediate_images(images, timestep, save_dir = "UNet/GenInt"):
    """
    Save a batch of intermediate images at a specific timestep.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    grid = vutils.make_grid(images, nrow=4, normalize=True, scale_each=True)
    grid = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')  # Convert to uint8
    img = Image.fromarray(grid)
    img.save(os.path.join(save_dir, f"timestep_{timestep:04d}.png"))




def print_memory_usage(tag=""):
    print(f"[{tag}] Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"[{tag}] Reserved Memory: {torch.cuda.memory_reserved() / 1e6} MB")

