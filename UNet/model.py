import torch.nn as nn
import torch
import torch.nn.functional as F
from UNet.blocks import ConvBlock, DownBlock, UpBlock, SelfAttention
from UNet.alphabeta import compute_linear_beta_schedule, compute_alpha_schedule, compute_alpha_cumulative_product

import os
from PIL import Image
import torchvision.utils as vutils

class UNet(nn.Module):
    def __init__(self, dim = 32, in_channels=3, out_channels=3, time_dim=256, device="cuda"):
        super(UNet, self).__init__()

        # Explanaition:
            # in channels from rgb
            # time dimension: number of channels in pos encoding. 
            # input dimensions: (batch size, in channels, h, w)
            # Example: (128,3,32,32)
            # channels: 64, 128, 256, 512, 512

            # A convolution to go from in channels -> 64, (128,64,32,32)
            
            # Down:
                # MaxPooling (128,64,16,16)
                # Conv with channel change: (128,128,16,16)
                # GN+SiLU
                # Time embedding: Linear to go from emb dimensions to out dimensions. 
                    # t is in (batch size, embeddings (now in out dimensions after linear))
                    # now we reshape to match batch size, channels, h, w) and add to x!
                # SiLU
            # self attention
            # repeat until (128,256,4,4)
            
            # Bottle neck
            # Conv with channel change: (128,512,4,4)
            # GN+SiLU
            # Conv with channel change: (128,512,4,4)
            # GN+SiLU
            # Conv with channel change: (128,256,4,4)
            # GN+SiLU

            # Up:
                # Upsampling to go to (128,256,8,8)
                # Taking the 1st last down layer and concatanete it with this to get: (128,512,8,8)
                # Conv to go to (128,256,8,8)
                # GN+SiLU
                # Conv to go to (128,128,8,8)
                # GN+SiLU
                # Time embedding
                # SiLU
            # Self attention
            # repeat, demonstrated for purposes:
                # Upsampling to go to (128,128,16,16)
                # Taking the 2nd last down layer and concatanete it with this to get: (128,256,16,16)
                # Conv to go to (128,128,16,16)
                # GN+SiLU
                # Conv to go to (128,64,16,16)
                # GN+SiLU
                # Time embedding
                # SiLU
            # Self attention
            # repeat, demonstrated for purposes:
                # Upsampling to go to (128,64,32,32)
                # Taking the 3th last down layer and concatanete it with this to get: (128,128,32,32)
                # Conv to go to (128,64,32,32)
                # GN+SiLU
                # Conv to go to (128,64,32,32) # no downsampling here
                # GN+SiLU
                # Time embedding
                # SiLU
            # Conv to go (128,3,32,32)
 
        self.device = device
        self.time_dim = time_dim

        # Encoder (Downsampling path)
        self.inc = ConvBlock(in_channels, 64, time_dim)  # (b, 64, 32, 32)
        self.down1 = DownBlock(64, 128, time_dim)         # (b, 128, 16, 16)
        self.attn1 = SelfAttention(128, dim // 2)  # Self-attention after first downblock
        self.down2 = DownBlock(128, 256, time_dim)        # (b, 256, 8, 8)
        self.attn2 = SelfAttention(256, dim // 4)  # Self-attention after second downblock
        self.down3 = DownBlock(256, 256, time_dim)        # (b, 256, 4, 4)
        self.attn3 = SelfAttention(256, dim // 8)  # Self-attention after third downblock

        # Bottleneck
        self.bot1 = ConvBlock(256, 512, time_dim)          # (b, 512, 4, 4)
        self.bot2 = ConvBlock(512, 512, time_dim)          # (b, 512, 4, 4)
        self.bot3 = ConvBlock(512, 256, time_dim)          # (b, 256, 4, 4)

        # Decoder (Upsampling path)
        self.up1 = UpBlock(256, 128, time_dim)             # (b, 128, 8, 8)
        self.attn4 = SelfAttention(128, dim // 4)  # Self-attention after first upblock
        self.up2 = UpBlock(128, 64, time_dim)              # (b, 64, 16, 16)
        self.attn5 = SelfAttention(64, dim // 2)  # Self-attention after second upblock
        self.up3 = UpBlock(64, 64, time_dim)              # (b, 64, 32, 32)

        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)  # Final output (b, c_out, 32, 32)


    def pos_encoding(self, t):
        """Generate sinusoidal time embeddings."""
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.time_dim // 2, device=t.device).float() / self.time_dim)
        )
        pos_enc_a = torch.sin(t * inv_freq)  # No need for `t[:, None]` here, `t` already has the right shape
        pos_enc_b = torch.cos(t * inv_freq)  # Same for `pos_enc_b`
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t):
        # Generate positional encoding for time `t`
        t = t.unsqueeze(-1).type(torch.float).to(self.device)  # Add a new dimension to t
        t = self.pos_encoding(t)  # (batch_size, time_dim)

        # Encoder
        x1 = self.inc(x)        # (b, 64, 32, 32)
        x2 = self.down1(x1, t)     # (b, 128, 16, 16)
        print(f"down1: {x2.shape}")
        x2 = self.attn1(x2)        # Apply self-attention
        print(f"attn1: {x2.shape}")
        x3 = self.down2(x2, t)     # (b, 256, 8, 8)
        x3 = self.attn2(x3)        # Apply self-attention
        x4 = self.down3(x3, t)     # (b, 256, 4, 4)
        x4 = self.attn3(x4)        # Apply self-attention
        
        # Bottleneck
        x4 = self.bot1(x4)      # (b, 512, 4, 4)
        x4 = self.bot2(x4)      # (b, 512, 4, 4)
        x4 = self.bot3(x4)      # (b, 256, 4, 4)
        
        # Decoder path (Upsampling)
        x = self.up1(x4, x3, t)    # (b, 128, 8, 8)
        x = self.attn4(x)           # Apply self-attention
        x = self.up2(x, x2, t)     # (b, 64, 16, 16)
        x = self.attn5(x)           # Apply self-attention
        x = self.up3(x, x1, t)     # (b, 64, 32, 32)

        # Final output
        output = self.outc(x)      # (b, c_out, 32, 32)
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
    predicted_noise = u_net(x_t, t / timesteps)
    
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
