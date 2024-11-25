import torch.nn as nn
import torch
import torch.nn.functional as F


# Taken from https://github.com/dome272/Diffusion-Models-pytorch
class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class ConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),  # Convolution
            nn.GroupNorm(32, out_channels),  # Group normalization
            nn.SiLU(),  # SiLU activation
            nn.Dropout(p=0.1),
        )
        
        # Optional 1x1 convolution for residual connection if dimensions mismatch
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x): 
        return self.conv(x) + self.residual_conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=0.1),
        )
        self.emb = nn.Sequential(
            nn.Linear(time_dim, out_channels),
            nn.SiLU()
        )
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.activation = nn.SiLU()  # Final activation

    def forward(self, x, t):
        x_conv = self.conv(x)  # Add time-step embedding
        
        emb = self.emb(t)
        emb = emb[:, :, None, None].repeat(1, 1, x_conv.shape[-2], x_conv.shape[-1])

        residual = self.residual_conv(x)

        x = x_conv + emb

        x_out = self.activation(x + residual) 
        return x_out  # Apply activation to the summed output



class DownBlock(nn.Module): # 
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.res_block_1 = ResBlock(in_channels, out_channels, time_dim)
        self.downSample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)  # Strided Conv for lerning!
        self.res_block_2 = ResBlock(out_channels, out_channels, time_dim)

        self.emb = nn.Sequential(
            nn.Linear(time_dim, out_channels),
            nn.SiLU()
        )

    def forward(self, x, t): 
        # Put t in emb
        emb = self.emb(t)
        
        x = self.res_block_1(x, t)
        x_down = self.downSample(x)  # Apply max pooling
        x_down = self.res_block_2(x_down, t)  # Pass x and t to ResBlock

        # reshape emb to fit x
        emb = emb[:, :, None, None].repeat(1, 1, x_down.shape[-2], x_down.shape[-1])
        # return addition of the two 
        return x_down + emb

class UpBlock(nn.Module): # 
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.emb = nn.Sequential(
            nn.Linear(time_dim, out_channels),  
            nn.SiLU()  
        )

        self.upSample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2) # added ConvTranspose2d for learnable upscaling!

        # Up block with channel halfing 
        self.res_block_1 = ResBlock(in_channels*2, in_channels, time_dim)  # In channels x2 because concatenation with x skip
        self.res_block_2 =ResBlock(in_channels, out_channels, time_dim)  # actual out channel now
        

    def forward(self, x, x_skip, t): 
        # upsample x
        x_up = self.upSample(x)
        # Concatenate upsampled x and x skip along channel dimension (1)
        x_cat = torch.cat([x_skip, x_up], dim=1)
        # up
        x_up = self.res_block_1(x_cat, t)
        x_up = self.res_block_2(x_up, t)
        # time emb
        emb = self.emb(t)[:, :, None, None].repeat(1, 1, x_up.shape[-2], x_up.shape[-1])
        # return addition of the two 
        return x_up + emb        

