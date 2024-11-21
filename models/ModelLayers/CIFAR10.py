import torch.nn as nn
import torch
import torch.nn.functional as F
from models.ModelLayers.SelfAttention import SelfAttention

def GetEncDecLayers():
    chs = [64, 128, 256, 512, 512]
    num_groups = 16  # Number of groups for GroupNorm
    in_channels=3
    dropout = 0.1
    
    encLayer = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels + 1, chs[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, chs[0]),  # Add GroupNorm
            nn.SiLU(),
            SelfAttention(chs[0]),  # Self-attention layer
            nn.Dropout(dropout),
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, chs[1]),  # Add GroupNorm
            nn.SiLU(),
            SelfAttention(chs[1]),  # Self-attention layer
            nn.Dropout(dropout),
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, chs[2]),  # Add GroupNorm
            nn.SiLU(),
            SelfAttention(chs[2]),  # Self-attention layer
            nn.Dropout(dropout),
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, chs[3]),  # Add GroupNorm
            nn.SiLU(),
            SelfAttention(chs[3]),  # Self-attention layer
            nn.Dropout(dropout),
        ),
        nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, chs[4]),  # Add GroupNorm
            nn.SiLU(),
            SelfAttention(chs[4]),  # Self-attention layer
            nn.Dropout(dropout),
        ),
    ])
    
    decLayer = nn.ModuleList([
        nn.Sequential(
            nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(num_groups, chs[3]),  # Add GroupNorm
            nn.SiLU(),
            SelfAttention(chs[3]),  # Self-attention layer
            nn.Dropout(dropout),
        ),
        nn.Sequential(
            nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(num_groups, chs[2]),  # Add GroupNorm
            nn.SiLU(),
            SelfAttention(chs[2]),  # Self-attention layer
            nn.Dropout(dropout),
        ),
        nn.Sequential(
            nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(num_groups, chs[1]),  # Add GroupNorm
            nn.SiLU(),
            SelfAttention(chs[1]),  # Self-attention layer
            nn.Dropout(dropout),
        ),
        nn.Sequential(
            nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(num_groups, chs[0]),  # Add GroupNorm
            nn.SiLU(),
            SelfAttention(chs[0]),  # Self-attention layer
            nn.Dropout(dropout),
        ),
        nn.Sequential(
            nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, chs[0]),  # Add GroupNorm
            nn.SiLU(),
            SelfAttention(chs[0]),  # Self-attention layer
            nn.Conv2d(chs[0], in_channels, kernel_size=3, padding=1),
        ),
    ])
    return encLayer, decLayer

# Decoder layers
# def GetDecoderLayers():
#     chs = [32, 64, 128, 256, 256]
#     num_groups = 32  # Number of groups for GroupNorm
#     in_channels=3
#     decLayer = nn.ModuleList([
#         nn.Sequential(
#             nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.GroupNorm(num_groups, chs[3]),  # Add GroupNorm
#             nn.SiLU(),
#             SelfAttention(chs[3]),  # Self-attention layer
#         ),
#         nn.Sequential(
#             nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.GroupNorm(num_groups, chs[2]),  # Add GroupNorm
#             nn.SiLU(),
#             SelfAttention(chs[2]),  # Self-attention layer
#         ),
#         nn.Sequential(
#             nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.GroupNorm(num_groups, chs[1]),  # Add GroupNorm
#             nn.SiLU(),
#             SelfAttention(chs[1]),  # Self-attention layer
#         ),
#         nn.Sequential(
#             nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.GroupNorm(num_groups, chs[0]),  # Add GroupNorm
#             nn.SiLU(),
#             SelfAttention(chs[0]),  # Self-attention layer
#         ),
#         nn.Sequential(
#             nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
#             nn.GroupNorm(num_groups, chs[0]),  # Add GroupNorm
#             nn.SiLU(),
#             SelfAttention(chs[0]),  # Self-attention layer
#             nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),
#         ),
#     ])