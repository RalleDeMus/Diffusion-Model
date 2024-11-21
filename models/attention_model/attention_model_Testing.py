import torch.nn as nn
import torch
import torch.nn.functional as F


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

# ScoreNetwork Class
class ScoreNetwork(nn.Module):
    def __init__(self, layers, in_channels):
        super().__init__()
        # # Encoder layers and decoder layers
        # self._convs, self._tconvs = layers

        super().__init__()
        chs = [64, 128, 256, 512, 512]  # Increased network capacity

        # Encoder layers
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels + 1, chs[0], kernel_size=3, padding=1),
                nn.SiLU(),
                SelfAttention(chs[0]),
                nn.Dropout(0.1),  # Added dropout
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),
                nn.SiLU(),
                SelfAttention(chs[1]),
                nn.Dropout(0.1),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),
                nn.SiLU(),
                SelfAttention(chs[2]),
                nn.Dropout(0.1),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),
                nn.SiLU(),
                SelfAttention(chs[3]),
                nn.Dropout(0.1),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),
                nn.SiLU(),
                SelfAttention(chs[4]),
                nn.Dropout(0.1),
            ),
        ])

        # Decoder layers
        self._tconvs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size (512, 4, 4)
                nn.SiLU(),
                SelfAttention(chs[3]),
                nn.Dropout(0.1),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size (256, 8, 8)
                nn.SiLU(),
                SelfAttention(chs[2]),
                nn.Dropout(0.1),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size (128, 16, 16)
                nn.SiLU(),
                SelfAttention(chs[1]),
                nn.Dropout(0.1),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size (64, 32, 32)
                nn.SiLU(),
                SelfAttention(chs[0]),
                nn.Dropout(0.1),
            ),
            nn.Sequential(
                nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
                nn.SiLU(),
                SelfAttention(chs[0]),
                nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # Output 3 channels for RGB
            ),
        ])

        

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        log = False
        # Assume x is already standardized to (batch_size, channels, height, width)
        if log:
            print(f"Input x shape: {x.shape}")
            print(f"Input t shape: {t.shape}")

        # Expand t to match the batch and spatial dimensions of x (batch_size, 1, H, W)
        tt = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])
        if log:
            print(f"Expanded t shape: {tt.shape}")

        # Concatenate x and t along the channel dimension
        x = torch.cat((x, tt), dim=1)
        if log:
            print(f"Concatenated x shape: {x.shape}")

        # Encoder with skip connections
        enc_signals = []
        for i, conv in enumerate(self._convs):
            x = conv(x)
            if log:
                print(f"After encoder layer {i}, x shape: {x.shape}")
            if i < len(self._convs) - 1:  # Store for skip connections, except for the last layer
                enc_signals.append(x)
                if log:
                    print(f"Stored skip connection {i} shape: {x.shape}")

        # Decoder with concatenated skip connections
        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                x = tconv(x)
                if log:
                    print(f"After decoder layer {i}, x shape: {x.shape}")
            else:
                x = torch.cat((x, enc_signals[-i]), dim=1)  # Concatenate encoder output along channel dimension
                if log:
                    print(f"After concatenation with skip connection {len(enc_signals) - i}, x shape: {x.shape}")
                x = tconv(x)
                if log:
                    print(f"After decoder layer {i}, x shape: {x.shape}")

        return x

# Define a customizable noising schedule
def noising_schedule(t: torch.Tensor, schedule_type: str = "linear") -> torch.Tensor:
    if schedule_type == "linear":
        return 0.1 + (20 - 0.1) * t
    elif schedule_type == "cosine":
        s = 0.008
        f = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
        #f = torch.clamp(f, min=1e-6, max=1 - 1e-6)  # Clamp to avoid division by zero
        output = (1 - f) / f
        if torch.isnan(output).any():
            print(f"output nan, f nan: {torch.isnan(f).any()}, f zero: {torch.eq(f, 0).any()}")
        return output
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


def linear_beta_schedule(timesteps):
    beta_start = 0.0001  # Small noise variance at start
    beta_end = 0.02  # Larger noise variance at end
    return torch.linspace(beta_start, beta_end, timesteps)

def compute_alpha_and_alpha_bar(betas):
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar

def calc_loss(score_network, x, timesteps=1000):
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
    predicted_noise = score_network(x_t, t / timesteps)
    
    # Compute the loss as the difference between predicted and actual noise
    loss = torch.mean((predicted_noise - noise) ** 2)
    return loss

def generate_samples(score_network, nsamples, image_shape, timesteps=1000):
    # Define the linear beta schedule
    betas = linear_beta_schedule(timesteps)
    alphas, alpha_bar = compute_alpha_and_alpha_bar(betas)

    device = next(score_network.parameters()).device
    x_t = torch.randn((nsamples, *image_shape), device=device)  # Start from pure noise
    
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((x_t.size(0),), t, device=device).long()  # Current time step
        
        # Compute the variance terms
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bar[t]
        alpha_bar_prev = alpha_bar[t - 1] if t > 0 else 1.0
        
        # Predict the noise
        predicted_noise = score_network(x_t, t_tensor / timesteps).detach()
        
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





def print_memory_usage(tag=""):
    print(f"[{tag}] Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"[{tag}] Reserved Memory: {torch.cuda.memory_reserved() / 1e6} MB")

