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
        chs = [32, 64, 128, 256, 256]
        num_groups = 8  # Number of groups for GroupNorm

        # Encoder, decoder layers
        self._convs,self._tconvs = layers
        
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


def calc_loss(score_network: torch.nn.Module, x: torch.Tensor, alpha_bars: torch.Tensor, betas: torch.Tensor, num_timesteps) -> torch.Tensor:
    t = torch.randint(1, num_timesteps, (x.shape[0],), device=x.device)
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    alpha_bar_prev = alpha_bars[t - 1].view(-1, 1, 1, 1)

    # Compute variance and mean for q(x_t | x_0)
    var_t = betas[t].view(-1, 1, 1, 1)
    mu_t = torch.sqrt(alpha_bar_prev) / torch.sqrt(alpha_bar_t) * x

    # Sample x_t from q(x_t | x_0)
    noise = torch.randn_like(x)
    x_t = mu_t + noise * torch.sqrt(var_t)

    # Compute score
    score = score_network(x_t, t.view(-1, 1).float() / num_timesteps)

    # Gradient of the log probability of the forward process
    grad_log_p = -noise / torch.sqrt(var_t)

    # Loss computation
    loss = (score - grad_log_p) ** 2
    weighted_loss = var_t * loss
    return torch.mean(weighted_loss)




## Debugging
def generate_samples(score_network: torch.nn.Module, nsamples: int, image_shape: tuple, alpha_bars: torch.Tensor, betas: torch.Tensor, num_timesteps) -> torch.Tensor:
    device = next(score_network.parameters()).device

    x_t = torch.randn((nsamples, *image_shape), device=device)
    
    time_pts = torch.linspace(1, 0, num_timesteps, device=device)
    
    # Debug initial sample
    if torch.isnan(x_t).any():
        print("Initial x_t contains NaN")
    else:
        print(f"Initial x_t stats: min={x_t.min()}, max={x_t.max()}, mean={x_t.mean()}")

    for i in range(num_timesteps - 1, 0, -1):
        t = i
        beta_t = betas[t] * 20  # Ensure beta_t scales between 0 and 20
        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev = alpha_bars[t - 1]

        # Debug beta_t, alpha_bar_t, alpha_bar_prev
        if beta_t <= 0:
            print(f"Step {t}: beta_t is non-positive: {beta_t}")
        if alpha_bar_t <= 0 or alpha_bar_prev <= 0:
            print(f"Step {t}: Invalid alpha_bar_t={alpha_bar_t} or alpha_bar_prev={alpha_bar_prev}")

        # Compute variance and mean for reverse process
        var_t = beta_t
        mu_t = torch.sqrt(alpha_bar_prev + 1e-8) / torch.sqrt(alpha_bar_t + 1e-8) * x_t

        # Debug variance and mean
        if torch.isnan(mu_t).any():
            print(f"Step {t}: NaN detected in mu_t")
        else:
            print(f"Step {t}: mu_t stats: min={mu_t.min()}, max={mu_t.max()}, mean={mu_t.mean()}")

        batch_time = torch.full((x_t.shape[0], 1), t, device=device).float() / num_timesteps
        score = score_network(x_t, batch_time).detach()

        # Debug score
        if torch.isnan(score).any():
            print(f"Step {t}: NaN detected in score")
        else:
            print(f"Step {t}: score stats: min={score.min()}, max={score.max()}, mean={score.mean()}")

        # Compute drift and diffusion terms
        drift = mu_t - var_t * score
        diffusion = torch.sqrt(var_t + 1e-8)  # Add epsilon for numerical stability

        # Debug drift and diffusion
        if torch.isnan(drift).any():
            print(f"Step {t}: NaN detected in drift")
        else:
            print(f"Step {t}: drift stats: min={drift.min()}, max={drift.max()}, mean={drift.mean()}")

        if torch.isnan(diffusion).any():
            print(f"Step {t}: NaN detected in diffusion")
        else:
            print(f"Step {t}: diffusion stats: min={diffusion.min()}, max={diffusion.max()}, mean={diffusion.mean()}")

        # Update x_t for the next step
        if t > 1:
            noise = torch.randn_like(x_t)
            if torch.isnan(noise).any():
                print(f"Step {t}: NaN detected in noise")
            x_t += drift + diffusion * noise
        else:
            x_t += drift

        # Debug x_t


# ### Working
# def generate_samples(score_network: torch.nn.Module, nsamples: int, image_shape: tuple, alpha_bars: torch.Tensor, betas: torch.Tensor, num_timesteps) -> torch.Tensor:
#     device = next(score_network.parameters()).device

#     x_t = torch.randn((nsamples, *image_shape), device=device)

#     time_pts = torch.linspace(1, 0, num_timesteps, device=device)

#     for i in range(num_timesteps - 1, 0, -1):
#         t = i
#         beta_t = betas[t] * 20  # Ensure beta_t scales between 0 and 20
#         alpha_bar_t = alpha_bars[t]
#         alpha_bar_prev = alpha_bars[t - 1]

#         # Compute variance and mean for reverse process
#         var_t = beta_t
#         mu_t = torch.sqrt(alpha_bar_prev) / torch.sqrt(alpha_bar_t) * x_t

#         # Reuse x_t to avoid memory overhead
#         batch_time = torch.full((x_t.shape[0], 1), t, device=device).float() / num_timesteps
#         score = score_network(x_t, batch_time).detach()

#         # Compute drift and diffusion terms
#         drift = mu_t - var_t * score
#         diffusion = torch.sqrt(var_t)

#         # Update x_t for the next step
#         if t > 1:
#             noise = torch.randn_like(x_t)
#             x_t += drift + diffusion * noise
#         else:
#             x_t += drift

#     return x_t


def print_memory_usage(tag=""):
    print(f"[{tag}] Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"[{tag}] Reserved Memory: {torch.cuda.memory_reserved() / 1e6} MB")

