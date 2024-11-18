import torch
import torch.nn.functional as F
import torchvision
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn

# Configuration for the dataset and training
dataset_name = "CIFAR10"
image_shape = (3, 32, 32)
project_name = f"{dataset_name} diffusion"
batch_size = 128
epochs = 1000
learning_rate = 1e-4


# Initialize W&B
wandb.init(
    project=project_name,
    name="Cifar10 - Self attention, more layers, lower lr",
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

# ScoreNetwork Class with increased capacity, dropout, and LeakyReLU
class ScoreNetwork(nn.Module):
    def __init__(self, in_channels=3):  # Set to 3 for CIFAR-10 RGB images
        super().__init__()
        chs = [64, 128, 256, 512, 512]  # Increased network capacity

        # Encoder layers
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels + 1, chs[0], kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[0]),
                nn.Dropout(0.2),  # Added dropout
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[1]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[2]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[3]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[4]),
                nn.Dropout(0.2),
            ),
        ])

        # Decoder layers
        self._tconvs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size (512, 4, 4)
                nn.LeakyReLU(0.2),
                SelfAttention(chs[3]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size (256, 8, 8)
                nn.LeakyReLU(0.2),
                SelfAttention(chs[2]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size (128, 16, 16)
                nn.LeakyReLU(0.2),
                SelfAttention(chs[1]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size (64, 32, 32)
                nn.LeakyReLU(0.2),
                SelfAttention(chs[0]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[0]),
                nn.Conv2d(chs[0], 3, kernel_size=3, padding=1),  # Output 3 channels for RGB
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Expand t to match the batch and spatial dimensions of x (batch_size, 1, H, W)
        tt = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])
        
        # Concatenate x and t along the channel dimension
        x = torch.cat((x, tt), dim=1)

        # Encoder with skip connections
        enc_signals = []
        for i, conv in enumerate(self._convs):
            x = conv(x)
            if i < len(self._convs) - 1:  # Store for skip connections, except for the last layer
                enc_signals.append(x)

        # Decoder with concatenated skip connections
        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                x = tconv(x)
            else:
                x = torch.cat((x, enc_signals[-i]), dim=1)  # Concatenate encoder output along channel dimension
                x = tconv(x)

        return x

# Initialize the model
score_network = ScoreNetwork(in_channels=image_shape[0])

# Loss calculation function
def calc_loss(score_network: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device) * (1 - 1e-4) + 1e-4
    int_beta = (0.1 + 0.5 * (20 - 0.1) * t) * t
    int_beta = int_beta.view(-1, 1, 1, 1).expand_as(x)
    mu_t = x * torch.exp(-0.5 * int_beta)
    var_t = -torch.expm1(-int_beta)
    x_t = torch.randn_like(x) * var_t ** 0.5 + mu_t
    grad_log_p = -(x_t - mu_t) / var_t
    score = score_network(x_t, t)
    loss = (score - grad_log_p) ** 2
    lmbda_t = var_t
    weighted_loss = lmbda_t * loss
    return torch.mean(weighted_loss)


# Define sample generation function
def generate_samples(score_network: torch.nn.Module, nsamples: int) -> torch.Tensor:
    device = next(score_network.parameters()).device
    x_t = torch.randn((nsamples, *image_shape), device=device)
    time_pts = torch.linspace(1, 0, 1000, device=device)
    beta = lambda t: 0.1 + (20 - 0.1) * t
    for i in range(len(time_pts) - 1):
        t = time_pts[i]
        dt = time_pts[i + 1] - t
        fxt = -0.5 * beta(t) * x_t
        gt = beta(t) ** 0.5
        score = score_network(x_t, t.expand(x_t.shape[0], 1)).detach()
        drift = fxt - gt * gt * score
        diffusion = gt
        x_t = x_t + drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
    return x_t

# Training setup with repeated dataset usage
opt = torch.optim.Adam(score_network.parameters(), lr=config.learning_rate)
dloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
score_network = score_network.to(device)

# Training loop with 1000 epochs
for i_epoch in range(config.epochs):
    total_loss = 0
    for data, _ in dloader:
        data = data.to(device)
        opt.zero_grad()
        loss = calc_loss(score_network, data)
        loss.backward()
        opt.step()
        total_loss += loss.item() * data.shape[0]

    avg_loss = total_loss / len(dataset)
    wandb.log({"loss": avg_loss})
    
    if i_epoch % 5 == 0:  # Save samples every 10 epochs
        print(f"Epoch {i_epoch}, Loss: {avg_loss}")
        samples = generate_samples(score_network, 16).detach().reshape(-1, *image_shape)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for ax, img in zip(axes.flatten(), samples):
            img = img.permute(1, 2, 0).cpu().numpy()
            ax.imshow((img - img.min()) / (img.max() - img.min()))
            ax.axis('off')
        wandb.log({"Generated samples": wandb.Image(fig)})
        plt.close(fig)

wandb.finish()
