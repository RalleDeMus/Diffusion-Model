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
    name="Cifar10 - Self attention, cosine?",
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

# Score Network
class ScoreNetwork(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        chs = [64, 128, 256, 512, 512]

        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels + 1, chs[0], kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[0]),
                nn.Dropout(0.2),
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

        self._tconvs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[3]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[2]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[1]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[0]),
                nn.Dropout(0.2),
            ),
            nn.Sequential(
                nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                SelfAttention(chs[0]),
                nn.Conv2d(chs[0], 3, kernel_size=3, padding=1),
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tt = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])
        x = torch.cat((x, tt), dim=1)
        enc_signals = []
        for i, conv in enumerate(self._convs):
            x = conv(x)
            if i < len(self._convs) - 1:
                enc_signals.append(x)
        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                x = tconv(x)
            else:
                x = torch.cat((x, enc_signals[-i]), dim=1)
                x = tconv(x)
        return x

# Cosine schedule
def cosine_alpha_bar(t, s=0.008):
    return torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2

def compute_betas(num_timesteps=1000, s=0.008):
    timesteps = torch.linspace(0, 1, num_timesteps)
    alpha_bars = cosine_alpha_bar(timesteps, s)
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    betas = torch.cat([torch.tensor([betas[0]]), betas])
    return alpha_bars, betas

# Precompute alpha_bar and beta values
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_timesteps = 1000
timesteps = torch.linspace(0, 1, num_timesteps, device=device)

# Precompute alpha_bar and beta values on the correct device
alpha_bars, betas = compute_betas(num_timesteps)
alpha_bars = alpha_bars.to(device)
betas = betas.to(device)

# Precompute square roots for efficiency
sqrt_alpha_bars = torch.sqrt(alpha_bars).to(device)
sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars).to(device)




# Loss function
def calc_loss(score_network, x):
    batch_size = x.shape[0]
    t = torch.randint(0, num_timesteps, (batch_size,), device=x.device).long()
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    beta_t = betas[t].view(-1, 1, 1, 1)
    noise = torch.randn_like(x)
    x_t = sqrt_alpha_bars[t].view(-1, 1, 1, 1) * x + sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1) * noise
    predicted_score = score_network(x_t, t.float() / num_timesteps)
    loss = (predicted_score + noise / sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)) ** 2
    return loss.mean()

# Sampling
def generate_samples(score_network, nsamples):
    x_t = torch.randn((nsamples, *image_shape), device=device)
    for t in reversed(range(1, num_timesteps)):
        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev = alpha_bars[t - 1] if t > 0 else 0
        predicted_score = score_network(x_t, torch.full((nsamples,), t, device=device).float() / num_timesteps)
        x_t = torch.sqrt(alpha_bar_prev / alpha_bar_t) * x_t - torch.sqrt(1 - alpha_bar_prev) * predicted_score
        if t > 1:
            x_t += torch.sqrt(betas[t]) * torch.randn_like(x_t)
    return x_t


# Training setup with repeated dataset usage
score_network = ScoreNetwork(in_channels=image_shape[0]).to(device)
opt = torch.optim.Adam(score_network.parameters(), lr=config.learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(device)
# Define debug flag
debug = True

# Adjust dataset and DataLoader for debug mode
if debug:
    small_dataset = torch.utils.data.Subset(dataset, range(5))  # Use only 5 images
    dloader = torch.utils.data.DataLoader(small_dataset, batch_size=config.batch_size, shuffle=True)
else:
    dloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# Training loop with debug option
for i_epoch in range(config.epochs):
    total_loss = 0
    for data, _ in dloader:
        data = data.to(device)
        opt.zero_grad()
        loss = calc_loss(score_network, data)
        loss.backward()
        opt.step()
        total_loss += loss.item() * data.shape[0]

    avg_loss = total_loss / (len(small_dataset) if debug else len(dataset))
    wandb.log({"loss": avg_loss})
    
    if i_epoch % 5 == 0:  # Save samples every 5 epochs
        print(f"Epoch {i_epoch}, Loss: {avg_loss}")
        samples = generate_samples(score_network, 16).detach().reshape(-1, *image_shape)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for ax, img in zip(axes.flatten(), samples):
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
            ax.imshow(img)
            ax.axis('off')
        wandb.log({"Generated samples": wandb.Image(fig)})
        plt.close(fig)

wandb.finish()
