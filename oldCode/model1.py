import torch
import torchvision
import wandb
import matplotlib.pyplot as plt

# Initialize W&B
wandb.init(
    project="MNIST diffusion, tester",
    name="5 epoch run",
    config={
        "learning_rate": 3e-4,
        "epochs": 5,
        "batch_size": 64,
    }
)
config = wandb.config

# Generate the MNIST dataset
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
mnist_dset = torchvision.datasets.MNIST("mnist", download=True, transform=transforms)
print(mnist_dset[0][0].shape)

# Model definition
class ScoreNetwork0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        nch = 2
        chs = [32, 64, 128, 256, 256]
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(2, chs[0], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
            ),
        ])
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),
                torch.nn.LogSigmoid(),
                torch.nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))
        tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)
        x2t = torch.cat((x2, tt), dim=-3)
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))
        return signal

score_network = ScoreNetwork0()

# Loss calculation function
def calc_loss(score_network: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device) * (1 - 1e-4) + 1e-4
    int_beta = (0.1 + 0.5 * (20 - 0.1) * t) * t
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
    x_t = torch.randn((nsamples, 28 * 28), device=device)
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


# Training setup
opt = torch.optim.Adam(score_network.parameters(), lr=config.learning_rate)
dloader = torch.utils.data.DataLoader(mnist_dset, batch_size=config.batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
score_network = score_network.to(device)

# Training loop
for i_epoch in range(config.epochs):
    total_loss = 0
    for data, _ in dloader:
        data = data.reshape(data.shape[0], -1).to(device)
        opt.zero_grad()
        loss = calc_loss(score_network, data)
        loss.backward()
        opt.step()
        total_loss += loss.item() * data.shape[0]

    # Log loss to W&B
    avg_loss = total_loss / len(mnist_dset)
    wandb.log({"loss": avg_loss})
    
    # Generate and log samples at intervals
    if i_epoch % 1 == 0:
        print(f"Epoch {i_epoch}, Loss: {avg_loss}")
        samples = generate_samples(score_network, 16).detach().reshape(-1, 28, 28)
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for ax, img in zip(axes.flatten(), samples):
            ax.imshow(1 - img.cpu().numpy(), cmap="Greys")
            ax.axis('off')
        wandb.log({"Generated samples": wandb.Image(fig)})
        plt.close(fig)

wandb.finish()
