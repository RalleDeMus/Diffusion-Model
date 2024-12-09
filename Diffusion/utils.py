import os
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(64, 64))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)




def get_data(args):
    # Define transformations for CIFAR dataset
    transform = transforms.Compose([
        #torchvision.transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(
        root='./cifar10', 
        train=True, 
        transform=transform, 
        download=True
    )

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader



def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)