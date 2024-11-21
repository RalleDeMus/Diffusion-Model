import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader

def load_dataset(config, small_sample=False):
    if config["dataset_name"] == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST("mnist", download=True, transform=transform)

    elif config["dataset_name"] == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR10("cifar10", download=True, transform=transform)

    else:
        raise ValueError(f"Dataset {config['dataset_name']} is not supported!")
    
    if small_sample:
        # Use only the first 10 samples for quick testing
        dataset = Subset(dataset, range(5000))
    
    return dataset

