import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader
import os

def load_dataset(config, small_sample=False, validation=False):
    validationSize = 0.1
    if config["dataset_name"] == "MNIST":
        transform = transforms.Compose([
            transforms.Pad((2, 2, 2, 2), fill=0),  # Adds 2 pixels to each side with black padding
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ])
        dataset = datasets.MNIST("mnist", download=True, transform=transform)

    elif config["dataset_name"] == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # RGB Normalize to [-1, 1]
        ])
        dataset = datasets.CIFAR10("cifar10", download=True, transform=transform)

    elif config["dataset_name"] == "CELEBA":
        # Define transformation
        transform = transforms.Compose([
            transforms.Resize((config["image_shape"][1], config["image_shape"][2])),  
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])

        # Get all data batch files
        data_batch_dir = "cropped_celeba_bin"
        batch_files = [os.path.join(data_batch_dir, f) for f in os.listdir(data_batch_dir) if f.startswith("data_batch_")]

        # Load all binary files
        dataset = CombinedBinaryDataset(bin_files=batch_files, img_size=(128, 128), num_channels=3, transform=transform)

    else:
        raise ValueError(f"Dataset {config['dataset_name']} is not supported!")
    
    if small_sample:
        # Use only the first 100 samples for quick testing
        dataset = Subset(dataset, range(10))
    
    if validation:
        dataset_size = len(dataset)
        val_size = int(validationSize * dataset_size)
        dataset = Subset(dataset, range(dataset_size - val_size, dataset_size))
        print("Loading validation set")
    else:
        dataset_size = len(dataset)
        train_size = int((1-validationSize) * dataset_size)
        dataset = Subset(dataset, range(0, train_size))
        print("Loading training set")

    return dataset

class CombinedBinaryDataset(torch.utils.data.Dataset):
    def __init__(self, bin_files, img_size, num_channels=3, transform=None):
        self.bin_files = bin_files  # List of binary file paths
        self.img_size = img_size
        self.num_channels = num_channels
        self.samples = []
        self.transform = transform

        # Read all batches into memory
        for bin_file in self.bin_files:
            file_size = os.path.getsize(bin_file)
            sample_size = 1 + num_channels * img_size[0] * img_size[1]  # 1 byte label + pixel data
            num_samples = file_size // sample_size

            #print(f"Loading {num_samples} samples from {bin_file}")

            with open(bin_file, "rb") as f:
                for _ in range(num_samples):
                    raw = f.read(sample_size)
                    self.samples.append(raw)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw = self.samples[idx]
        label = raw[0]  # Dummy label
        pixels = torch.tensor(
            list(raw[1:]), dtype=torch.float32
        ).reshape(self.num_channels, *self.img_size) / 255.0  # Normalize to [0, 1]

        if self.transform:
            # Convert to PIL Image for compatibility with transforms
            pixels = transforms.ToPILImage()(pixels)
            pixels = self.transform(pixels)

        return pixels, label






# # Example configuration for CIFAR10
# config = {
#     "dataset_name": "CELEBA",
#     "image_shape": (3, 64, 64),  # Channels, Height, Width
# }

# # Load the dataset (set validation=False for training data or True for validation data)
# dataset = load_dataset(config, small_sample=False, validation=False)

# # Create a DataLoader to iterate through the dataset
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Get the length of the dataset
# dataset_length = len(data_loader.dataset)

# print(f"Number of samples in the dataset: {dataset_length}")




# file_path = "cropped_celeba_bin/data_batch_1"

# # Get the size of the file in bytes
# file_size = os.path.getsize(file_path)

# # Print the file size
# print(f"Size of the file '{file_path}': {file_size} bytes")

