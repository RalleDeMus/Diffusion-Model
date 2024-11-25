# from datasets import load_dataset

# dataset = load_dataset("pcuenq/lsun-bedrooms", split="train")
# print(dataset[0]["image"].size)  # PIL.Image object, size should be 128x128


from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# Custom Dataset Wrapper
class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx]["image"]  # PIL Image
        if self.transform:
            image = self.transform(image)
        return image

# Function to get the DataLoader
def get_data(args):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(128),  # Resize the smaller dimension to 128 (maintains aspect ratio)
        transforms.CenterCrop((128, 128)),  # Crop the center of the resized image
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load the dataset from Hugging Face
    hf_dataset = load_dataset("Artificio/WikiArt", "default", split="train")
    # Wrap it with the custom Dataset class
    dataset = HuggingFaceDataset(hf_dataset, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

# Example usage
class Args:
    batch_size = 32  # Set the batch size

args = Args()
dataloader = get_data(args)



# Iterate over the DataLoader
for batch in dataloader:
    print(batch.shape)  # Each batch will have shape [batch_size, 3, 128, 128]
    break

import torchvision.utils as vutils

# Iterate over the DataLoader
for i, batch in enumerate(dataloader):
    # De-normalize the images to bring them back to [0, 1]
    batch = batch * 0.5 + 0.5  # Undo the normalization applied earlier

    # Select the first 8 images from the batch
    grid_images = batch[:64]  # Take the first 8 images

    # Create a 4x2 grid
    grid = vutils.make_grid(grid_images, nrow=8, padding=2)  # nrow=4 makes a 4x2 grid

    # Convert the grid to a PIL image
    grid_image = transforms.ToPILImage()(grid)

    # Save the grid as a single .jpg file
    grid_image.save("images/grid.jpg")

    break  # Process only the first batch