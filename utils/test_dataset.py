import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset_loader import load_dataset

# Function to test the dataset loader and save the first 8 images
def test_dataset(config, output_dir="test_output"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(config)
    
    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Get the first batch of images
    images, labels = next(iter(dataloader))
    
    # Denormalize the images to [0, 1] for saving
    images = images * 0.5 + 0.5  # Reverse normalization: [-1, 1] -> [0, 1]
    
    # Save the first 8 images as a row
    save_path = os.path.join(output_dir, "first_8_images.png")
    save_image(images, save_path, nrow=8)
    print(f"Saved the first 8 images to {save_path}")

# Example configuration for CelebA dataset
config = {
    "dataset_name": "CELEBA",
    "data_dir": "cropped_celeba_bin",  # Path to your .bin files
    "image_shape": (3, 128, 128)  # For reference (not directly used here)
}

# Run the test
test_dataset(config)
