import os
import torch
import numpy as np
from dataset_loader import CustomBinaryDataset

def preprocess_and_save(dataset, save_path):
    os.makedirs(save_path, exist_ok=True)
    data, labels = [], []

    for img, label in dataset:
        data.append(np.array(img))  # Convert image to NumPy array
        labels.append(label)

    data = np.stack(data)
    labels = np.array(labels)

    torch.save((data, labels), os.path.join(save_path, "celeba_single_batch.pt"))
    print(f"Preprocessed single batch saved at {save_path}/celeba_single_batch.pt")


# Path to the single binary file
single_file = "cropped_celeba_bin/data_batch_1"

# Define the dataset for a single file
my_celeba_dataset = CustomBinaryDataset(bin_file=single_file, img_size=(32, 32), num_channels=3, transform=None)
print(f"len of celeba: {len(my_celeba_dataset)}")
# Preprocess and save
preprocess_and_save(my_celeba_dataset, "celeba_preprocessed_single_batch")
