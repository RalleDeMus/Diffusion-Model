import wandb
import torch
import os
from utils.dataset_loader import load_dataset
from configs.CIFAR10 import config
from UNet.model import UNet
from UNet.trainer import train_model


# Initialize W&B
logwandb = False

# Use only small subset of data (for debugging)
debugDataSize = False

save_model = False
model_name = "UNET" # File that the model is saved as. Only relevant if save_model = True


if logwandb: 
    wandb.init(project=config["project_name"], name="UNET", config=config)

# Load dataset
dataset = load_dataset(config,small_sample=debugDataSize)

channels = config["image_shape"][0]
# Initialize model
model = UNet(in_channels=channels, dim = config["image_shape"][1], out_channels=channels)

# Train model
train_model(model, dataset, config,model_name="CIFAR10_tests",log=logwandb, save_model = save_model)

