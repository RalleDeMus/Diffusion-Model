import wandb
import torch
import os
from utils.dataset_loader import load_dataset
from configs.config import config_MNIST, config_CIFAR10, config_CELEBA
from UNet.model import UNet
from UNet.trainer import train_model

config = config_MNIST


# Initialize W&B
logwandb = False

# Use only small subset of data (for debugging)
debugDataSize = True
modelNameTest = "_test" if debugDataSize else ""

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
train_model(model, dataset, config,model_name=model_name=config["dataset_name"]+modelNameTest,log=logwandb, save_model = save_model)

