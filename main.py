import wandb
import torch
import os
from models.ModelLayers.CIFAR10 import GetEncDecLayers
from configs.config import config_MNIST, config_CIFAR10, config_CELEBA
from utils.dataset_loader import load_dataset
from models.attention_model.attention_model import ScoreNetwork
from trainers.trainer import train_model

config = config_MNIST

# Initialize W&B
logwandb = True

# Use only small subset of data (for debugging)
debugDataSize = False

save_model = True
model_name = "CIFARNetwork_500epochs" # File that the model is saved as. Only relevant if save_model = True


if logwandb: 
    wandb.init(project=config["project_name"], name="CIFAR 500 epochs", config=config)

# Load dataset
dataset = load_dataset(config,small_sample=debugDataSize)

# Initialize model
model = ScoreNetwork(in_channels=config["image_shape"][0], layers = GetEncDecLayers())

# Train model
train_model(model, dataset, config, config["image_shape"],log=logwandb, save_model = save_model, model_name = model_name)
