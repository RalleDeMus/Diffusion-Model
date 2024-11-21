import wandb
import torch
import os
from models.ModelLayers.CIFAR10 import GetEncDecLayers
from configs.CIFAR10 import config
from utils.dataset_loader import load_dataset
from models.attention_model.attention_model import ScoreNetwork
from trainers.trainer import train_model

# Initialize W&B
logwandb = False

# Use only small subset of data (for debugging)
debugDataSize = False

save_model = False
model_name = "MNISTNetwork_500epochs2" # File that the model is saved as. Only relevant if save_model = True


if logwandb: 
    wandb.init(project=config["project_name"], name="Mnist 500 epochs", config=config)

# Load dataset
dataset = load_dataset(config,small_sample=debugDataSize)

# Initialize model
model = ScoreNetwork(in_channels=config["image_shape"][0], layers = GetEncDecLayers())

# Train model
train_model(model, dataset, config, config["image_shape"],log=logwandb, save_model = save_model, model_name = model_name)
