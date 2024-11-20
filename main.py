import wandb
import torch
import os
from models.ModelLayers.MNIST import GetEncDecLayers
from configs.MNIST import config
from utils.dataset_loader import load_dataset
from models.attention_model.attention_model import ScoreNetwork
from trainers.trainer import train_model

# Initialize W&B
logwandb = True

# Use only small subset of data (for debugging)
debugDataSize = False

if logwandb: 
    #wandb.init(project=config["project_name"], name=config["wandb_name"], config=config)
    wandb.init(project=config["project_name"], name="Mnist 5 epochs", config=config)

# Load dataset
dataset = load_dataset(config,small_sample=debugDataSize)


# Initialize model
model = ScoreNetwork(in_channels=config["image_shape"][0], layers = GetEncDecLayers())

# Train model
train_model(model, dataset, config, config["image_shape"],log=logwandb, save_model = True, model_name = "MNISTNetwork_500epochs")
