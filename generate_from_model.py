import os
import torch
import matplotlib.pyplot as plt
from configs.MNIST import config
from models.ModelLayers.MNIST import GetEncDecLayers
from models.attention_model.attention_model import generate_samples, calc_loss
from models.attention_model.attention_model import ScoreNetwork


# Load the saved model
model_folder = "savedModels"
model_filename = "MNISTNetwork_500epochs.pt"
model_path = os.path.join(model_folder, model_filename)


# Replace these with your dataset-specific parameters
image_shape = (1, 28, 28)  # Example shape (channels, height, width)
# alpha_bars, betas = compute_alphas_and_betas()  # Assuming you have this function
# num_timesteps = 1000

# Instantiate and load the model
score_network = ScoreNetwork(in_channels=config["image_shape"][0], layers = GetEncDecLayers())  # Initialize your ScoreNetwork with appropriate arguments

state_dict = torch.load(model_path, weights_only=True)
score_network.load_state_dict(state_dict)

score_network.eval()

# Generate a single sample
sample = generate_samples(score_network, 1, image_shape).detach().reshape(*image_shape)


# # Debug: Print the values from the middle row
# middle_row_index = sample.shape[1] // 2  # Calculate the middle row index
# middle_row_values = sample[:, middle_row_index, :]  # Get the middle row
# print(f"Middle row values of the generated sample: {middle_row_values}")

# Create a folder to save the generated image
output_folder = "generated_samples"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "single_sample.png")

# Plot the single sample
fig, ax = plt.subplots(figsize=(4, 4))
img = sample.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) for plotting
img = (img - img.min()) / (img.max() - img.min())  # Normalize image
ax.imshow(img)
ax.axis('off')

# Save the plot to a file
plt.tight_layout()
plt.savefig(output_file)
plt.close(fig)

print(f"Sample saved to {output_file}")
