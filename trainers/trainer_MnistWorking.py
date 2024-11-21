import torch
import wandb
import os
from models.attention_model.attention_model import generate_samples, calc_loss
import matplotlib.pyplot as plt



def train_model(score_network, dataset, config, image_shape, log=False, save_model = False, model_name = ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_network = score_network.to(device)

    # Optimizer
    opt = torch.optim.Adam(score_network.parameters(), lr=config["learning_rate"])
    
    # DataLoader
    dloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    for i_epoch in range(config["epochs"]):
        print(f"Epoch {i_epoch} started")
        total_loss = 0

        # Training loop
        for data, _ in dloader:
            data = data.to(device)
            opt.zero_grad()
            loss = calc_loss(score_network, data)  # Using simplified calc_loss
            loss.backward()
            opt.step()
            total_loss += loss.item() * data.shape[0]

            
        # Compute average loss for the epoch
        avg_loss = total_loss / len(dataset)
        if log:
            wandb.log({"loss": avg_loss})

        # Generate and log samples every few epochs
        if i_epoch % (1 if log else 1) == 0:
            print(f"Epoch {i_epoch}, Loss: {avg_loss}")
            samples = generate_samples(score_network, 16, image_shape).detach().reshape(-1, *image_shape)  # Using simplified generate_samples
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for ax, img in zip(axes.flatten(), samples):
                img = img.permute(1, 2, 0).cpu().numpy()
                ax.imshow((img - img.min()) / (img.max() - img.min()))  # Normalize image
                ax.axis('off')
            if log:
                wandb.log({"Generated samples": wandb.Image(fig)})
            plt.close(fig)

    # Save the best model: .pt in models folder
    if (save_model and model_name != ""):
        model_folder = "savedModels"
        os.makedirs(model_folder, exist_ok=True)
        model_filename = model_name+".pt"
        model_path = os.path.join(model_folder, model_filename)
        torch.save(score_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print("Training complete.")

# def train_model(score_network, dataset, config, image_shape, log=False):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     score_network = score_network.to(device)

#     opt = torch.optim.Adam(score_network.parameters(), lr=config["learning_rate"])
#     dloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

#     # Compute alpha_bars and betas
#     num_timesteps = 1000
#     alpha_bars, betas = compute_alphas_and_betas(num_timesteps)
#     alpha_bars = alpha_bars.to(device)
#     betas = betas.to(device)
    
#     for i_epoch in range(config["epochs"]):
#         print(f"Epoch {i_epoch} started")
#         total_loss = 0
#         for data, _ in dloader:
#             data = data.to(device)
#             opt.zero_grad()
#             loss = calc_loss(score_network, data, alpha_bars, betas, num_timesteps)
#             loss.backward()
#             opt.step()
#             total_loss += loss.item() * data.shape[0]

#         avg_loss = total_loss / len(dataset)
#         if log:
#             wandb.log({"loss": avg_loss})
        
#         if i_epoch % (1 if log else 1) == 0:  # Save samples every few epochs
#             print(f"Epoch {i_epoch}, Loss: {avg_loss}")
#             samples = generate_samples(score_network, 16, image_shape, alpha_bars, betas, num_timesteps).detach().reshape(-1, *image_shape)
#             fig, axes = plt.subplots(4, 4, figsize=(8, 8))
#             for ax, img in zip(axes.flatten(), samples):
#                 img = img.permute(1, 2, 0).cpu().numpy()
#                 ax.imshow((img - img.min()) / (img.max() - img.min()))  # Normalize image
#                 ax.axis('off')
#             if log:
#                 wandb.log({"Generated samples": wandb.Image(fig)})
#             plt.close(fig)

#         # save the best model: .pt in models folder:


# def cosine_alpha_bar(t, s=0.008):
#     return torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2

# def compute_alphas_and_betas(num_timesteps=1000, s=0.008):
#     timesteps = torch.linspace(0, 1, num_timesteps)
#     alpha_bars = cosine_alpha_bar(timesteps, s)
#     betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
#     betas = torch.cat([torch.tensor([betas[0]]), betas])  # Add beta_0 for consistency
#     return alpha_bars, betas
