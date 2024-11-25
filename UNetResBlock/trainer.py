import torch
import wandb
import os
from PIL import Image
from UNetResBlock.model import generate_samples, calc_loss
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(u_net, dataset, config, model_name, log=False, save_model=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_net = u_net.to(device)

    image_shape = config["image_shape"]
    save_dir = os.path.join("UNetResBlock/results", model_name)  # Create the full directory path
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimizer
    opt = torch.optim.Adam(u_net.parameters(), lr=config["learning_rate"])

    # DataLoader
    dloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    results_txt = '/zhome/1a/a/156609/public_html/ShowResults/resultOwn.txt'

    clear_txt_file(results_txt)

    p_bar = False

    for i_epoch in range(config["epochs"]):
        print(f"Epoch {i_epoch} started")
        total_loss = 0

        # Wrap DataLoader with or without tqdm for progress bar
        if p_bar:  # Default to True if not specified
            num_batches = len(dloader)
            progress_bar = tqdm(dloader, total=num_batches, desc=f"Epoch {i_epoch}", ncols=100)
        else:
            progress_bar = dloader  # Use the dataloader directly without tqdm

        # Training loop
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            opt.zero_grad()
            loss = calc_loss(u_net, data)
            loss.backward()
            opt.step()
            total_loss += loss.item() * data.shape[0]

            # Update the description of the progress bar with current loss
            if p_bar:  # Only update tqdm if it's enabled
                progress_bar.set_postfix(loss=loss.item(), total_loss=total_loss)

        avg_loss = total_loss / len(dataset)
        
        torch.save(u_net.state_dict(), f"UNetResBlock/models/{model_name}ckpt.pt")

        if (i_epoch < 5 or i_epoch % 5 == 0):
            print("generating samples")
            # Generate 8 samples after each epoch
            generated_samples = generate_samples(u_net, nsamples=8, image_shape=image_shape, timesteps=1000)
            
            #print_samples_and_data(data, generated_samples)

            # Save the generated samples as a row in the specified folder
            save_samples(generated_samples, save_dir, f"epoch{i_epoch}")
            save_samples(generated_samples, "/zhome/1a/a/156609/public_html/ShowResults", "resultOwn")
            
            # Save the first 8 images from the dataset
            first_batch, _ = next(iter(torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)))
            save_samples(first_batch.to(device), save_dir, f"dataset_samples")
            
            add_line_to_txt(results_txt, str(avg_loss))

        # Compute average loss for the epoch
        print(f"Epoch {i_epoch}, Loss: {avg_loss}")

    print("Training complete.")

def print_samples_and_data(data, generated_samples):
    # Calculate and print dataset statistics
    data_min = data.min().item()
    data_max = data.max().item()
    data_mean = data.mean().item()
    data_std = data.std().item()

    print(f"Dataset Statistics:")
    print(f"  Min: {data_min}, Max: {data_max}")
    print(f"  Avg: {data_mean:.4f}, Std: {data_std:.4f}")

    # Calculate and print generated sample statistics
    samples_min = generated_samples.min().item()
    samples_max = generated_samples.max().item()
    samples_mean = generated_samples.float().mean().item()
    samples_std = generated_samples.float().std().item()

    print(f"Generated Samples Statistics:")
    print(f"  Min: {samples_min}, Max: {samples_max}")
    print(f"  Avg: {samples_mean:.4f}, Std: {samples_std:.4f}")

def save_samples(samples, save_dir, filename):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert each sample (tensor) to a PIL image and arrange them into a row
    samples = samples.cpu()  # Move the samples to CPU if on GPU
    samples = (samples + 1) / 2  # Scale from [-1, 1] to [0, 1]
    samples = samples.clamp(0, 1)  # Ensure values are within [0, 1]
    
    images = []
    
    for i in range(samples.shape[0]):
        img = samples[i].permute(1, 2, 0).clamp(0, 1).numpy() * 255  # Convert to image format
        if samples.shape[1] == 1:  # Check if it's grayscale
            img = img[:, :, 0]  # Remove the channel dimension
            img = Image.fromarray(img.astype('uint8'), mode='L')  # Save as grayscale
        elif samples.shape[1] == 3:  # Check if it's RGB
            img = Image.fromarray(img.astype('uint8'))  # Save as RGB
        else:
            raise ValueError(f"Unexpected number of channels: {samples.shape[1]}")
        images.append(img)

    # Concatenate all images into a row (horizontally)
    concatenated_image = Image.new(
        'RGB' if samples.shape[1] == 3 else 'L', 
        (samples.shape[0] * images[0].width, images[0].height)
    )
    
    # Paste images into the new concatenated image
    for i, img in enumerate(images):
        concatenated_image.paste(img, (i * img.width, 0))
    
    # Save the image as a PNG
    concatenated_image.save(os.path.join(save_dir, f"{filename}.png"))



def add_line_to_txt(file_path, line):
    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file does not exist, create it
        with open(file_path, 'w') as f:
            f.write(line + '\n')  # Write the line and add a newline character
    else:
        # If the file exists, append the line
        with open(file_path, 'a') as f:
            f.write(line + '\n')  # Append the line and add a newline character

def clear_txt_file(file_path):
    with open(file_path, 'w') as f:
        # Opening the file in 'w' mode will clear its contents
        pass  # No need to write anything, just clearing the file