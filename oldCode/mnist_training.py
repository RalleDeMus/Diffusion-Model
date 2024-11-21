# mnist_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

# Initialize Weights & Biases
wandb.init(
    project="mnist-classification-HPC",  # Name of the project on W&B
    name="Run name is this xd",  # Name of the run
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 64,
    }
)

print("Is GPU available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

# Load W&B config and set device
config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# if device == "cuda"
print("Using GPU", torch.cuda.get_device_name(0))

# Data Loading and Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Training loop with W&B logging
for epoch in range(config.epochs):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Log batch loss to W&B
        wandb.log({"batch_loss": loss.item()})

    # Calculate and log epoch loss
    epoch_loss = running_loss / len(train_loader)
    wandb.log({"epoch_loss": epoch_loss})
    print(f"Epoch {epoch+1}/{config.epochs}, Loss: {epoch_loss}")

# Evaluation
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
wandb.log({"accuracy": accuracy})
print(f"Test accuracy: {accuracy:.2f}%")
