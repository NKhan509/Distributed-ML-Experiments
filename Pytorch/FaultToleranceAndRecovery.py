import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the model creation function
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)

# Load dataset (MNIST example)
def load_data(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Function to simulate fault-tolerant training
def fault_tolerant_training():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fault Tolerance and Recovery ~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Load data
    train_loader, test_loader = load_data()

    # Initialize the model, optimizer, and loss function
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Directory for checkpoints
    checkpoint_dir = "./fault_tolerance_checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "ckpt.pth")

    # Create checkpoint folder if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"Restoring from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Resumed training from epoch {epoch + 1}")
    else:
        epoch = 0
        loss = None
        print("Starting training from scratch...")

    # Training configuration
    epochs = 5
    batch_size = 128

    # Start normal training
    start_time = time.time()
    try:
        print("\nNormal training...")
        for epoch in range(epoch, epochs):
            model.train()
            running_loss = 0.0
            for data, target in train_loader:
                data, target = data.view(data.size(0), -1).to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

            # Simulate failure after epoch 3
            if epoch == 3:
                raise Exception("Simulated failure during training")

            # Save checkpoint after each epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)

        # Evaluate model after normal training
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy_before_failure = 100 * correct / total
        print(f"Accuracy before failure (Epoch {epoch + 1}): {accuracy_before_failure:.4f}")

    except Exception as e:
        print(f"Error during training: {e}")

    # Simulate recovery from failure
    print("Recovering and resuming training...")
    model.train()
    recovered_start_time = time.time()

    for epoch in range(epoch + 1, epochs):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.view(data.size(0), -1).to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

    # Final evaluation after recovery
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(data.size(0), -1).to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy_after_recovery = 100 * correct / total
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTraining time (including recovery): {total_time:.2f} seconds")
    print(f"Accuracy before failure: {accuracy_before_failure:.4f}")
    print(f"Final accuracy after recovery: {accuracy_after_recovery:.4f}")

# Main execution
if __name__ == "__main__":
    # Check if CUDA is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fault_tolerant_training()