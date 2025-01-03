import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Function to create a simple model
def create_model():
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.Softmax(dim=1)
    )

# Load dataset with a variable size ratio
def load_data(dataset_size_ratio=1.0, batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load MNIST dataset
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Calculate the number of samples to load based on the ratio
    num_train_samples = int(len(train_data) * dataset_size_ratio)
    num_test_samples = int(len(test_data) * dataset_size_ratio)

    # Subset the dataset based on the size ratio
    train_data = Subset(train_data, range(num_train_samples))
    test_data = Subset(test_data, range(num_test_samples))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Function to train with synchronous strategy
def train_with_sync_strategy(dataset_size_ratio):
    # Load data
    train_loader, _ = load_data(dataset_size_ratio)

    # Create the model and move it to the GPU or CPU
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Start training and measure time
    start_time = time.time()
    total_samples = 0

    model.train()
    for data, target in train_loader:
        data, target = data.view(data.size(0), -1).to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_samples += data.size(0)

    end_time = time.time()

    # Calculate training time and latency
    training_time = end_time - start_time
    worker_latency = training_time / total_samples  # Simulate worker latency as training time per sample

    return training_time, worker_latency

# Function to train with asynchronous strategy
def train_with_async_strategy(dataset_size_ratio):
    # Load data
    train_loader, _ = load_data(dataset_size_ratio, batch_size=64)  # Smaller batch size for "asynchronous"

    # Create the model and move it to the GPU or CPU
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Start training and measure time
    start_time = time.time()
    total_samples = 0

    model.train()
    for data, target in train_loader:
        data, target = data.view(data.size(0), -1).to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_samples += data.size(0)

    end_time = time.time()

    # Calculate training time and latency
    training_time = end_time - start_time
    worker_latency = training_time / total_samples  # Simulate worker latency as training time per sample

    return training_time, worker_latency

# Main function to compare synchronous and asynchronous training
if __name__ == "__main__":
    dataset_ratios = [0.10, 0.25, 0.50, 0.75, 1.00]

    # Check for device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ratio in dataset_ratios:
        # Run synchronous training
        sync_training_time, sync_worker_latency = train_with_sync_strategy(ratio)

        # Run asynchronous training
        async_training_time, async_worker_latency = train_with_async_strategy(ratio)

        # Output results for the current dataset ratio
        print("\nComparison of Synchronous vs Asynchronous Training:")
        print(f"Dataset Size Ratio: {ratio:.2f}")
        print(f"  Synchronous Training Time (s): {sync_training_time:.2f}")
        print(f"  Synchronous Worker Latency (s): {sync_worker_latency:.4f}")
        print(f"  Asynchronous Training Time (s): {async_training_time:.2f}")
        print(f"  Asynchronous Worker Latency (s): {async_worker_latency:.4f}")
        print("----------------------------------------")