# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-NGCvbPWtj6hDGCAyhBqslWrGiyTenHo
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Load the dataset
def load_fashion_mnist(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Step 2: Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Step 3: Train and measure training time
def train_model_on_device(device, train_loader, test_loader):
    print(f"Training on device: {device}")

    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    start_time = time.time()
    model.train()
    for epoch in range(5):  # Train for 5 epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    end_time = time.time()

    training_time = end_time - start_time

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total

    print(f"Training time on {device}: {training_time:.2f} seconds")
    print(f"Test accuracy on {device}: {accuracy:.4f}")
    return training_time, accuracy

# Step 4: Run experiments
if __name__ == "__main__":
    train_loader, test_loader = load_fashion_mnist()

    # Training without GPU (CPU only)
    cpu_time, cpu_accuracy = train_model_on_device('cpu', train_loader, test_loader)

    # Training with GPU 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_time, gpu_accuracy = train_model_on_device(device, train_loader, test_loader)

    # Compare results
    print("\nComparison:")
    print(f"Training time without GPU: {cpu_time:.2f} seconds")
    print(f"Training time with GPU: {gpu_time:.2f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
    print(f"Accuracy on CPU: {cpu_accuracy:.4f}")
    print(f"Accuracy on GPU: {gpu_accuracy:.4f}")