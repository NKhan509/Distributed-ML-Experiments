import os
import ray
import psutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Generate synthetic dataset
def generate_data(dataset_size, input_size, output_size):
    x = np.random.rand(dataset_size, input_size).astype(np.float32)
    y = np.random.rand(dataset_size, output_size).astype(np.float32)
    return x, y

# Measure memory usage
def measure_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Convert bytes to MB

# Training function
def train_model(model, optimizer, criterion, data_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# Main function
def main():
    # Ray initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ray.shutdown()
    ray.init()

    # Parameters
    input_size = 10
    output_size = 1
    base_dataset_size = 10000
    dataset_ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
    batch_size = 64
    epochs = 10

    print("Testing with different dataset size ratios...\n")
    results = []

    for ratio in dataset_ratios:
        dataset_size = int(base_dataset_size * ratio)
        print(f"Testing with dataset size ratio: {ratio:.2f}")

        # Generate data
        x, y = generate_data(dataset_size, input_size, output_size)
        dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model, optimizer, and criterion
        model = SimpleNN(input_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Measure initial memory usage
        initial_memory = measure_memory_usage()

        # Training time measurement
        start_time = time.time()
        train_model(model, optimizer, criterion, data_loader, epochs)
        end_time = time.time()

        # Measure memory usage after training
        final_memory = measure_memory_usage()

        # Calculate throughput
        training_time = end_time - start_time
        throughput = dataset_size / training_time

        # Store results
        results.append({
            "dataset_ratio": ratio,
            "training_time": training_time,
            "memory_usage": final_memory - initial_memory,
            "throughput": throughput
        })

    # Display results
    print("\nResults:")
    for result in results:
        print(f"Dataset Size Ratio: {result['dataset_ratio']:.2f}")
        print(f"  Training Time (s): {result['training_time']:.2f}")
        print(f"  Memory Usage (MB): {result['memory_usage']:.2f}")
        print(f"  Throughput (samples/s): {result['throughput']:.2f}")
        print("-" * 40)

# Run the main function
if __name__ == "__main__":
    main()

