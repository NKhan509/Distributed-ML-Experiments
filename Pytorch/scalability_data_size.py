import time
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Function to load and prepare datasets
def prepare_data(dataset_size_ratio=1.0):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the FashionMNIST dataset
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Downsample the training dataset based on dataset_size_ratio
    if dataset_size_ratio < 1.0:
        train_size = int(len(train_data) * dataset_size_ratio)
        train_data = Subset(train_data, range(train_size))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Function to define, train, and measure performance metrics of the model
class SimpleModel(nn.Module):
    def __init__(self, layers_list):
        super(SimpleModel, self).__init__()
        layers = []
        input_size = 784  # 28x28 images flattened
        for layer_size in layers_list[:-1]:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU())
            input_size = layer_size
        layers.append(nn.Linear(layers_list[-2], layers_list[-1]))  # Final output layer
        layers.append(nn.Softmax(dim=1))  # Softmax activation on the output layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Function to train and measure training time, memory usage, and throughput
def train_and_measure(train_loader, test_loader, layers_list, epochs, batch_size, device):
    process = psutil.Process()  # To track memory usage

    # Build the model and move it to the device
    model = SimpleModel(layers_list).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Measure initial memory usage
    initial_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB

    # Measure training time
    start_time = time.time()
    total_samples = 0

    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            # Move data to the device (CPU/GPU)
            data, target = data.view(data.size(0), -1).to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_samples += data.size(0)

    training_time = time.time() - start_time

    # Measure memory usage after training
    final_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB

    # Calculate throughput (samples per second)
    throughput = total_samples / training_time

    # Evaluate the model
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

    accuracy = correct / total

    return {
        "training_time": training_time,
        "memory_usage": final_memory - initial_memory,
        "throughput": throughput,
        "loss": loss.item(),
        "accuracy": accuracy,
    }

# Experiment configurations
dataset_ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
layers_list = [784, 128, 64, 10]  # Layer sizes
epochs = 10
batch_size = 64

# Check for device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Collect results
results = []
for ratio in dataset_ratios:
    print(f"Testing with dataset size ratio: {ratio}")
    train_loader, test_loader = prepare_data(ratio)
    metrics = train_and_measure(train_loader, test_loader, layers_list, epochs, batch_size, device)
    results.append({"dataset_size_ratio": ratio, **metrics})

# Print results
print("\nResults:")
for result in results:
    print(f"Dataset Size Ratio: {result['dataset_size_ratio']:.2f}")
    print(f"  Training Time (s): {result['training_time']:.2f}")
    print(f"  Memory Usage (MB): {result['memory_usage']:.2f}")
    print(f"  Throughput (samples/s): {result['throughput']:.2f}")
    print("-" * 40)
