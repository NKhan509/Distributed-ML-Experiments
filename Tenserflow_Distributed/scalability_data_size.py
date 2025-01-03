import time
import psutil
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

# Function to load and prepare datasets
def prepare_data(dataset_size_ratio=1.0):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize and reshape the data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Downsample the training dataset based on dataset_size_ratio
    if dataset_size_ratio < 1.0:
        train_size = int(len(x_train) * dataset_size_ratio)
        x_train, y_train = x_train[:train_size], y_train[:train_size]
    
    return x_train, y_train, x_test, y_test

# Function to define, train, and measure performance metrics of the model
def train_and_measure(x_train, y_train, x_test, y_test, layers_list, epochs, batch_size):
    process = psutil.Process()  # To track memory usage
    
    # Build the model
    model = models.Sequential()
    model.add(layers.Dense(layers_list[0], activation='relu', input_shape=(784,)))
    for layer_size in layers_list[1:-1]:
        model.add(layers.Dense(layer_size, activation='relu'))
    model.add(layers.Dense(layers_list[-1], activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Measure initial memory usage
    initial_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    # Measure training time
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    training_time = time.time() - start_time
    
    # Measure memory usage after training
    final_memory = process.memory_info().rss / (1024 * 1024)
    
    # Calculate throughput (samples per second)
    total_samples = len(x_train) * epochs
    throughput = total_samples / training_time
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    return {
        "training_time": training_time,
        "memory_usage": final_memory - initial_memory,
        "throughput": throughput,
        "loss": loss,
        "accuracy": accuracy,
    }

# Experiment configurations
dataset_ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
layers_list = [784, 128, 64, 10]  # Layer sizes
epochs = 10
batch_size = 64

# Collect results
results = []
for ratio in dataset_ratios:
    print(f"Testing with dataset size ratio: {ratio}")
    x_train, y_train, x_test, y_test = prepare_data(ratio)
    metrics = train_and_measure(x_train, y_train, x_test, y_test, layers_list, epochs, batch_size)
    results.append({"dataset_size_ratio": ratio, **metrics})

# Print results
print("\nResults:")
for result in results:
    print(f"Dataset Size Ratio: {result['dataset_size_ratio']:.2f}")
    print(f"  Training Time (s): {result['training_time']:.2f}")
    print(f"  Memory Usage (MB): {result['memory_usage']:.2f}")
    print(f"  Throughput (samples/s): {result['throughput']:.2f}")
    print("-" * 40)