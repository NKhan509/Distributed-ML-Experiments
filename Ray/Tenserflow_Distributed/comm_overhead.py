import tensorflow as tf
import time
import numpy as np

# Function to create a simple model
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Load dataset with a variable size ratio
def load_data(dataset_size_ratio=1.0):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Calculate the number of samples to load based on the ratio
    num_train_samples = int(len(x_train) * dataset_size_ratio)
    num_test_samples = int(len(x_test) * dataset_size_ratio)
    
    # Resize the dataset according to the ratio
    x_train = x_train[:num_train_samples].reshape(-1, 784).astype("float32") / 255.0
    y_train = y_train[:num_train_samples]
    x_test = x_test[:num_test_samples].reshape(-1, 784).astype("float32") / 255.0
    y_test = y_test[:num_test_samples]
    
    return (x_train, y_train), (x_test, y_test)

# Function to train with synchronous strategy
def train_with_sync_strategy(dataset_size_ratio):
    strategy_sync = tf.distribute.MirroredStrategy()
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data(dataset_size_ratio)

    # Create and compile the model inside the strategy scope
    with strategy_sync.scope():
        model = create_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Start training and measure time
    start_time = time.time()
    model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)
    end_time = time.time()

    # Calculate training time and latency
    training_time = end_time - start_time
    worker_latency = training_time / len(x_train)  # Simulate worker latency as training time per sample

    return training_time, worker_latency

# Function to train with asynchronous strategy
def train_with_async_strategy(dataset_size_ratio):
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data(dataset_size_ratio)

    # Create and compile the model
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Simulate asynchronous by training on smaller batches
    start_time = time.time()
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)  # Use smaller batch size for "asynchronous"
    end_time = time.time()

    # Calculate training time and latency
    training_time = end_time - start_time
    worker_latency = training_time / len(x_train)  # Simulate worker latency as training time per sample

    return training_time, worker_latency

# Main function to compare synchronous and asynchronous training
if __name__ == "__main__":
    dataset_ratios = [0.10, 0.25, 0.50, 0.75, 1.00]
    
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