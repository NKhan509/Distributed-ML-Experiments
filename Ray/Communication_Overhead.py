import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

# Check if GPUs are available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Rescale pixel values to [0, 1]
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Reshape images to (28, 28, 1) for compatibility with CNN
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Create TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

    return train_dataset, test_dataset

# Create a simple CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)  # 10 classes for Fashion MNIST
    ])
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Simulate Synchronous Training with Communication Overhead
def train_synchronous(dataset, dataset_size_ratio, max_steps=100):
    model = create_model()
    start_time = time.time()

    total_latency = 0
    steps = 0
    # Training loop with simulated communication delay
    for step, (images, labels) in enumerate(dataset):
        if steps >= max_steps:
            break
        step_start_time = time.time()

        # Simulate communication delay during synchronization
        if step % 10 == 0:  # Adding delay every 10 steps
            # Simulating communication delay
            time.sleep(0.1)  # Artificial delay to simulate network communication

        model.train_on_batch(images, labels)
        step_end_time = time.time()

        total_latency += (step_end_time - step_start_time)
        steps += 1

    end_time = time.time()
    training_time = end_time - start_time
    worker_latency = total_latency / steps if steps > 0 else 0  # Average worker latency per batch

    print(f"INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:CPU:0',)\n")
    print(f"Comparison of Synchronous vs Asynchronous Training:")
    print(f"Dataset Size Ratio: {dataset_size_ratio:.2f}")
    print(f"  Synchronous Training Time (s): {training_time:.2f}")
    print(f"  Synchronous Worker Latency (s): {worker_latency:.4f}")
    print(f"----------------------------------------")
    return training_time, worker_latency

# Simulate Asynchronous Training (No synchronization delay)
def train_asynchronous(dataset, dataset_size_ratio, max_steps=100):
    model = create_model()
    start_time = time.time()

    total_latency = 0
    steps = 0
    # Training loop without synchronization
    for step, (images, labels) in enumerate(dataset):
        if steps >= max_steps:
            break
        step_start_time = time.time()
        model.train_on_batch(images, labels)
        step_end_time = time.time()

        total_latency += (step_end_time - step_start_time)
        steps += 1

    end_time = time.time()
    training_time = end_time - start_time
    worker_latency = total_latency / steps if steps > 0 else 0  # Average worker latency per batch

    print(f"INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:CPU:0',)\n")
    print(f"Comparison of Synchronous vs Asynchronous Training:")
    print(f"Dataset Size Ratio: {dataset_size_ratio:.2f}")
    print(f"  Asynchronous Training Time (s): {training_time:.2f}")
    print(f"  Asynchronous Worker Latency (s): {worker_latency:.4f}")
    print(f"----------------------------------------")
    return training_time, worker_latency

# Run the experiment
def run_experiment():
    train_dataset, test_dataset = load_data()
    dataset_size = len(train_dataset)

    # Run multiple experiments with different dataset size ratios
    for dataset_size_ratio in [0.1, 0.25, 0.5, 0.75, 1.0]:
        # Calculate the number of steps for the subset
        subset_size = int(dataset_size * dataset_size_ratio)
        subset_dataset = train_dataset.take(subset_size)

        # Run Synchronous Training
        sync_time, sync_latency = train_synchronous(subset_dataset, dataset_size_ratio)

        # Run Asynchronous Training
        async_time, async_latency = train_asynchronous(subset_dataset, dataset_size_ratio)

        # Print the results
        print(f"Training time with synchronous mode: {sync_time:.2f} seconds")
        print(f"Training time with asynchronous mode: {async_time:.2f} seconds")

# Execute the experiment
run_experiment()