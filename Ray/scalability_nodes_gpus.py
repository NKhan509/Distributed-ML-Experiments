import time
import ray
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import os

# Check TensorFlow GPU status
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Data loading function (using Fashion MNIST)
def load_data():
    # Load the Fashion MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Rescale pixel values to [0, 1]
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Reshape images to (28, 28, 1) for compatibility with the CNN
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Create the TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

    return train_dataset, test_dataset

# Simple CNN model for Fashion MNIST
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

# Training function
def train_model(dataset, use_gpu=True):
    model = create_model()
    if use_gpu:
        with tf.device('/GPU:0'):  # Specify using one GPU
            start_time = time.time()
            for images, labels in dataset:
                model.train_on_batch(images, labels)  # Train on each batch
            end_time = time.time()
            print(f"Training time on GPU: {end_time - start_time:.2f} seconds")
    else:
        start_time = time.time()
        for images, labels in dataset:
            model.train_on_batch(images, labels)  # Train on each batch
        end_time = time.time()
        print(f"Training time without GPU: {end_time - start_time:.2f} seconds")

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Function to run experiment in Ray
@ray.remote
def run_experiment(use_gpu):
    print("Starting experiment...")
    train_dataset, test_dataset = load_data()
    train_model(train_dataset, use_gpu)
    return "Experiment completed"

# Run without GPU
ray.get(run_experiment.remote(use_gpu=False))

# Run with one GPU (simulated by setting use_gpu=True)
ray.get(run_experiment.remote(use_gpu=True))

# Shutdown Ray after experiment
ray.shutdown()