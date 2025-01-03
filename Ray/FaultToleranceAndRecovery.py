import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import os

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

# Simulate failure and recovery
def train_model_with_fault_tolerance(dataset, fault_tolerance=False, checkpoint_path='./fault_tolerance_checkpoints/'):
    model = create_model()
    os.makedirs(checkpoint_path, exist_ok=True)  # Ensure checkpoint directory exists
    start_time = time.time()
    initial_accuracy = None
    epoch_count = 0

    try:
        # Train for a few epochs and simulate failure at a random point
        for epoch in range(1, 6):  # Train for 5 epochs as an example
            epoch_count = epoch
            print(f"\nEpoch {epoch}:")
            model.fit(dataset, epochs=1)  # Train for 1 epoch
            model.save_weights(f'{checkpoint_path}ckpt-{epoch:02d}.weights.h5')  # Save after each epoch
            if fault_tolerance and epoch == 4:  # Simulate failure at epoch 4
                print(f"Simulating failure at epoch {epoch}")
                initial_accuracy = model.evaluate(dataset)  # Capture accuracy before failure
                raise Exception(f"Simulated failure during training at epoch {epoch}")
    except Exception as e:
        print(f"Error during training: {e}")
        print(f"Restoring from checkpoint: {checkpoint_path}ckpt-{epoch_count-1:02d}.weights.h5")
        # Simulate recovery by continuing training after failure
        model = create_model()  # Re-initialize the model for recovery
        model.load_weights(f'{checkpoint_path}ckpt-{epoch_count-1:02d}.weights.h5')  # Load the last checkpoint
        print(f"Resuming at epoch {epoch_count}")
        model.fit(dataset, epochs=2, initial_epoch=epoch_count-1)  # Continue training from the failure point
        recovered_accuracy = model.evaluate(dataset)
        end_time = time.time()

    print(f"Training time (including recovery): {end_time - start_time:.2f} seconds")
    return initial_accuracy, recovered_accuracy

# Run experiment without fault tolerance (normal training)
def run_experiment_without_fault_tolerance():
    train_dataset, test_dataset = load_data()
    model = create_model()
    start_time = time.time()
    model.fit(train_dataset, epochs=1)  # Train for 1 epoch without failure
    accuracy = model.evaluate(test_dataset)
    end_time = time.time()
    print(f"\nNormal training...")
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Final accuracy (without failure): {accuracy[1]:.4f}")

# Run the experiment with fault tolerance enabled
def run_experiment_with_fault_tolerance():
    train_dataset, test_dataset = load_data()
    initial_accuracy, recovered_accuracy = train_model_with_fault_tolerance(train_dataset, fault_tolerance=True)

    print(f"\nAccuracy before failure (Epoch 4): {initial_accuracy[1] if initial_accuracy else 'N/A'}")
    print(f"Accuracy after recovery: {recovered_accuracy[1]:.4f}")

# Run both experiments
run_experiment_without_fault_tolerance()  # Without fault tolerance (for comparison)
run_experiment_with_fault_tolerance()  # With fault tolerance enabled