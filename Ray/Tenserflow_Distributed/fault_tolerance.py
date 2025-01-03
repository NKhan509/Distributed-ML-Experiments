import tensorflow as tf
import os
import time

# Use TensorFlow Distributed Strategy
strategy = tf.distribute.MirroredStrategy()

# Define the model creation function
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# Load dataset (MNIST example)
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)

# Function to simulate fault-tolerant training
def fault_tolerant_training():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Fault Tolerance and Recovery ~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Directory for checkpoints
    checkpoint_dir = "./fault_tolerance_checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "ckpt")

    # Define distributed strategy and model
    with strategy.scope():
        model = create_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Create a checkpoint manager
    checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # Restore from checkpoint if available
    if manager.latest_checkpoint:
        print(f"Restoring from checkpoint: {manager.latest_checkpoint}")
        checkpoint.restore(manager.latest_checkpoint)

    # Training configuration
    batch_size = 128
    epochs = 5

    # Start normal training
    start_time = time.time()
    try:
        print("\nNormal training...")
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}...")
            history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=2)
            if epoch == 3:  # Simulate a failure at epoch 4
                raise Exception("Simulated failure during training")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Accuracy before failure (Epoch {epoch + 1}): {accuracy:.4f}")
    except Exception as e:
        print(f"Error during training: {e}")

    # Simulate recovery
    print("Recovering and resuming training...")
    recovered_start_time = time.time()
    for epoch in range(4, epochs):
        print(f"Epoch {epoch + 1}...")
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=2)
        manager.save()  # Save checkpoint after each epoch

    # Evaluate after recovery
    final_loss, final_accuracy = model.evaluate(x_test, y_test, verbose=0)
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTraining time (including recovery): {total_time:.2f} seconds")
    print(f"Accuracy before failure: {history.history['accuracy'][-1]:.4f}")
    print(f"Final accuracy after recovery: {final_accuracy:.4f}")

# Main execution
if __name__ == "__main__":
    fault_tolerant_training()