import tensorflow as tf
import time

# Step 1: Load the dataset
def load_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

# Step 2: Define the model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 3: Train and measure training time
def train_model_on_device(device, x_train, y_train, x_test, y_test):
    print(f"Training on device: {device}")
    with tf.device(device):
        model = create_model()
        
        start_time = time.time()
        history = model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)
        end_time = time.time()

        training_time = end_time - start_time
        _, accuracy = model.evaluate(x_test, y_test, verbose=0)

        print(f"Training time on {device}: {training_time:.2f} seconds")
        print(f"Test accuracy on {device}: {accuracy:.4f}")
        return training_time, accuracy

# Step 4: Run experiments
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()

    # Training without GPU (CPU only)
    cpu_time, cpu_accuracy = train_model_on_device('/CPU:0', x_train, y_train, x_test, y_test)

    # Training with GPU
    gpu_time, gpu_accuracy = train_model_on_device('/GPU:0', x_train, y_train, x_test, y_test)

    # Compare results
    print("\nComparison:")
    print(f"Training time without GPU: {cpu_time:.2f} seconds")
    print(f"Training time with GPU: {gpu_time:.2f} seconds")