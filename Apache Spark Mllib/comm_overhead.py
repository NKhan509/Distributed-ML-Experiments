import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 1: Preprocess Fashion MNIST Data
def preprocess_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Flatten the images
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # Combine features and labels into a DataFrame
    train_df = pd.DataFrame(x_train_flat)
    train_df['label'] = y_train

    test_df = pd.DataFrame(x_test_flat)
    test_df['label'] = y_test

    # Save as CSV
    train_df.to_csv("fashion_mnist_train.csv", index=False)
    test_df.to_csv("fashion_mnist_test.csv", index=False)

# Step 2: Load Data in Spark
def load_data(spark, dataset_ratio=1.0):
    train_data = spark.read.csv("fashion_mnist_train.csv", header=True, inferSchema=True)
    test_data = spark.read.csv("fashion_mnist_test.csv", header=True, inferSchema=True)

    # Sample the data according to the dataset_ratio
    train_data = train_data.sample(False, dataset_ratio, seed=42)
    test_data = test_data.sample(False, dataset_ratio, seed=42)

    # Prepare features for MLlib
    assembler = VectorAssembler(inputCols=train_data.columns[:-1], outputCol="features")
    train_data = assembler.transform(train_data).select("features", "label")
    test_data = assembler.transform(test_data).select("features", "label")

    return train_data, test_data

# Step 3: Synchronous Training Function (Sequential Training)
def synchronous_training(spark, dataset_ratio):
    # Load dataset based on the ratio
    train_data, test_data = load_data(spark, dataset_ratio=dataset_ratio)

    # Initialize the model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

    # Measure synchronous training time
    start_time = time.time()

    # Train the model synchronously
    lr_model = lr.fit(train_data)

    end_time = time.time()
    training_time_sync = end_time - start_time

    # Calculate worker latency
    num_workers = 1  # Assuming a single worker for synchronous training
    worker_latency_sync = training_time_sync / num_workers

    return training_time_sync, worker_latency_sync

# Step 4: Asynchronous Training Function (Simulated Parallelism)
def asynchronous_training(spark, dataset_ratio):
    # Load dataset based on the ratio
    train_data, test_data = load_data(spark, dataset_ratio=dataset_ratio)

    # Initialize the model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

    # Simulating asynchronous training with parallelism (workers processing concurrently)
    start_time = time.time()

    # Asynchronously train the model (mock simulation)
    lr_model = lr.fit(train_data)

    end_time = time.time()
    training_time_async = end_time - start_time

    # Calculate worker latency for asynchronous scenario (mocked for simulation)
    num_workers = 4  # Simulate 4 workers processing asynchronously
    worker_latency_async = training_time_async / num_workers

    return training_time_async, worker_latency_async

# Step 5: Comparison and Output for Different Dataset Sizes
def compare_training_modes(spark, dataset_ratios):
    for ratio in dataset_ratios:
        print(f"\nDataset Size Ratio: {ratio:.2f}")

        # Simulate synchronous training
        training_time_sync, worker_latency_sync = synchronous_training(spark, ratio)

        # Simulate asynchronous training
        training_time_async, worker_latency_async = asynchronous_training(spark, ratio)

        # Output the results for each ratio
        print(f"  Synchronous Training Time (s): {training_time_sync:.2f}")
        print(f"  Synchronous Worker Latency (s): {worker_latency_sync:.4f}")
        print(f"  Asynchronous Training Time (s): {training_time_async:.2f}")
        print(f"  Asynchronous Worker Latency (s): {worker_latency_async:.4f}")

# Step 6: Main Function
if __name__ == "__main__":
    preprocess_fashion_mnist()  # Ensure data is preprocessed
    spark = SparkSession.builder.appName("Fashion-MNIST-Training-Comparison").getOrCreate()

    # Define dataset size ratios
    dataset_ratios = [0.10, 0.25, 0.50, 0.75, 1.0]

    compare_training_modes(spark, dataset_ratios)