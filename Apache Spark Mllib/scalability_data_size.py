import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import psutil

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

# Step 3: Measure Training Time, Memory Usage, and Throughput
def measure_scalability(spark, dataset_ratios):
    results = []

    for ratio in dataset_ratios:
        print(f"\nTesting with dataset ratio: {ratio:.2f}")

        # Load data with the current ratio
        train_data, test_data = load_data(spark, dataset_ratio=ratio)

        # Initialize the model
        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

        # Measure memory usage before training
        memory_before = psutil.virtual_memory().used / (1024 * 1024)

        # Measure training time
        start_time = time.time()
        lr_model = lr.fit(train_data)
        end_time = time.time()

        # Measure memory usage after training
        memory_after = psutil.virtual_memory().used / (1024 * 1024)

        # Evaluate the model
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        predictions = lr_model.transform(test_data)
        accuracy = evaluator.evaluate(predictions)

        # Calculate throughput (samples processed per second)
        training_time = end_time - start_time
        num_samples = train_data.count()
        throughput = num_samples / training_time

        # Store results
        results.append({
            "Dataset Ratio": ratio,
            "Training Time (s)": training_time,
            "Memory Usage (MB)": memory_after - memory_before,
            "Throughput (samples/sec)": throughput,
            "Accuracy": accuracy
        })

    return results

# Step 4: Main Function
if __name__ == "__main__":
    preprocess_fashion_mnist()  # Ensure data is preprocessed
    spark = SparkSession.builder.appName("Fashion-MNIST-Scalability").getOrCreate()

    # Define dataset size ratios
    dataset_ratios = [0.1, 0.25, 0.5, 0.75, 1.0]

    # Measure scalability
    results = measure_scalability(spark, dataset_ratios)

    # Print results
    print("\nScalability Results:")
    for result in results:
        print(f"Dataset Ratio: {result['Dataset Ratio']:.2f}")
        print(f"  Training Time (s): {result['Training Time (s)']:.2f}")
        print(f"  Memory Usage (MB): {result['Memory Usage (MB)']:.2f}")
        print(f"  Throughput (samples/sec): {result['Throughput (samples/sec)']:.2f}")
        print(f"  Accuracy: {result['Accuracy']:.4f}")
        print("----------------------------------------")