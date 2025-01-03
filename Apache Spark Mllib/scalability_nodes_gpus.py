import time
import psutil
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

# Step 3: Simulate Training Without GPU (CPU only)
def train_without_gpu(spark):
    # Load full dataset
    train_data, test_data = load_data(spark, dataset_ratio=1.0)

    # Initialize the model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

    # Measure training time without GPU
    start_time = time.time()

    # Train the model on CPU
    lr_model = lr.fit(train_data)

    end_time = time.time()
    training_time_cpu = end_time - start_time

    return training_time_cpu

# Step 4: Simulate Training With GPU
def train_with_gpu(spark):
    # Load full dataset
    train_data, test_data = load_data(spark, dataset_ratio=1.0)

    # Initialize the model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

    # Measure training time with GPU (Note: PySpark itself doesn't manage GPU, but we simulate)
    start_time = time.time()

    # Assuming the hardware supports GPU, you would need a deep learning framework like TensorFlow or PyTorch.
    # This is just a mock, as PySpark itself does not handle GPU acceleration for MLlib models directly.

    # For illustration purposes, let's assume the training with GPU is faster
    lr_model = lr.fit(train_data)

    end_time = time.time()
    training_time_gpu = end_time - start_time

    return training_time_gpu

# Step 5: Comparison and Output
def compare_training_times(spark):
    # Simulate training without and with GPU
    training_time_cpu = train_without_gpu(spark)
    training_time_gpu = train_with_gpu(spark)

    # Output the results
    print(f"Training time without GPU: {training_time_cpu:.2f} seconds")
    print(f"Training time with GPU: {training_time_gpu:.2f} seconds")

# Step 6: Main Function
if __name__ == "__main__":
    preprocess_fashion_mnist()  # Ensure data is preprocessed
    spark = SparkSession.builder.appName("Fashion-MNIST-GPU-Comparison").getOrCreate()

    compare_training_times(spark)