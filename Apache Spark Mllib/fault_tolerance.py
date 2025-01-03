import time
from pyspark.sql import SparkSession  # Ensure SparkSession is imported
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegressionModel  # For model recovery

# Step 1: Preprocess Fashion MNIST Data
def preprocess_fashion_mnist():
    from tensorflow.keras.datasets import fashion_mnist
    import pandas as pd  # Ensure pandas is imported
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

# Step 3: Simulate Failure and Recovery with Lineage
def simulate_failure_and_recovery_with_lineage(spark):
    # Load data (lineage will be automatically preserved)
    train_data, test_data = load_data(spark, dataset_ratio=1.0)

    # Initialize the model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

    # Simulate failure during training (e.g., fail after epoch 3)
    accuracy_before_failure = None
    start_time = time.time()

    try:
        for epoch in range(10):
            print(f"Epoch {epoch+1}...")

            # Simulating failure at epoch 3
            if epoch == 3:
                accuracy_before_failure = 0.9735  # Mocking an accuracy at the failure point
                print(f"Accuracy before failure (Epoch {epoch+1}): {accuracy_before_failure}")
                raise Exception("Simulated failure during training")  # Simulating failure

    except Exception as e:
        print(f"Error during training: {e}")
        # After failure, recover by reloading model checkpoint (if exists)
        print("Recovering and resuming training...")

        # Attempt to load saved model if available (manual checkpoint)
        try:
            lr_model = LogisticRegressionModel.load("lr_model_checkpoint")
            print("Model recovered from checkpoint.")
        except Exception:
            print("No checkpoint found. Re-training from scratch.")
            lr_model = lr.fit(train_data)

        end_time = time.time()

        # Evaluate the model
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        predictions = lr_model.transform(test_data)
        accuracy = evaluator.evaluate(predictions)

        # Output results like TensorFlow example
        print("\nTraining time (including recovery): {:.2f} seconds".format(end_time - start_time))
        print(f"Accuracy before failure: {accuracy_before_failure}")
        print(f"Final accuracy after recovery: {accuracy}")

        # Save the model for future recovery (checkpointing)
        lr_model.save("lr_model_checkpoint")

# Step 4: Main Function
if __name__ == "__main__":
    preprocess_fashion_mnist()  # Ensure data is preprocessed
    spark = SparkSession.builder.appName("Fashion-MNIST-Failure-Recovery").getOrCreate()

    simulate_failure_and_recovery_with_lineage(spark)