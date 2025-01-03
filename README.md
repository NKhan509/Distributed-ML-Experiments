# Distributed-ML-Experiments 


#**How To Set Up Apache Spark Mllib And Run The Experiments:**

**I. Install Python:**

•	Download and install Python from https://www.python.org/downloads/  if it's not already installed. 
•	Verify installation by running python --version in CMD.

**II. Install Java:**
•	Apache Spark requires Java, so ensure you have Java 8 or later installed.
•	You can check if Java is installed by running the command java -version in the CMD.

**III. Set Python and Java in the Environment Variables:**
To set up Python and Java in the environment variables:
On Windows:
**•	Python:**
1.	Open the Start menu and search for Environment Variables.
2.	Click Edit the system environment variables.
3.	In the System Properties window, click on the Environment Variables button.
4.	Under System variables, find and select Path, then click Edit.
5.	Add the path to the Python installation directory (e.g. C:\Users\PMLS\AppData\Local\Microsoft\WindowsApps\python.exe).
6.	Add the path to the Scripts folder inside the Python installation directory (e.g. C:\Users\PMLS\AppData\Local\Microsoft\WindowsApps\python.exe).
7.	Click OK.

**•	Java:**
1.	Find the Java installation directory, e.g., C:\Program Files\Java\jdk-11.0.2.
2.	Follow the same steps as above to add the path to the Java bin directory (e.g., C:\Program Files\Java\jdk-11.0.2\bin) under Path.
3.	Click OK.

**IV. Install Apache Spark:**
•	Download and install Apache Spark from Apache Spark official website.
•	Unzip it and set the SPARK_HOME environment variable to the folder where you extracted Spark.

**V. Install Hadoop:**
•	If you want to use Hadoop with Spark, download Hadoop from Apache Hadoop website.
•	Set the HADOOP_HOME environment variable similarly.


**VI. Install PySpark:**
•	Install PySpark, the Python API for Spark, using the following command:
pip install pyspark


**VII. Run the Experiments:**
•	Using MLlib in Apache Spark, define the machine learning model you wish to train (e.g., regression, classification).
•	Load your dataset into Apache Spark.














#**How To Set Up Tensorflow Distributed And Run The Experiments:**

**I. Install Python:**
Download and install Python from https://www.python.org/downloads/  if it's not already installed. Verify installation by running python --version in CMD.

**II. Install TensorFlow:**
Run the command pip install tensorflow in CMD to install TensorFlow.

**III. Verify TensorFlow Installation:**
Confirm successful installation by running:
python -c "import tensorflow as tf; print(tf.__version__)"

**IV. Install Additional Libraries for TensorFlow Distributed:**
•	pip install tensorflow (for core TensorFlow)
•	pip install tensorflow_datasets (for datasets)
•	pip install tensorflow_addons (for extra utilities)

**V. Run the Experiments:**
•	Load your dataset and define the model within the strategy scope.
•	Run the code on your local machine or Google Colab for the Scalability with Number of Nodes/GPUs experiment. Use Google Colab if you don't have multiple GPUs available locally.












#**How To Setup Ray And Run The Experiments:**

In order to replicate the experiments, the following steps can be followed:
**I. Ensure Python is Installed:**
Ensure python is installed in the PC. Run the following command in the command line terminal (CMD) to check:
python –version

**II. Install Ray**
Install Ray by running the following command in CMD:
pip install ray

**III. Verify Ray Installation**
Check if Ray is properly installed by entering the following code in CMD:
python -c "import ray; print(ray.__version__)"

**IV. Install Additional Libraries for Ray**
Install the following libraries to integrate features of Ray in your program:
pip install "ray[default]"
pip install "ray[tune]"  # For hyperparameter tuning
pip install "ray[serve]"  # For model serving

**V. Run the Experiments**
Run the codes included in the GitHub repository either on your local machine or on Google Colab. If you do not have access to multiple GPUs then run the resource scalability experiment on Google Colab.


