# Distributed-ML-Experiments 


# **How To Set Up Apache Spark Mllib And Run The Experiments:**

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














# **How To Set Up Tensorflow Distributed And Run The Experiments:**

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












# **How To Setup Ray And Run The Experiments:**

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



# Setting Up PyTorch and Running Experiments in Google Colab

**Step 1: Open Google Colab**
1.	Go to Google Colab.
2.	Create a new notebook.
   
**Step 2: Install PyTorch (if required)**
1.	By default, PyTorch is pre-installed in Colab. Check the version:
import torch
print(torch.__version__)
2.	If you need a specific version, install it:
o	For GPU support (CUDA 11.8):
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
o	For CPU-only version:
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

**Step 3: Verify Installation**
1.	Run the following code to ensure PyTorch and GPU are correctly set up:
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

**Step 4: Enable GPU in Colab**
1.	Go to Runtime > Change runtime type.
2.	Under Hardware accelerator, select GPU and click Save.

**Step 5: Run a Simple Experiment**
Here’s an example experiment to ensure everything is working:


1.	Create a small neural network:
import torch
import torch.nn as nn
import torch.optim as optim

•	Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)
model = SimpleNN().to('cuda' if torch.cuda.is_available() else 'cpu')
2.	Generate synthetic data:
import torch

•	Create dummy data
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda' if torch.cuda.is_available() else 'cpu')
targets = torch.tensor([[1.0], [2.0], [3.0]], device='cuda' if torch.cuda.is_available() else 'cpu')
3.	Train the model:
•	Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

•	Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

4.	Test the model:
test_input = torch.tensor([[7.0, 8.0]], device='cuda' if torch.cuda.is_available() else 'cpu')
prediction = model(test_input)
print(f"Prediction for input {test_input.cpu().numpy()}: {prediction.item():.4f}")

**Step 6: Run Your Experiments**
Now that PyTorch is set up, you can proceed with your experiments. Modify the model, dataset, or training loop as needed based on your use case (e.g., MNIST, CIFAR-10, or custom datasets).


