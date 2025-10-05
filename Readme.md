# CNN - Image Classification

## 📝 Overview
This project implements a complete end-to-end Deep Learning pipeline using a Convolutional Neural Network (CNN) for image classification.  
The goal of this project is to develop a proof of concept for a structured end-to-end Deep Learning model.
It covers all stages — from data collection and preprocessing to model training, evaluation, deployment and a CI/CD workflow prepared.

## 🚀 Project Pipeline
1. **Data Collection** – CIFAR10 dataset that contains images with the following categories: 'plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'
2. **Preprocessing** – Flipping, Cropping, Shifting, Contrast Adjustment.
3. **Model Architecture** – CNN built with PyTorch.
4. **Training** 
    - Epochs: 50
    - Batch size: 256
    - Optimizer: Adam
    - Metric to evaluate: Accuracy 
    - Hardware: Nvidia RTX 4070 SUPER
5. **Deployment** 
    - Used the model via Flask API (+HTML and CSS)
    - Created a Docker File
    - Uploaded to EC2
6. **Workflow** - when app.py or CNN_architecture.py changes deployment is performed again. 


## 🧩 Project Structure

```project/
├── 00_ModelDevelpment/ # Create and Save the model
│ ├── 00_TrainingData/ # Dataset for training and testing
│ ├── 01_TestingDataFromInternet/ # Images to test if the model is working after training
│ ├── Model_Development.ipynb # Entire model creation process

├── 01_ModelDeployment/ # Create the Flask API and Dockerfile
│ ├── .dockerignore
│ ├── app.py # Flask API
│ ├── CNN_architecture.py # CNN architecture
│ ├── Dockerfile 
│ └── requirements.txt

├── 02_Workflow/ # Automated process to deploy the app when app.py or CNN_architecture are uploaded.
│ ├── Workflow.yml

└── README.md
```


## ⚙️ Requirements
- Python 3.13
- Main libraries:
  - PyTorch
  - Flask
  - Numpy
  - Torch
  - TorchVision


## 📊 Results
- **Final Accuracy:** 90.96%
- **Loss:** 0.1821
- **Training Time:** +-15 minutes with NVIDIA RTX 4070 SUPER  

- **Note**: The model’s performance can be improved. However, since this project serves as a proof of concept, it has not been fully optimized 


## 👤 Author
- **Name:** Ricardo Montenegro Dona
- **LinkedIn:** https://www.linkedin.com/in/ricardomontenegrodona/

