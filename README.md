# MNIST Digit Classification with PyTorch

A deep learning project that achieves high accuracy in classifying handwritten digits from the MNIST dataset using PyTorch.

## Project Overview

This project implements a neural network classifier for the MNIST handwritten digit dataset. The model demonstrates excellent performance with:

- Training Accuracy: 98.03%
- Test Accuracy: 97.85%

## Dataset

The MNIST dataset consists of:
- 60,000 training images
- 10,000 test images
- 28x28 pixel grayscale images of handwritten digits (0-9)
- Normalized pixel values (0-1)

## Model Architecture

The classifier uses a fully connected neural network with:

- Input layer: 784 neurons (28x28 flattened images)
- Hidden layer 1: 128 neurons with ReLU activation and 0.2 dropout
- Hidden layer 2: 64 neurons with ReLU activation and 0.2 dropout
- Output layer: 10 neurons (one for each digit)

## Training Process

- Optimizer: Adam (learning rate = 0.001)
- Loss Function: Cross Entropy Loss
- Epochs: 10
- Batch Processing: Mini-batch training

## Features

- Data preprocessing and normalization
- Model training with accuracy tracking
- Comprehensive evaluation metrics
- Confusion matrix visualization
- Per-class performance analysis

## Project Structure

```
├── mnist_classifier.py  # Neural network model definition
├── mnist_loader.py      # Data loading utilities
├── mnist_kaggle.ipynb   # Training and evaluation notebook
└── README.md           # Project documentation
```

## Requirements

- PyTorch
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Results

The model achieves excellent performance across all digit classes, with detailed metrics including:
- Confusion matrix visualization
- Per-class precision, recall, and F1-scores
- Overall accuracy of 97.85% on the test set

## Usage

1. Clone the repository
2. Install the required dependencies
3. Run the Jupyter notebook or use the Python scripts directly
4. Train the model and evaluate its performance

## License

This project is open-source and available under the MIT License.