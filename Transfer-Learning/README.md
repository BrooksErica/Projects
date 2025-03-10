# Character Recognition using Transfer Learning on CNN

## Overview

This project implements a convolutional neural network (CNN) to recognize handwritten characters (A-E) using transfer learning. The model is initially trained on my own handwritten MNIST dataset and then fine-tuned on a custom dataset I created containing characters A, B, C, D, and E. The goal is to leverage knowledge from digit classification to improve character recognition performance.  


### Dataset

1. MNIST Dataset

The model is first trained on a my MNIST dataset, which consists of handwritten digits (0-9).

It is used to pre-train the CNN before fine-tuning it for character recognition.

2. Custom Character Dataset

My dataset containing images of letters A-E is used for fine-tuning.

Images are loaded in grayscale and resized to 28x28 pixels to match the MNIST format.  



### Model Architecture

The model is a Convolutional Neural Network (CNN) with the following structure:

Convolutional Layer 1: 32 filters, kernel size 3x3, ReLU activation.

Max Pooling Layer 1: Pool size 2x2.

Convolutional Layer 2: 64 filters, kernel size 3x3, ReLU activation.

Max Pooling Layer 2: Pool size 2x2.

Fully Connected Layer 1: 128 neurons, ReLU activation.

Fully Connected Layer 2: Output layer with 10 neurons (for MNIST) initially, later modified to 5 neurons (for character dataset).  



### Process

1. Data Preprocessing

Images are normalized to match the MNIST dataset statistics (mean=0.1307, std=0.3081).

Grayscale conversion is applied to the character dataset images.

Images are resized to 28x28 pixels.

2. Training on MNIST

The model is trained on MNIST for 5 epochs using a batch size of 64.

The Adam optimizer is used with a learning rate of 0.001.

The loss function is Cross-Entropy Loss.

3. Fine-tuning on Custom Dataset

The last fully connected layer is modified to have 5 output neurons (for A-E classification).

The model is fine-tuned on the character dataset for 10 epochs with a reduced learning rate of 0.0001.

The Adam optimizer is used for optimization.

4. Evaluation and Visualization

Sample images from the datasets are displayed before training.

Model predictions on character images are visualized after fine-tuning.  



### Results

The model successfully learns to recognize characters A-E after transfer learning.

Predictions are displayed with true and predicted labels.

The trained model weights are saved.
