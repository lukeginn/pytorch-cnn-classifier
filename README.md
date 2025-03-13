# PyTorch-CNN-MNIST

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/yourusername/PyTorch-CNN-MNIST/releases)

This repository contains the code and configuration files for the PyTorch-CNN-MNIST project. The project aims to classify MNIST digit images using a Convolutional Neural Network (CNN) implemented in PyTorch.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Model Architecture](#model-architecture)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Introduction

The PyTorch-CNN-MNIST project addresses the challenge of classifying MNIST digit images. By leveraging a Convolutional Neural Network (CNN), this project builds a model that can accurately classify digit images from the MNIST dataset.

This project is structured into several key stages:

1. **Data Loading and Preprocessing**: Loading and preprocessing the MNIST dataset.
2. **Model Definition**: Defining the CNN model.
3. **Model Compilation**: Compiling the CNN model.
4. **Model Training**: Training the CNN model on the MNIST dataset.
5. **Model Evaluation**: Evaluating the performance of the trained CNN model.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Configuration

The configuration file `config.yaml` contains various settings and hyperparameters for the project. You can find this file in the `config` directory of the repository. Modify this file to adjust the behavior of the CNN model.

## Usage

Run the main script:

```bash
python main.py
```

## Code Overview

- `main.py`: The main script to run the data processing and model pipeline.
- `config/paths.py`: Defines paths used in the project.
- `src/data/reader.py`: Module to read the MNIST dataset.
- `src/data/processor.py`: Module to process the MNIST dataset.
- `src/model/classes/compiler.py`: Compiles the CNN model.
- `src/model/classes/trainer.py`: Trains the CNN model.
- `src/model/classes/tuner.py`: Tunes hyperparameters using cross-validation.
- `src/utils/setup.py`: Sets up paths and configurations.
- `src/utils/check_gpu.py`: Utility to check GPU availability.
- `src/utils/config.py`: Utility to read the configuration file.

## Model Architecture

The CNN model architecture in this project is designed to classify MNIST digit images. Here's a summary of its architecture and the purpose of each layer:

### Convolutional Layers

- **Conv2D**: Applies convolution to the input image.
- **BatchNorm2D**: Normalizes the output of the convolutional layer.
- **MaxPool2D**: Downsamples the feature maps by taking the maximum value over a window.

### Fully Connected Layers

- **Linear**: Fully connected layers to reduce the dimensionality.
- **LayerNorm**: Normalizes the output of the fully connected layer.
- **Dropout**: Randomly drops units to prevent overfitting.

### Activation Function

- **Tanh**: Activation function used in the model.

### Optimizer

- **Adam**: Optimizes the model with a learning rate of 0.001.

## Model Evaluation

The model is evaluated using various metrics such as accuracy, precision, recall, and F1-score. The evaluation is performed on both the training and test datasets.

## License

This project is licensed under the MIT License.