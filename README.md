# cGAN-MNIST

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/yourusername/cGAN-MNIST/releases)

This repository contains the code and configuration files for the cGAN-MNIST project. The project aims to generate MNIST digit images using a Conditional Generative Adversarial Network (cGAN).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Introduction

The cGAN-MNIST project addresses the challenge of generating realistic MNIST digit images conditioned on class labels. By leveraging a Conditional Generative Adversarial Network (cGAN), this project builds a model that can generate digit images corresponding to specific digit classes.

This project is structured into several key stages:

1. **Data Loading and Preprocessing**: Loading and preprocessing the MNIST dataset.
2. **Model Definition**: Defining the Generator and Discriminator models.
3. **Model Compilation**: Compiling the GAN model.
4. **Model Training**: Training the cGAN model on the MNIST dataset.
5. **Model Evaluation**: Evaluating the performance of the trained cGAN model.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Configuration

The configuration file `config.yaml` contains various settings and hyperparameters for the project. You can find this file in the root directory of the repository. Modify this file to adjust the behavior of the cGAN model.

## Usage

Run the main script:

```bash
python main.py
```

## Code Overview

- `main.py`: The main script to train the cGAN model.
- `check_gpu.py`: Utility to check GPU availability.
- `data_reader.py`: Module to read the MNIST dataset.
- `data_processor.py`: Module to process the MNIST dataset.
- `sample_processor.py`: Module to process samples.
- `performance_processor.py`: Module to evaluate model performance.
- `discriminator.py`: Defines the Discriminator model.
- `generator.py`: Defines the Generator model.
- `gan_compiler.py`: Compiles the GAN model.
- `gan_trainer.py`: Trains the GAN model.

## Model Architecture

### Discriminator

The discriminator neural network architecture in the provided code is designed to distinguish between real and generated images. This architecture was previously derived and optimised through a prior image classification project and then introduced into this GAN project. Here's a summary of its architecture and the purpose of each layer:

**Inputs:**
- **Image Input:** Shape (28, 28, 1) for grayscale images.
- **Label Input:** Shape (1) for class labels.

**Label Branch:**
- **Embedding:** Converts class labels into dense vectors of dimension 50.
- **Dense:** Transforms the embedding to match the image dimensions (28 * 28).
- **Reshape:** Reshapes the dense output to (28, 28, 1).

**Merge Branches:**
- **Concatenate:** Merges the image input and label branch along the channel dimension.

**Convolutional Layers:**
- **Conv2D with SpectralNormalization:** Applies convolution with spectral normalization to stabilize training.
- **LeakyReLU:** Activation function with a small slope for negative values (alpha=0.2).
- **BatchNormalization:** Normalizes the output of the previous layer to improve training stability.
- **MaxPooling2D:** Downsamples the feature maps by taking the maximum value over a 2x2 window.
- **SpatialDropout2D:** Randomly drops entire channels to prevent overfitting (rate=0.5).

This sequence is repeated with additional convolutional layers to extract hierarchical features from the input images.

**Flatten:** Converts the 2D feature maps into a 1D vector.

**Dense Layers:**
- **Dense with SpectralNormalization:** Fully connected layers with spectral normalization.
- **LeakyReLU:** Activation function.
- **BatchNormalization:** Normalizes the output.
- **Dropout:** Randomly drops units to prevent overfitting (rate=0.5).

This sequence is repeated with decreasing number of units (600, 400, 200, 20) to progressively reduce the dimensionality.

**Output Layer:**
- **Dense with SpectralNormalization:** Final dense layer with a single unit and sigmoid activation to output a probability.

**Compilation:**
- **Adam Optimizer:** Optimizes the model with a learning rate of 0.0001 and beta_1 of 0.5.
- **Binary Crossentropy Loss:** Measures the difference between the predicted and actual labels.
- **Accuracy Metric:** Evaluates the model's performance.

Each layer and technique used in this architecture serves to improve the model's ability to learn and generalize from the data while preventing overfitting and ensuring stable training.

### Generator

The generator neural network architecture in the provided code is designed to generate images from latent vectors and class labels. This architecture was previously derived and optimised through a prior image generation project and then introduced into this GAN project. Here's a summary of its architecture and the purpose of each layer:

**Inputs:**
- **Latent Input:** Shape (100) for the latent vector.
- **Label Input:** Shape (1) for class labels.

**Latent Branch:**
- **Dense with SpectralNormalization:** Fully connected layer with spectral normalization to stabilize training, outputting a vector of size (128 * 3 * 3).
- **LeakyReLU:** Activation function with a small slope for negative values (alpha=0.2).
- **BatchNormalization:** Normalizes the output to improve training stability.
- **Reshape:** Reshapes the dense output to (3, 3, 128).

**Label Branch:**
- **Embedding:** Converts class labels into dense vectors of dimension 50.
- **Dense with SpectralNormalization:** Fully connected layer with spectral normalization, outputting a vector of size (3 * 3).
- **LeakyReLU:** Activation function.
- **BatchNormalization:** Normalizes the output.
- **Reshape:** Reshapes the dense output to (3, 3, 1).

**Merge Branches:**
- **Concatenate:** Merges the latent branch and label branch along the channel dimension.

**Convolutional Layers:**
- **Conv2DTranspose with SpectralNormalization:** Applies transposed convolution with spectral normalization to upsample the feature maps.
- **LeakyReLU:** Activation function.
- **BatchNormalization:** Normalizes the output.

This sequence is repeated with additional transposed convolutional layers to progressively upsample the feature maps to the desired image size.

**Output Layer:**
- **Conv2D with SpectralNormalization:** Final convolutional layer with a single output channel and sigmoid activation to generate the image.

Each layer and technique used in this architecture serves to improve the model's ability to generate realistic images while ensuring stable training and preventing overfitting.

### Label Smoothing

Label smoothing was used as a regularization technique to improve the generalization of the discriminator in the GAN. Instead of using hard labels (0 for fake and 1 for real), label smoothing assigns a range of values close to 0 and 1. This helps to prevent the discriminator from becoming overconfident and provides a smoother gradient for the generator to learn from.

### Learning Rate Decay

Learning rate decay is implemented to gradually reduce the learning rate of the Adam optimizer during training. This helps in stabilizing the training process and achieving better convergence. In this project, the learning rate for both the discriminator and the generator is updated periodically to ensure that the models do not overshoot the optimal weights and to fine-tune the learning as training progresses.

## Model Evaluation

Evaluating a GAN throughout the training is typically trickier than a standard image recognition or image generation project. This is because GANs involve two models (the Generator and the Discriminator) that are trained simultaneously in a competitive setting, making it challenging to assess their performance independently.

Here we evaluate the following metrics:

1. **Inception Score (IS)**: Measures the quality and diversity of generated images. It is calculated by passing the generated images through a pre-trained Inception model and computing the KL divergence between the conditional label distribution and the marginal label distribution. A higher Inception Score indicates that the generated images are both high quality and diverse.
2. **Frechet Inception Distance (FID)**: Compares the distribution of generated images to real images. FID is calculated by passing both the generated images and real images through a pre-trained Inception model to obtain their feature representations. The mean and covariance of these features are then computed for both sets of images. FID measures the distance between these two distributions using the Frechet distance formula, which takes into account both the mean and covariance differences. A lower FID score indicates that the generated images are more similar to the real images in terms of feature distribution.
3. **Discriminator Loss**: Indicates how well the Discriminator is distinguishing between real and fake images.
4. **Generator Loss**: Indicates how well the Generator is fooling the Discriminator.
5. **Discriminator Real Samples Accuracy**: Measures the accuracy of the Discriminator in correctly identifying real MNIST digit images from the dataset.
6. **Discriminator Generated Samples Accuracy**: Measures the accuracy of the Discriminator in correctly identifying images generated by the Generator as fake.

Importantly, we produce sample plots at each epoch to visually inspect how the learning is proceeding. We can visually inspect if overfitting is occurring in the GAN or if the quality of image generation has plateaued. These visual inspections are crucial for understanding the qualitative aspects of the generated images and ensuring that the GAN is learning effectively.

## Model Results

### Generated Samples

Below are some examples of digit images generated by the cGAN model after training:

![Training Progression at 100 Epochs](outputs/generated_plot_e100.png)

### Discriminator and Generator Loss

The following plots shows the loss curves from tensorboard for both the Discriminator and the Generator during training:

| ![Loss Curves](outputs/tensorboard_discriminator_loss.png) | ![Loss Curves](outputs/tensorboard_generator_loss.png) |
|:-------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|

We observe that the training of a GAN does not show a monotonic learning curve. We can also observe from the discriminator loss that the learning has yet to plateau and more training is needed.

### Visual Inspection

The following plots shows the progression of generated images over training epochs, providing a visual inspection of the model's learning process through 3 different stages (10, 50 and 100 epoch) of training:

![Training Progression at 10 Epochs](outputs/generated_plot_e010.png)
![Training Progression at 50 Epochs](outputs/generated_plot_e050.png)
![Training Progression at 100 Epochs](outputs/generated_plot_e100.png)

These results demonstrate the effectiveness of the cGAN model in generating realistic MNIST digit images conditioned on class labels. However, we observe that we have stopped the training prematurely, as we can observe that some of the numbers are still nonsensical. The quantitative metrics and visual inspections provide a comprehensive evaluation of the model's performance.

## License

This project is licensed under the MIT License.