import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging as logger


class ModelCompiler(nn.Module):
    """
    A convolutional neural network (CNN) model compiler for the MNIST dataset.

    This class defines a CNN architecture and provides methods for compiling the model,
    including setting up the loss function and optimizer.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        conv1 (nn.Conv2d): The first convolutional layer. Applies a 2D convolution over an input signal composed of several input planes.
            - in_channels (int): Number of channels in the input image.
            - out_channels (int): Number of channels produced by the convolution.
            - kernel_size (int or tuple): Size of the convolving kernel. Determines the dimensions of the filter that slides over the input image.
            - stride (int or tuple, optional): Stride of the convolution. Determines the step size with which the filter moves across the input image. Default: 1
            - padding (int or tuple, optional): Zero-padding added to both sides of the input. Determines the amount of padding added to the input image to control the spatial dimensions of the output. Default: 0
            Purpose: Extracts features from the input image by applying convolution operations.
        conv2 (nn.Conv2d): The second convolutional layer. Similar to conv1 but with different parameters.
            Purpose: Further extracts features from the output of the first convolutional layer.
        pool (nn.MaxPool2d): The max pooling layer. Applies a 2D max pooling over an input signal composed of several input planes.
            - kernel_size (int or tuple): Size of the window to take a max over.
            - stride (int or tuple, optional): Stride of the window. Default: kernel_size
            - padding (int or tuple, optional): Implicit zero padding to be added on both sides. Default: 0
            Purpose: Reduces the spatial dimensions (width and height) of the input volume, reducing the number of parameters and computation in the network.
        fc1 (nn.Linear): The first fully connected layer. Applies a linear transformation to the incoming data.
            - in_features (int): Size of each input sample.
            - out_features (int): Size of each output sample.
            Purpose: Maps the flattened input to a higher-dimensional space, enabling the network to learn complex representations.
        fc2 (nn.Linear): The second fully connected layer. Similar to fc1 but with different parameters.
            Purpose: Maps the output of the first fully connected layer to the final output classes.
        dropout (nn.Dropout): The dropout layer. Randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
            - p (float, optional): Probability of an element to be zeroed. Default: 0.5
            Purpose: Prevents overfitting by randomly setting some of the activations to zero during training.
        criterion (nn.CrossEntropyLoss): The loss function. Measures the performance of a classification model whose output is a probability value between 0 and 1.
            Purpose: Computes the loss between the predicted output and the true labels, guiding the optimization process.
        optimizer (optim.Adam): The optimizer. Implements the Adam algorithm.
            Purpose: Updates the model parameters based on the computed gradients to minimize the loss.

    Methods:
        forward(x):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor.
            Purpose: Defines how the input data passes through the network layers to produce the output.

        compile():
            Sets up the loss function and optimizer.
            Purpose: Initializes the loss function and optimizer for training the model.
    """

    def __init__(self, config):
        # Save the configuration
        self.learning_rate = config.model.learning_rate
        logger.info(
            f"Initializing ModelCompiler with learning rate: {self.learning_rate}"
        )

        # Initialize the parent class
        super(ModelCompiler, self).__init__()

        # Initialize the layers of the CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        logger.info("Model layers initialized")

    def forward(self, x):
        # Define the forward pass
        logger.debug("Starting forward pass")
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        logger.debug("Forward pass completed")
        return x

    def compile(self):
        # Set up the loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        logger.info("Model compiled with CrossEntropyLoss and Adam optimizer")
