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
        pool1 (nn.MaxPool2d): The first max pooling layer. Applies a 2D max pooling over an input signal composed of several input planes.
            - kernel_size (int or tuple): Size of the window to take a max over.
            - stride (int or tuple, optional): Stride of the window. Default: kernel_size
            - padding (int or tuple, optional): Implicit zero padding to be added on both sides. Default: 0
            Purpose: Reduces the spatial dimensions (width and height) of the input volume, reducing the number of parameters and computation in the network.
        pool2 (nn.MaxPool2d): The second max pooling layer. Similar to pool1 but with different parameters.
            Purpose: Further reduces the spatial dimensions of the input volume.
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
        optimizer (optim.Optimizer): The optimizer. Implements the selected optimization algorithm.
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
        self.optimizer_name = config.model.optimizer
        self.conv1_in_channels = config.model.conv1.in_channels
        self.conv1_out_channels = config.model.conv1.out_channels
        self.conv1_kernel_size = config.model.conv1.kernel_size
        self.conv1_stride = config.model.conv1.stride
        self.conv1_padding = config.model.conv1.padding
        self.pool1_kernel_size = config.model.pool1.kernel_size
        self.pool1_stride = config.model.pool1.stride
        self.pool1_padding = config.model.pool1.padding

        self.conv2_in_channels = config.model.conv2.in_channels
        self.conv2_out_channels = config.model.conv2.out_channels
        self.conv2_kernel_size = config.model.conv2.kernel_size
        self.conv2_stride = config.model.conv2.stride
        self.conv2_padding = config.model.conv2.padding
        self.pool2_kernel_size = config.model.pool2.kernel_size
        self.pool2_stride = config.model.pool2.stride
        self.pool2_padding = config.model.pool2.padding

        self.view_shape_channels = config.model.view_shape.channels
        self.view_shape_height = config.model.view_shape.height
        self.view_shape_width = config.model.view_shape.width
        self.fc1_in_features = (
            config.model.view_shape.channels
            * config.model.view_shape.height
            * config.model.view_shape.width
        )
        self.fc1_out_features = config.model.fc1.out_features
        self.fc2_in_features = config.model.fc2.in_features
        self.fc2_out_features = config.model.fc2.out_features
        self.dropout_p = config.model.dropout.p

        # Set up the activation function based on the configuration
        if config.model.activation_function == "relu":
            self.activation_function = F.relu
        elif config.model.activation_function == "sigmoid":
            self.activation_function = F.sigmoid
        elif config.model.activation_function == "tanh":
            self.activation_function = F.tanh
        else:
            raise ValueError(
                f"Unsupported activation function: {config.model.activation_function}"
            )

        logger.info("Model configuration loaded")

        # Initialize the parent class
        super(ModelCompiler, self).__init__()

        # Initialize the layers of the CNN using config parameters
        self.conv1 = nn.Conv2d(
            in_channels=self.conv1_in_channels,
            out_channels=self.conv1_out_channels,
            kernel_size=self.conv1_kernel_size,
            stride=self.conv1_stride,
            padding=self.conv1_padding,
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=self.pool1_kernel_size,
            stride=self.pool1_stride,
            padding=self.pool1_padding,
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.conv2_in_channels,
            out_channels=self.conv2_out_channels,
            kernel_size=self.conv2_kernel_size,
            stride=self.conv2_stride,
            padding=self.conv2_padding,
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=self.pool2_kernel_size,
            stride=self.pool2_stride,
            padding=self.pool2_padding,
        )

        self.fc1 = nn.Linear(
            in_features=self.fc1_in_features,
            out_features=self.fc1_out_features,
        )
        self.fc2 = nn.Linear(
            in_features=self.fc2_in_features,
            out_features=self.fc2_out_features,
        )
        self.dropout = nn.Dropout(p=self.dropout_p)
        logger.info("Model layers initialized")

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Purpose:
            Defines how the input data passes through the network layers to produce the output.
            The `view` function is used to reshape the tensor `x` before passing it to the fully connected layer `fc1`.
            This reshaping converts the 4D tensor output from the convolutional layers (batch size, channels, height, width)
            into a 2D tensor (batch size, number of features) that can be processed by the fully connected layers.
            The `-1` argument tells PyTorch to infer the size of this dimension automatically based on the other dimensions
            and the total number of elements in the tensor.
        """
        logger.debug("Starting forward pass")
        x = self.pool1(self.activation_function(self.conv1(x)))
        x = self.pool2(self.activation_function(self.conv2(x)))
        x = x.view(
            -1,
            self.view_shape_channels * self.view_shape_height * self.view_shape_width,
        )
        x = self.activation_function(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        logger.debug("Forward pass completed")
        return x

    def compile(self):
        # Set up the loss function
        self.criterion = nn.CrossEntropyLoss()
        logger.info("Loss function set to CrossEntropyLoss")

        # Set up the optimizer based on the configuration
        if self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "RMSprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        logger.info(
            f"Model compiled with {self.optimizer_name} optimizer and learning rate: {self.learning_rate}"
        )
