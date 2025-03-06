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
        conv_layers (nn.ModuleList): List of convolutional layers.
        pool_layers (nn.ModuleList): List of pooling layers.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        dropout (nn.Dropout): The dropout layer.
        criterion (nn.CrossEntropyLoss): The loss function.
        optimizer (optim.Optimizer): The optimizer.

    Methods:
        forward(x):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor.
        compile():
            Sets up the loss function and optimizer.
    """

    def __init__(self, config):
        super(ModelCompiler, self).__init__()
        self._load_config(config)
        self._initialize_layers()
        self._set_activation_function(config.model.activation_function)
        logger.info("Model configuration loaded")

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
        x = self._apply_conv_and_pool_layers(x)
        x = self._flatten(x)
        x = self._apply_fc_layers(x)
        logger.debug("Forward pass completed")
        return x

    def compile(self):
        self._set_loss_function()
        self._set_optimizer()
        logger.info(
            f"Model compiled with {self.optimizer_name} optimizer and learning rate: {self.learning_rate}"
        )

    def _load_config(self, config):
        self.learning_rate = config.model.learning_rate
        self.optimizer_name = config.model.optimizer

        self.conv_layers_config = config.model.conv_layers
        self.pool_layers_config = config.model.pool_layers

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

    def _initialize_layers(self):
        self.conv_layers = nn.ModuleList()
        for conv_layer in self.conv_layers_config:
            self.conv_layers.append(nn.Conv2d(
                in_channels=conv_layer['in_channels'],
                out_channels=conv_layer['out_channels'],
                kernel_size=conv_layer['kernel_size'],
                stride=conv_layer['stride'],
                padding=conv_layer['padding']
            ))

        self.pool_layers = nn.ModuleList()
        for pool_layer in self.pool_layers_config:
            self.pool_layers.append(nn.MaxPool2d(
                kernel_size=pool_layer['kernel_size'],
                stride=pool_layer['stride'],
                padding=pool_layer['padding']
            ))

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

    def _set_activation_function(self, activation_function):
        if activation_function == "relu":
            self.activation_function = F.relu
        elif activation_function == "sigmoid":
            self.activation_function = F.sigmoid
        elif activation_function == "tanh":
            self.activation_function = F.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
        
    def _apply_conv_and_pool_layers(self, x):
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            x = pool_layer(self.activation_function(conv_layer(x)))
        return x

    def _flatten(self, x):
        x = x.view(
            -1,
            self.view_shape_channels * self.view_shape_height * self.view_shape_width,
        )
        return x

    def _apply_fc_layers(self, x):
        x = self.activation_function(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _set_loss_function(self):
        self.criterion = nn.CrossEntropyLoss()
        logger.info("Loss function set to CrossEntropyLoss")

    def _set_optimizer(self):
        if self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "RMSprop":
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")


