import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging as logger
import torch
from torchviz import make_dot
from config.paths import Paths


class ModelCompiler(nn.Module):
    """
    A convolutional neural network (CNN) model compiler for the MNIST dataset.

    This class defines a CNN architecture and provides methods for compiling the model,
    including setting up the loss function and optimizer.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        optimizer_name (str): The name of the optimizer.
        conv_layers (nn.ModuleList): List of convolutional layers.
        pool_layers (nn.ModuleList): List of pooling layers.
        fc_layers (nn.ModuleList): List of fully connected layers.
        dropouts (nn.ModuleList): List of dropout layers.
        criterion (nn.CrossEntropyLoss): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        activation_function (callable): The activation function.
        view_shape_channels (int): Number of channels in the input view shape.
        view_shape_height (int): Height of the input view shape.
        view_shape_width (int): Width of the input view shape.
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
        """
        logger.debug("Starting forward pass")
        x = self._apply_conv_and_pool_layers(x)
        x = self._flatten(x)
        x = self._apply_fc_layers(x)
        logger.debug("Forward pass completed")
        return x

    def compile(self):
        """
        Sets up the loss function and optimizer.
        """
        self._set_loss_function()
        self._set_optimizer()
        logger.info(
            f"Model compiled with {self.optimizer_name} optimizer and learning rate: {self.learning_rate}"
        )

    def visualize(self):
        """
        Visualizes the model architecture.

        Args:
            input_size (tuple): The size of the input tensor (e.g., (1, 1, 28, 28) for MNIST).
        """
        x = torch.randn((1, 1, 28, 28))
        y = self.forward(x)

        dot = make_dot(y, params=dict(self.named_parameters()))

        # Customize graph attributes
        dot.graph_attr.update(dpi="300")  # Left to Right layout and increase DPI
        dot.node_attr.update(shape="box", style="filled", fillcolor="lightblue")
        dot.edge_attr.update(color="gray")

        path = str(Paths.MODEL_ARCHITECTURE_PATH.value).rsplit(".", 1)[0]
        dot.render(path, format="png")

        logger.info(
            f"Model architecture saved to {Paths.MODEL_ARCHITECTURE_PATH.value}"
        )

    def _load_config(self, config):
        self.learning_rate = config.model.learning_rate
        self.optimizer_name = config.model.optimizer

        self.conv_layers_config = config.model.conv_layers
        self.pool_layers_config = config.model.pool_layers

        self.view_shape_channels = config.model.view_shape.channels
        self.view_shape_height = config.model.view_shape.height
        self.view_shape_width = config.model.view_shape.width
        self.fc_layers_config = config.model.fc_layers

    def _initialize_layers(self):
        self.conv_layers = nn.ModuleList()
        for conv_layer in self.conv_layers_config:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=conv_layer["in_channels"],
                    out_channels=conv_layer["out_channels"],
                    kernel_size=conv_layer["kernel_size"],
                    stride=conv_layer["stride"],
                    padding=conv_layer["padding"],
                )
            )

        self.pool_layers = nn.ModuleList()
        for pool_layer in self.pool_layers_config:
            self.pool_layers.append(
                nn.MaxPool2d(
                    kernel_size=pool_layer["kernel_size"],
                    stride=pool_layer["stride"],
                    padding=pool_layer["padding"],
                )
            )

        self.fc_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_features = (
            self.view_shape_channels * self.view_shape_height * self.view_shape_width
        )
        for fc_layer in self.fc_layers_config:
            self.fc_layers.append(
                nn.Linear(
                    in_features=in_features, out_features=fc_layer["out_features"]
                )
            )
            in_features = fc_layer["out_features"]
            self.dropouts.append(
                nn.Dropout(p=fc_layer.get("dropout", 0.0))
            )  # Default to 0.0 if not specified

        logger.info("Model layers initialized")

    def _set_activation_function(self, activation_function):
        if activation_function == "relu":
            self.activation_function = F.relu
        elif activation_function == "leaky_relu":
            self.activation_function = F.leaky_relu
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
        for i, (fc_layer, dropout) in enumerate(zip(self.fc_layers, self.dropouts)):
            x = self.activation_function(fc_layer(x))
            if i < len(self.fc_layers) - 1:  # Apply dropout to all but the last layer
                x = dropout(x)
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
