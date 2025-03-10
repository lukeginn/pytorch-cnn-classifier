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
        self._set_activation_function()
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
        x = self._apply_conv_layers(x)
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
        Visualizes the model architecture with tensor shapes.
        """
        x = torch.randn((1, 1, 28, 28))
        y = self.forward(x)

        dot = self._create_dot_graph(y)
        self._customize_graph(dot)
        self._add_shapes_to_nodes(dot)

        path = str(Paths.MODEL_ARCHITECTURE_PATH.value).rsplit(".", 1)[0]
        dot.render(path, format="png")

        logger.info(
            f"Model architecture saved to {Paths.MODEL_ARCHITECTURE_PATH.value}"
        )

    def _load_config(self, config):
        self.learning_rate = config.model.learning_rate
        self.optimizer_name = config.model.optimizer
        self.activation_function_name = config.model.activation_function

        self.conv_layers_config = config.model.conv_layers
        self.view_shape_channels = config.model.view_shape.channels
        self.view_shape_height = config.model.view_shape.height
        self.view_shape_width = config.model.view_shape.width
        self.fc_layers_config = config.model.fc_layers

    def _initialize_layers(self):
        # Initialize fully connected layers
        self.fc_layers = nn.ModuleList()
        for layer_cfg in self.fc_layers_config:
            layer_type = layer_cfg['type']
            if layer_type == 'Linear':
                self.fc_layers.append(nn.Linear(layer_cfg['in_features'], layer_cfg['out_features']))
            elif layer_type == 'Dropout':
                self.fc_layers.append(nn.Dropout(layer_cfg['p']))
            elif layer_type == 'BatchNorm1d':
                self.fc_layers.append(nn.BatchNorm1d(layer_cfg['num_features']))
            elif layer_type == 'LayerNorm':
                self.fc_layers.append(nn.LayerNorm(layer_cfg['normalized_shape']))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        # Initialize deconvolutional layers
        self.conv_layers = nn.ModuleList()
        for layer_cfg in self.conv_layers_config:
            layer_type = layer_cfg['type']
            if layer_type == 'ConvTranspose2d':
                self.conv_layers.append(nn.ConvTranspose2d(
                    layer_cfg['in_channels'],
                    layer_cfg['out_channels'],
                    kernel_size=layer_cfg['kernel_size'],
                    stride=layer_cfg['stride'],
                    padding=layer_cfg['padding']
                ))
            elif layer_type == 'Dropout2d':
                self.conv_layers.append(nn.Dropout2d(layer_cfg['p']))
            elif layer_type == 'BatchNorm2d':
                self.conv_layers.append(nn.BatchNorm2d(layer_cfg['num_features']))
            elif layer_type == 'MaxPool2d':
                self.conv_layers.append(nn.MaxPool2d(kernel_size=layer_cfg['kernel_size'], stride=layer_cfg['stride'], padding=layer_cfg['padding']))
            elif layer_type == 'AvgPool2d':
                self.conv_layers.append(nn.AvgPool2d(kernel_size=layer_cfg['kernel_size'], stride=layer_cfg['stride'], padding=layer_cfg['padding']))
            elif layer_type == 'Conv2d':
                self.conv_layers.append(nn.Conv2d(
                    layer_cfg['in_channels'],
                    layer_cfg['out_channels'],
                    kernel_size=layer_cfg['kernel_size'],
                    stride=layer_cfg['stride'],
                    padding=layer_cfg['padding']
                ))
            elif layer_type == 'PixelShuffle':
                self.deconv_layers.append(nn.PixelShuffle(layer_cfg['upscale_factor']))
            elif layer_type == 'InstanceNorm2d':
                self.deconv_layers.append(nn.InstanceNorm2d(layer_cfg['num_features']))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

    def _set_activation_function(self):
        if self.activation_function_name == "relu":
            self.activation_function = F.relu
        elif self.activation_function_name == "leaky_relu":
            self.activation_function = F.leaky_relu
        elif self.activation_function_name == "sigmoid":
            self.activation_function = F.sigmoid
        elif self.activation_function_name == "tanh":
            self.activation_function = F.tanh
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function_name}")
    
    def _apply_conv_layers(self, x):
        for layer in self.conv_layers:
            x = self.activation_function(layer(x))
        return x

    def _flatten(self, x):
        x = x.view(
            -1,
            self.view_shape_channels * self.view_shape_height * self.view_shape_width,
        )
        return x
    
    def _apply_fc_layers(self, x):
        for layer in self.fc_layers:
            x = self.activation_function(layer(x))
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

    def _create_dot_graph(self, y):
        return make_dot(y, params=dict(self.named_parameters()))

    def _customize_graph(self, dot):
        dot.graph_attr.update(dpi="300")
        dot.node_attr.update(shape="box", style="filled", fillcolor="lightblue")
        dot.edge_attr.update(color="gray")

    def _add_shapes_to_nodes(self, dot):
        for layer in self.modules():
            if (
                isinstance(layer, (nn.Conv2d, nn.Linear))
                and "shape_str" in layer.__dict__
            ):
                node = dot.node(str(id(layer)))
                if node:
                    node.attr["label"] += f"\n{layer.shape_str}"
