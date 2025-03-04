from keras.models import Model
from keras.layers import (
    Dense,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    LeakyReLU,
    Reshape,
    Input,
    Embedding,
    Concatenate,
)
from tensorflow_addons.layers import SpectralNormalization
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Generator:
    n_classes: int = 10
    batch_size: int = 128
    latent_dim: int = 100
    embedding_dim: int = 50
    LeakyReLU_alpha: float = 0.2

    model: Model = field(init=False)

    def __post_init__(self):
        self.model = self.build_model()

    def build_model(self) -> Model:
        latent_input, label_input = self._create_inputs()
        latent_branch = self._create_latent_branch(latent_input)
        label_branch = self._create_label_branch(label_input)
        neural_net = self._merge_branches(latent_branch, label_branch)
        neural_net = self._add_conv_layers(neural_net)
        model = Model([latent_input, label_input], neural_net)
        return model

    def get_model(self) -> Model:
        """Return the compiled model."""
        return self.model

    def _create_inputs(self) -> Tuple[Input, Input]:
        """Create input layers for the latent and label branches."""
        latent_input = Input(shape=(self.latent_dim,))
        label_input = Input(shape=(1,))
        return latent_input, label_input

    def _create_latent_branch(self, latent_input: Input) -> Model:
        """Create the latent branch of the generator."""
        latent_branch = SpectralNormalization(Dense(self.batch_size * 3 * 3))(
            latent_input
        )
        latent_branch = LeakyReLU(alpha=self.LeakyReLU_alpha)(latent_branch)
        latent_branch = BatchNormalization()(latent_branch)
        latent_branch = Reshape((3, 3, self.batch_size))(latent_branch)
        return latent_branch

    def _create_label_branch(self, label_input: Input) -> Model:
        """Create the label branch of the generator."""
        label_branch = Embedding(self.n_classes, self.embedding_dim)(label_input)
        label_branch = SpectralNormalization(Dense(3 * 3))(label_branch)
        label_branch = LeakyReLU(alpha=self.LeakyReLU_alpha)(label_branch)
        label_branch = BatchNormalization()(label_branch)
        label_branch = Reshape((3, 3, 1))(label_branch)
        return label_branch

    def _merge_branches(self, latent_branch: Model, label_branch: Model) -> Model:
        """Merge the latent and label branches."""
        return Concatenate()([latent_branch, label_branch])

    def _add_conv_layers(self, neural_net: Model) -> Model:
        """Add convolutional layers to the neural network."""
        neural_net = SpectralNormalization(
            Conv2DTranspose(self.batch_size, (4, 4), strides=(1, 1))
        )(neural_net)
        neural_net = LeakyReLU(alpha=self.LeakyReLU_alpha)(neural_net)
        neural_net = BatchNormalization()(neural_net)
        neural_net = SpectralNormalization(
            Conv2DTranspose(self.batch_size, (4, 4), strides=(2, 2))
        )(neural_net)
        neural_net = LeakyReLU(alpha=self.LeakyReLU_alpha)(neural_net)
        neural_net = BatchNormalization()(neural_net)
        neural_net = SpectralNormalization(
            Conv2DTranspose(self.batch_size, (5, 5), strides=(2, 2), padding="same")
        )(neural_net)
        neural_net = LeakyReLU(alpha=self.LeakyReLU_alpha)(neural_net)
        neural_net = BatchNormalization()(neural_net)
        neural_net = SpectralNormalization(
            Conv2D(1, (10, 10), activation="sigmoid", padding="same")
        )(neural_net)
        return neural_net
