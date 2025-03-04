import numpy as np
from keras.models import Model
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class SampleProcessor:
    n_classes: int = 10
    label_smoothing: bool = True
    label_smoothing_degree: float = 0.1

    def real_samples(
        self, images: np.ndarray, labels: np.ndarray, n_samples: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate a batch of real samples with class labels."""
        ix = np.random.randint(0, images.shape[0], n_samples)
        X = images[ix]
        labels = labels[ix]

        if self.label_smoothing:
            y = np.ones((n_samples, 1)) * (
                1
                - self.label_smoothing_degree
                + self.label_smoothing_degree * np.random.rand(n_samples, 1)
            )
        else:
            y = np.ones((n_samples, 1))

        return [X, labels], y

    def latent_points(
        self, n_samples: int, latent_dim: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate points in the latent space and random labels."""
        # generate points in the latent space
        x_input = np.random.randn(latent_dim * n_samples)

        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)

        # generate random labels
        labels = np.random.randint(0, self.n_classes, n_samples)
        return x_input, labels

    def fake_samples(
        self, generator_model: Model, latent_dim: int, n_samples: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate a batch of fake samples using the generator model."""
        # generate points in latent space
        x_input, labels = self.latent_points(n_samples, latent_dim)

        # predict outputs
        X = generator_model.predict([x_input, labels])

        # create 'fake' class labels (0)
        if self.label_smoothing:
            y = np.zeros((n_samples, 1)) + self.label_smoothing_degree * np.random.rand(
                n_samples, 1
            )
        else:
            y = np.zeros((n_samples, 1))

        return [X, labels], y
