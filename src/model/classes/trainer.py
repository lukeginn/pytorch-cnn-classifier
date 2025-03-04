import numpy as np
from keras.models import Model
from dataclasses import dataclass, field
from typing import Tuple, Any
from src.utils.tensorboard import setup_tensorboard, log_metrics_to_tensorboard


@dataclass
class GANTrainer:
    generator_model: Model
    discriminator_model: Model
    gan_model: Model
    sample_processor: Any
    performance_processor: Any
    images: np.ndarray
    labels: np.ndarray
    latent_dim: int
    start_epoch: int = 0
    num_epochs: int = 100
    num_batch: int = 256
    learning_rate_decay_run: bool = True
    learning_rate_decay_rate: float = 0.8
    learning_rate_decay_epoch: int = 10

    def train_model(self) -> None:
        """Train the GAN model."""
        bat_per_epo = int(self.images.shape[0] / self.num_batch)
        half_batch = int(self.num_batch / 2)

        tensorboard_callback, file_writer = setup_tensorboard()

        for epoch in range(self.start_epoch, self.num_epochs):
            for batch in range(bat_per_epo):
                self._train_discriminator(half_batch)
                self._train_generator()

                print(f"Epoch: {epoch + 1}, Batch_No: {batch + 1}/{bat_per_epo}")

            acc_real, acc_fake, inception_score, fid_score = (
                self.performance_processor.summarize_performance(
                    epoch,
                    self.generator_model,
                    self.discriminator_model,
                    self.g_loss,
                    self.d_loss,
                    self.images,
                    self.labels,
                    self.latent_dim,
                )
            )

            # Update learning rate
            new_lr = self._scheduler(epoch, self.gan_model.optimizer.lr.numpy())
            self._update_learning_rate(new_lr)

            log_metrics_to_tensorboard(
                epoch,
                self.g_loss,
                self.d_loss,
                acc_real,
                acc_fake,
                inception_score,
                fid_score,
                new_lr,
                tensorboard_callback,
                file_writer,
            )

    def get_models(self) -> Tuple[Model, Model, Model]:
        """Return the generator, discriminator, and GAN models."""
        return self.generator_model, self.discriminator_model, self.gan_model

    def _train_discriminator(self, half_batch: int) -> None:
        """Train the discriminator model."""
        [X_real, X_real_labels], y_real = self.sample_processor.real_samples(
            self.images, self.labels, half_batch
        )
        [X_fake, X_fake_labels], y_fake = self.sample_processor.fake_samples(
            self.generator_model, self.latent_dim, half_batch
        )

        X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
        X_labels = np.concatenate((X_real_labels, X_fake_labels))

        self.d_loss, _ = self.discriminator_model.train_on_batch([X, X_labels], y)

    def _train_generator(self) -> None:
        """Train the generator model."""
        [X_gan, X_gan_labels] = self.sample_processor.latent_points(
            self.num_batch, self.latent_dim
        )
        y_gan = np.ones((self.num_batch, 1))

        self.g_loss, _ = self.gan_model.train_on_batch([X_gan, X_gan_labels], y_gan)

    def _scheduler(self, epoch, lr):
        if self.learning_rate_decay_run:
            if epoch % self.learning_rate_decay_epoch == 0 and epoch != 0:
                return lr * self.learning_rate_decay_rate
        return lr

    def _update_learning_rate(self, new_lr: float) -> None:
        """Update the learning rate for all models."""
        self.discriminator_model.optimizer.lr.assign(new_lr)
        self.gan_model.optimizer.lr.assign(new_lr)
