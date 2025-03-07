import logging as logger
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import wandb
from sklearn.model_selection import KFold
import numpy as np
from src.model.classes.trainer_utils import ModelTrainerUtils

# A bug was found in the numpy library that causes the int and bool types to be overwritten.
# This code snippet is a workaround to fix the issue.
np.int = int
np.bool = bool


class ModelTrainer:
    """
    A class to handle the training and evaluation of a convolutional neural network (CNN) model for the MNIST dataset.

    This class provides methods to train the model on the training dataset and evaluate its performance on the test dataset.

    Attributes:
        batch_size (int): The number of samples per batch to load.
        epochs (int): The number of times to iterate over the training dataset.
        model (nn.Module): The CNN model to be trained and evaluated.
        device (torch.device): The device on which to perform computations (CPU or GPU).

    Methods:
        train(train_images, train_labels):
            Trains the model on the provided training dataset.
            Args:
                train_images (numpy.ndarray): The training images.
                train_labels (numpy.ndarray): The training labels.
            Purpose: Performs the training loop, including forward pass, loss computation, backward pass, and optimizer step.

        evaluate(test_images, test_labels):
            Evaluates the model on the provided test dataset.
            Args:
                test_images (numpy.ndarray): The test images.
                test_labels (numpy.ndarray): The test labels.
            Purpose: Computes the accuracy of the model on the test dataset.
    """

    def __init__(self, model, config, log_to_wandb):
        """
        Initializes the ModelTrainer with the model and configuration.

        Args:
            model (nn.Module): The CNN model to be trained and evaluated.
            config (object): The configuration object containing training parameters.
        """
        self.config = ModelTrainerUtils.initialize_config(config)
        self.model, self.device = ModelTrainerUtils.initialize_model(model)
        self.config = ModelTrainerUtils.initialize_logging(config)

        self.log_to_wandb = self.config["logging"]["log_to_wandb"]
        self.batch_size = self.config["model"]["batch_size"]
        self.epochs = self.config["model"]["epochs"]
        self.training_shuffle = self.config["model"]["shuffle"]
        self.evaluation_shuffle = self.config["evaluation"]["shuffle"]
        self.evaluation_frequency = self.config["evaluation"]["epoch_frequency"]
        self.k_folds = config["cross_validation"]["k_folds"]
        self.cross_validation_shuffle = config["cross_validation"]["shuffle"]

    def train(self, train_images, train_labels, test_images, test_labels):
        """
        Trains the model on the provided training dataset.

        Args:
            train_images (numpy.ndarray): The training images.
            train_labels (numpy.ndarray): The training labels.
            test_images (numpy.ndarray): The test images for evaluation.
            test_labels (numpy.ndarray): The test labels for evaluation.

        Purpose:
            - Sets the model to training mode.
            - Creates a DataLoader for the training dataset.
            - Iterates over the dataset for the specified number of epochs.
            - For each batch, performs the following steps:
                - Moves the images and labels to the appropriate device (CPU or GPU).
                - Computes the loss between the predictions and the true labels.
                - Performs a backward pass to compute the gradients.
                - Updates the model's parameters using the optimizer.
                - Accumulates the running loss for monitoring.
            - Logs the average loss for each epoch.
        """
        self.model.train()
        data_loader = ModelTrainerUtils.create_data_loader(
            train_images, train_labels, self.batch_size, self.training_shuffle
        )

        for epoch in range(self.epochs):
            avg_loss = ModelTrainerUtils.train_one_epoch(
                self.model, data_loader, self.device
            )
            logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
            if self.log_to_wandb:
                wandb.log({"epoch": epoch + 1, "loss": avg_loss}, step=epoch + 1)

            if (epoch + 1) % self.evaluation_frequency == 0:
                ModelTrainerUtils.evaluate_and_log(
                    self, train_images, train_labels, test_images, test_labels, epoch
                )

        if self.log_to_wandb:
            wandb.finish()
            logger.info("WandB run has been stopped.")

    def evaluate(self, images, labels, dataset_type):
        """
        Evaluates the model on the provided dataset.

        Args:
            images (numpy.ndarray): The images to evaluate.
            labels (numpy.ndarray): The labels corresponding to the images.
            dataset_type (str): The type of dataset being evaluated (e.g., "train", "test", "validation").

        Purpose:
            - Sets the model to evaluation mode.
            - Creates a DataLoader for the dataset.
            - Iterates over the dataset without computing gradients.
            - For each batch, performs the following steps:
                - Moves the images and labels to the appropriate device (CPU or GPU).
                - Performs a forward pass to compute the model's predictions.
                - Computes the number of correct predictions.
            - Computes and logs the accuracy, precision, recall, and F1-score of the model on the dataset.
        """
        self.model.eval()
        data_loader = ModelTrainerUtils.create_data_loader(
            images, labels, self.batch_size, self.evaluation_shuffle
        )

        all_labels, all_predictions = ModelTrainerUtils.gather_predictions(
            self.model, data_loader, self.device
        )

        metrics = ModelTrainerUtils.compute_metrics(all_labels, all_predictions)
        ModelTrainerUtils.log_evaluation_metrics(metrics, dataset_type)

        return metrics

    def cross_validate(self, images, labels):
        """
        Performs k-fold cross-validation on the provided dataset.

        Args:
            images (numpy.ndarray): The images for cross-validation.
            labels (numpy.ndarray): The labels corresponding to the images.
            k_folds (int): The number of folds for cross-validation.

        Purpose:
            - Splits the dataset into k folds.
            - Trains and evaluates the model on each fold.
            - Logs the average metrics across all folds.
        """
        original_log_to_wandb = self.log_to_wandb
        self.log_to_wandb = False  # Disable wandb logging for cross-validation

        kf = KFold(n_splits=self.k_folds, shuffle=self.cross_validation_shuffle)
        fold_metrics = []

        for fold, (train_index, val_index) in enumerate(kf.split(images)):
            logger.info(f"Fold {fold+1}/{self.k_folds}")
            metrics = ModelTrainerUtils.train_and_evaluate_fold(
                self, images, labels, train_index, val_index, fold
            )
            fold_metrics.append(metrics)

        avg_metrics = ModelTrainerUtils.compute_average_metrics(fold_metrics)
        ModelTrainerUtils.log_cross_validation_metrics(avg_metrics, self.k_folds)

        self.log_to_wandb = (
            original_log_to_wandb  # Re-enable wandb logging after cross-validation
        )

        return avg_metrics
