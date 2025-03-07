import torch
from torch.utils.data import DataLoader, TensorDataset
import logging as logger
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import wandb
from sklearn.model_selection import KFold
import numpy as np
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
        self._initialize_config(config)
        self._initialize_model(model)
        self._initialize_logging(config)

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
        data_loader = self._create_data_loader(train_images, train_labels, self.training_shuffle)

        for epoch in range(self.epochs):
            avg_loss = self._train_one_epoch(data_loader)
            logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
            if self.log_to_wandb:
                wandb.log({"epoch": epoch + 1, "loss": avg_loss}, step=epoch + 1)

            if (epoch + 1) % self.evaluation_frequency == 0:
                self._evaluate_and_log(train_images, train_labels, test_images, test_labels, epoch)

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
        data_loader = self._create_data_loader(images, labels, shuffle=self.evaluation_shuffle)

        all_labels, all_predictions = self._gather_predictions(data_loader)

        metrics = self._compute_metrics(all_labels, all_predictions)
        self._log_evaluation_metrics(metrics, dataset_type)

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
            metrics = self._train_and_evaluate_fold(images, labels, train_index, val_index, fold)
            fold_metrics.append(metrics)

        avg_metrics = self._compute_average_metrics(fold_metrics)
        self._log_cross_validation_metrics(avg_metrics, self.k_folds)

        self.log_to_wandb = original_log_to_wandb  # Re-enable wandb logging after cross-validation

        return avg_metrics

    def _initialize_config(self, config):
        """
        Initializes the configuration settings.

        Args:
            config (object): The configuration object containing training parameters.
        """
        self.config = config
        self.batch_size = config.model.batch_size
        self.epochs = config.model.epochs
        self.learning_rate = config.model.learning_rate
        self.optimizer = config.model.optimizer
        self.activation_function = config.model.activation_function
        self.training_shuffle = config.model.shuffle
        self.evaluate_on_train = config.evaluation.train
        self.evaluate_on_test = config.evaluation.test
        self.evaluation_frequency = config.evaluation.epoch_frequency
        self.evaluation_shuffle = config.evaluation.shuffle
        self.k_folds = config.cross_validation.k_folds
        self.cross_validation_shuffle = config.cross_validation.shuffle
        self.log_to_wandb = config.logging.log_to_wandb

    def _initialize_model(self, model):
        """
        Initializes the model and device settings.

        Args:
            model (nn.Module): The CNN model to be trained and evaluated.
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(
            f"ModelTrainer initialized with batch size: {self.batch_size} and epochs: {self.epochs}"
        )

    def _initialize_logging(self, config):
        """
        Initializes logging settings, including Weights & Biases if enabled.

        Args:
            config (object): The configuration object containing logging parameters.
        """
        if self.log_to_wandb:
            wandb.init(project=config.logging.project_name, config=config)
            self.config = wandb.config

    def _create_data_loader(self, images, labels, shuffle):
        dataset = TensorDataset(
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _train_one_epoch(self, data_loader):
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.model.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.model.criterion(outputs, labels)
            loss.backward()
            self.model.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(data_loader)

    def _evaluate_and_log(self, train_images, train_labels, test_images, test_labels, epoch):
        if self.evaluate_on_train:
            train_metrics = self.evaluate(train_images, train_labels, "train")
            self._log_metrics(train_metrics, "train", epoch)
        if self.evaluate_on_test:
            test_metrics = self.evaluate(test_images, test_labels, "test")
            self._log_metrics(test_metrics, "test", epoch)

    def _log_metrics(self, metrics, dataset_type, epoch):
        accuracy, precision, recall, f1 = metrics
        if self.log_to_wandb:
            wandb.log({
                f"{dataset_type}_accuracy": accuracy,
                f"{dataset_type}_precision": precision,
                f"{dataset_type}_recall": recall,
                f"{dataset_type}_f1": f1
            }, step=epoch + 1)

    def _gather_predictions(self, data_loader):
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        return all_labels, all_predictions

    def _compute_metrics(self, labels, predictions):
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")
        return accuracy, precision, recall, f1

    def _log_evaluation_metrics(self, metrics, dataset_type):
        accuracy, precision, recall, f1 = metrics
        logger.debug(f"Accuracy of the model on the {dataset_type} images: {accuracy:.4f}")
        logger.debug(f"Precision of the model on the {dataset_type} images: {precision:.4f}")
        logger.debug(f"Recall of the model on the {dataset_type} images: {recall:.4f}")
        logger.debug(f"F1 Score of the model on the {dataset_type} images: {f1:.4f}")

    def _train_and_evaluate_fold(self, images, labels, train_index, val_index, fold):
        """
        Trains and evaluates the model on a single fold.

        Args:
            images (numpy.ndarray): The images for cross-validation.
            labels (numpy.ndarray): The labels corresponding to the images.
            train_index (array-like): The indices for the training set.
            val_index (array-like): The indices for the validation set.
            fold (int): The current fold number.

        Returns:
            tuple: The metrics for the current fold.
        """
        train_images, val_images = images[train_index], images[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        self.train(train_images, train_labels, val_images, val_labels)
        metrics = self.evaluate(val_images, val_labels, dataset_type=f"fold_{fold+1}")
        return metrics

    def _compute_average_metrics(self, fold_metrics):
        """
        Computes the average metrics across all folds.

        Args:
            fold_metrics (list): A list of metrics for each fold.

        Returns:
            dict: The average accuracy, precision, recall, and F1-score.
        """
        avg_accuracy = sum([metrics[0] for metrics in fold_metrics]) / len(fold_metrics)
        avg_precision = sum([metrics[1] for metrics in fold_metrics]) / len(fold_metrics)
        avg_recall = sum([metrics[2] for metrics in fold_metrics]) / len(fold_metrics)
        avg_f1 = sum([metrics[3] for metrics in fold_metrics]) / len(fold_metrics)

        scores = {
            "average_accuracy": avg_accuracy,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1
        }

        return scores

    def _log_cross_validation_metrics(self, metrics, k_folds):
        """
        Logs the average metrics across all folds.

        Args:
            metrics (dict): The average accuracy, precision, recall, and F1-score.
            k_folds (int): The number of folds for cross-validation.
        """
        accuracy = metrics['average_accuracy']
        precision = metrics['average_precision']
        recall = metrics['average_recall']
        f1 = metrics['average_f1']

        logger.info(f"Cross-Validation with {k_folds} folds:")
        logger.info(f"Average Accuracy: {accuracy:.4f}")
        logger.info(f"Average Precision: {precision:.4f}")
        logger.info(f"Average Recall: {recall:.4f}")
        logger.info(f"Average F1 Score: {f1:.4f}")

        if self.log_to_wandb:
            wandb.log({
                "cv_average_accuracy": accuracy,
                "cv_average_precision": precision,
                "cv_average_recall": recall,
                "cv_average_f1": f1
            })
