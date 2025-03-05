import torch
from torch.utils.data import DataLoader, TensorDataset
import logging as logger


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

    def __init__(self, model, config):
        """
        Initializes the ModelTrainer with the model and configuration.

        Args:
            model (nn.Module): The CNN model to be trained and evaluated.
            config (object): The configuration object containing training parameters.
        """
        self.batch_size = config.model.batch_size
        self.epochs = config.model.epochs

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(
            f"ModelTrainer initialized with batch size: {self.batch_size} and epochs: {self.epochs}"
        )

    def train(self, train_images, train_labels):
        """
        Trains the model on the provided training dataset.

        Args:
            train_images (numpy.ndarray): The training images.
            train_labels (numpy.ndarray): The training labels.

        Purpose:
            - Sets the model to training mode.
            - Creates a DataLoader for the training dataset.
            - Iterates over the dataset for the specified number of epochs.
            - For each batch, performs the following steps:
                - Moves the images and labels to the appropriate device (CPU or GPU).
                - Resets the gradients of the model's parameters.
                - Performs a forward pass to compute the model's predictions.
                - Computes the loss between the predictions and the true labels.
                - Performs a backward pass to compute the gradients.
                - Updates the model's parameters using the optimizer.
                - Accumulates the running loss for monitoring.
            - Logs the average loss for each epoch.
        """
        self.model.train()
        train_dataset = TensorDataset(
            torch.tensor(train_images, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.model.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.model.criterion(outputs, labels)
                loss.backward()
                self.model.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, test_images, test_labels):
        """
        Evaluates the model on the provided test dataset.

        Args:
            test_images (numpy.ndarray): The test images.
            test_labels (numpy.ndarray): The test labels.

        Purpose:
            - Sets the model to evaluation mode.
            - Creates a DataLoader for the test dataset.
            - Iterates over the test dataset without computing gradients.
            - For each batch, performs the following steps:
                - Moves the images and labels to the appropriate device (CPU or GPU).
                - Performs a forward pass to compute the model's predictions.
                - Computes the number of correct predictions.
            - Computes and logs the accuracy of the model on the test dataset.
        """
        self.model.eval()
        test_dataset = TensorDataset(
            torch.tensor(test_images, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.long),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f"Accuracy of the model on the test images: {accuracy:.2f}%")
