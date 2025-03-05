from dataclasses import dataclass
import numpy as np
import torch
from torchvision import transforms
import logging as logger


@dataclass
class DataProcessor:

    @staticmethod
    def run(data) -> tuple:
        logger.info("Processing the MNIST dataset")

        train_images, train_labels, test_images, test_labels = (
            DataProcessor._split_data_into_train_and_test_images(data)
        )

        train_images = DataProcessor._reshape_images(train_images)
        test_images = DataProcessor._reshape_images(test_images)

        train_images = DataProcessor._scale_images(train_images)
        test_images = DataProcessor._scale_images(test_images)

        train_image_mean, train_image_std = DataProcessor._calculate_mean_and_std(
            train_images
        )
        train_images = DataProcessor._normalize_images(
            train_images, train_image_mean, train_image_std
        )
        test_images = DataProcessor._normalize_images(
            test_images, train_image_mean, train_image_std
        )

        DataProcessor._check_image_shape(train_images)
        DataProcessor._check_image_shape(test_images)
        DataProcessor._check_label_shape(train_labels)
        DataProcessor._check_label_shape(test_labels)

        return train_images, train_labels, test_images, test_labels

    @staticmethod
    def _split_data_into_train_and_test_images(data: tuple) -> tuple:
        train_images = data["train_data"].data
        train_labels = data["train_data"].targets
        test_images = data["test_data"].data
        test_labels = data["test_data"].targets
        return train_images, train_labels, test_images, test_labels

    @staticmethod
    def _scale_images(images: np.ndarray) -> np.ndarray:
        logger.info("Scaling the images")

        # Check if images are already tensors
        if isinstance(images, torch.Tensor):
            # Scales the pixel values to a range between 0 and 1 if not already scaled
            return images.float() / images.float().max()

        # Convert the images to tensors
        # This scales the pixel values to a range between 0 and 1
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        scaled_images = torch.stack([transform(image) for image in images])

        return scaled_images

    @staticmethod
    def _calculate_mean_and_std(images: np.ndarray) -> tuple:
        mean = images.float().mean()
        std = images.float().std()
        return mean, std

    @staticmethod
    def _reshape_images(images: torch.Tensor) -> torch.Tensor:
        logger.info("Reshaping the images")

        # The shape of the images tensor should be (channel, height, width)
        # Here we add the channel dimension to the images tensor
        images = images.unsqueeze(1)
        return images

    @staticmethod
    def _normalize_images(images: np.ndarray, mean: float, std: float) -> torch.Tensor:
        logger.info("Normalizing the images")

        # Normalize the images
        # This scales the pixel values to have a mean of 0 and a standard deviation of 1
        transform = transforms.Compose(
            [
                transforms.Normalize((mean,), (std,)),
            ]
        )
        normalized_images = torch.stack([transform(image) for image in images])

        return normalized_images

    @staticmethod
    def _check_image_shape(images: np.ndarray) -> None:
        # Check the shape of the images tensor
        logger.info(f"Images have the shape: {images.shape}")

        return None

    @staticmethod
    def _check_label_shape(labels: np.ndarray) -> None:
        # Check the shape of the labels tensor
        logger.info(f"Labels have the shape: {labels.shape}")

        return None
