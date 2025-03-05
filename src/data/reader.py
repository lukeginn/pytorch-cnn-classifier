from dataclasses import dataclass
from torchvision import datasets
import logging as logger


@dataclass
class DataReader:

    @staticmethod
    def load_data():
        logger.info("Loading the MNIST dataset")

        # Load the MNIST dataset
        train_data = datasets.MNIST(root="./inputs", train=True, download=True)
        test_data = datasets.MNIST(root="./inputs", train=False, download=True)

        data = {"train_data": train_data, "test_data": test_data}

        return data
