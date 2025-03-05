from config import paths
from src.utils.setup import Setup
from src.data.reader import DataReader
from src.data.processor import DataProcessor
from src.model.classes.compiler import ModelCompiler
from src.model.classes.trainer import ModelTrainer
import logging as logger
import warnings

logger.basicConfig(level=logger.INFO)
warnings.filterwarnings("ignore")


def main() -> None:
    """Main function to run the data processing and model pipeline."""

    logger.info("Pipeline started")

    setup_instance = Setup()
    config = setup_instance.config

    data = DataReader.load_data()
    train_images, train_labels, test_images, test_labels = DataProcessor.run(data)

    model = ModelCompiler(config)
    model.compile()

    trainer = ModelTrainer(model, config)
    trainer.train(train_images, train_labels)
    trainer.evaluate(test_images, test_labels)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
