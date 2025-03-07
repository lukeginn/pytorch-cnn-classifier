from src.utils.setup import Setup
from src.data.reader import DataReader
from src.data.processor import DataProcessor
from src.model.classes.compiler import ModelCompiler
from src.model.classes.trainer import ModelTrainer
from src.model.classes.tuner import HyperparameterTuner
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

    tuner = HyperparameterTuner(config)
    best_params, best_score, results_table, config = tuner.tune(
        train_images, train_labels
    )
    
    model = ModelCompiler(config)
    model.compile()

    trainer = ModelTrainer(model, config, log_to_wandb=config.logging.log_to_wandb)
    trainer.train(train_images, train_labels, test_images, test_labels)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
