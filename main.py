from config import paths
from src.utils.setup import Setup
from src.data.data_reader import DataReader
from src.data.data_processor import DataProcessor

# from src.model.classes.sample_processor import SampleProcessor
# from src.model.classes.performance_processor import PerformanceProcessor
# from model.classes.builder import Generator
# from model.classes.trainer import GANTrainer
import logging as logger
import warnings

logger.basicConfig(level=logger.INFO)
warnings.filterwarnings("ignore")


def main() -> None:
    """Main function to run the data processing and model pipeline."""

    logger.info("Pipeline started")

    setup_instance = Setup()
    config = setup_instance.config

    # data_processor = DataProcessor(
    #     image_height=config.image.pix_height,
    #     image_width=config.image.pix_width,
    #     num_channels=config.image.num_channels,
    # )
    # sample_processor = SampleProcessor(
    #     n_classes=config.model.num_classes,
    #     label_smoothing=config.model.label_smoothing.run,
    #     label_smoothing_degree=config.model.label_smoothing.degree,
    # )
    # performance_processor = PerformanceProcessor(
    #     sample_processor=sample_processor,
    #     save_dir=paths.Paths.OUTPUTS_PATH.value,
    #     latent_dim=config.model.latent_dim,
    # )
    # discriminator = Discriminator(
    #     n_classes=config.model.num_classes,
    #     image_height=config.image.pix_height,
    #     image_width=config.image.pix_width,
    #     num_channels=config.image.num_channels,
    #     batch_size=config.model.batch_size,
    #     embedding_dim=config.model.embedding_dim,
    #     LeakyReLU_alpha=config.discriminator.LeakyReLU_alpha,
    #     dropout_rate=config.discriminator.dropout_rate,
    #     adam_lr=config.discriminator.adam_lr,
    #     adam_beta_1=config.discriminator.adam_beta_1,
    # )
    # generator = Generator(
    #     n_classes=config.model.num_classes,
    #     batch_size=config.model.batch_size,
    #     latent_dim=config.model.latent_dim,
    #     embedding_dim=config.model.embedding_dim,
    #     LeakyReLU_alpha=config.generator.LeakyReLU_alpha,
    # )

    data = DataReader.load_data()
    train_images, train_labels, test_images, test_labels = DataProcessor.run(data)

    discriminator.build_model()
    discriminator_model = discriminator.get_model()

    generator.build_model()
    generator_model = generator.get_model()

    gan_compiler = GANCompiler(
        generator_model=generator_model,
        discriminator_model=discriminator_model,
        adam_lr=config.generator.adam_lr,
        adam_beta_1=config.generator.adam_beta_1,
    )
    gan_compiler.build_model()
    gan_model = gan_compiler.get_model()

    gan_trainer = GANTrainer(
        generator_model=generator_model,
        discriminator_model=discriminator_model,
        gan_model=gan_model,
        sample_processor=sample_processor,
        performance_processor=performance_processor,
        images=train_images,
        labels=train_labels,
        latent_dim=config.model.latent_dim,
        start_epoch=config.model.starting_epoch,
        num_epochs=config.model.epochs,
        num_batch=config.model.num_batches,
        learning_rate_decay_run=config.model.learning_rate_decay.run,
        learning_rate_decay_rate=config.model.learning_rate_decay.rate,
        learning_rate_decay_epoch=config.model.learning_rate_decay.epoch,
    )
    gan_trainer.train_model()
    generator_model, discriminator_model, gan_model = gan_trainer.get_models()

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
