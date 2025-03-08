import torch
import logging as logger


def check_gpu():
    cuda_available = torch.cuda.is_available()
    logger.info("CUDA available: %s", cuda_available)

    if cuda_available:
        logger.info("CUDA version: %s", torch.version.cuda)
        logger.info("cuDNN version: %s", torch.backends.cudnn.version())
        logger.info("PyTorch version: %s", torch.__version__)
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA is not available. Please check your CUDA installation.")
        device = torch.device("cpu")
        logger.info("No GPU found, using CPU")

    return device
