import numpy as np
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3, preprocess_input


def calculate_fid(real_images: np.ndarray, fake_images: np.ndarray) -> float:
    """Calculate the Frechet Inception Distance (FID) between real and fake images."""
    # Load the InceptionV3 model
    model = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))

    # Resize images to 299x299 and preprocess them
    real_images_resized = np.array(
        [np.resize(image, (299, 299, 3)) for image in real_images]
    )
    fake_images_resized = np.array(
        [np.resize(image, (299, 299, 3)) for image in fake_images]
    )
    real_images_resized = preprocess_input(real_images_resized)
    fake_images_resized = preprocess_input(fake_images_resized)

    # Predict the activations for real and fake images
    act_real = model.predict(real_images_resized)
    act_fake = model.predict(fake_images_resized)

    # Calculate the mean and covariance of the activations
    mu_real, sigma_real = act_real.mean(axis=0), cov(act_real, rowvar=False)
    mu_fake, sigma_fake = act_fake.mean(axis=0), cov(act_fake, rowvar=False)

    # Calculate the Frechet distance
    ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
    covmean = sqrtm(sigma_real.dot(sigma_fake))

    # Check and correct imaginary numbers from sqrtm
    if iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid
