import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.stats import entropy


def calculate_inception_score(
    images: np.ndarray, n_split: int = 10, eps: float = 1e-16
) -> float:
    """Calculate the Inception Score (IS) for the generated images."""
    # Load the InceptionV3 model
    model = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))

    # Resize images to 299x299 and preprocess them
    images_resized = np.array([np.resize(image, (299, 299, 3)) for image in images])
    images_resized = preprocess_input(images_resized)

    # Predict the probabilities for each image
    preds = model.predict(images_resized)

    # Split the predictions into groups
    scores = []
    n_part = int(np.floor(preds.shape[0] / n_split))
    for i in range(n_split):
        part = preds[i * n_part : (i + 1) * n_part]
        p_y = np.mean(part, axis=0)
        scores.append(np.exp(np.mean([entropy(p, p_y) for p in part])))

    # Calculate the mean and standard deviation of the scores
    is_mean, is_std = np.mean(scores), np.std(scores)
    return is_mean
