import tensorflow as tf
from keras.callbacks import TensorBoard
import datetime


def setup_tensorboard():
    # To visualize the training process, run TensorBoard from the command line:
    # tensorboard --logdir=logs/fit
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback, file_writer


def log_metrics_to_tensorboard(
    epoch: int,
    g_loss: float,
    d_loss: float,
    acc_real: float,
    acc_fake: float,
    inception_score: float,
    fid_score: float,
    new_lr: float,
    tensorboard_callback: TensorBoard,
    file_writer: tf.summary.SummaryWriter,
) -> None:
    # Log metrics to TensorBoard
    with file_writer.as_default():
        tf.summary.scalar("Generator Loss", g_loss, step=epoch)
        tf.summary.scalar("Discriminator Loss", d_loss, step=epoch)
        tf.summary.scalar("Accuracy Real", acc_real, step=epoch)
        tf.summary.scalar("Accuracy Fake", acc_fake, step=epoch)
        tf.summary.scalar("Inception Score", inception_score, step=epoch)
        tf.summary.scalar("FID Score", fid_score, step=epoch)
        tf.summary.scalar("Learning Rate", new_lr, step=epoch)
    file_writer.flush()
