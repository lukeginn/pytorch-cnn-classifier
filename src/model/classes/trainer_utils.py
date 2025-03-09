import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import logging as logger
import wandb
import time


class ModelTrainerUtils:
    @staticmethod
    def create_data_loader(images, labels, batch_size, shuffle):
        dataset = TensorDataset(
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def gather_predictions(model, data_loader, device):
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        return all_labels, all_predictions

    @staticmethod
    def compute_metrics(labels, predictions):
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="weighted")
        recall = recall_score(labels, predictions, average="weighted")
        f1 = f1_score(labels, predictions, average="weighted")
        return accuracy, precision, recall, f1

    @staticmethod
    def compute_average_metrics(fold_metrics):
        avg_accuracy = sum([metrics[0] for metrics in fold_metrics]) / len(fold_metrics)
        avg_precision = sum([metrics[1] for metrics in fold_metrics]) / len(
            fold_metrics
        )
        avg_recall = sum([metrics[2] for metrics in fold_metrics]) / len(fold_metrics)
        avg_f1 = sum([metrics[3] for metrics in fold_metrics]) / len(fold_metrics)

        scores = {
            "average_accuracy": avg_accuracy,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1,
        }

        return scores

    @staticmethod
    def initialize_model(model):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = "cpu"
        model.to(device)
        return model, device

    @staticmethod
    def initialize_logging(config):
        if config.logging.log_to_wandb:
            experiment_name = "experiment_run_on_" + time.strftime("%Y-%m-%d_%H-%M-%S")
            wandb.init(
                project=config.logging.project_name, config=config, name=experiment_name
            )
            return wandb.config

    @staticmethod
    def train_one_epoch(model, data_loader, device):
        running_loss = 0.0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            model.optimizer.zero_grad()
            outputs = model(images)
            loss = model.criterion(outputs, labels)
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(data_loader)

    @staticmethod
    def evaluate_and_log(
        trainer, train_images, train_labels, test_images, test_labels, epoch
    ):
        train_metrics = trainer.evaluate(train_images, train_labels, "train")
        ModelTrainerUtils.log_metrics(trainer, train_metrics, "train", epoch)

        test_metrics = trainer.evaluate(test_images, test_labels, "test")
        ModelTrainerUtils.log_metrics(trainer, test_metrics, "test", epoch)

    @staticmethod
    def log_metrics(trainer, metrics, dataset_type, epoch):
        accuracy, precision, recall, f1 = metrics
        if trainer.log_to_wandb:
            wandb.log(
                {
                    f"{dataset_type}_accuracy": accuracy,
                    f"{dataset_type}_precision": precision,
                    f"{dataset_type}_recall": recall,
                    f"{dataset_type}_f1": f1,
                },
                step=epoch + 1,
            )

    @staticmethod
    def log_evaluation_metrics(metrics, dataset_type):
        accuracy, precision, recall, f1 = metrics
        logger.debug(
            f"Accuracy of the model on the {dataset_type} images: {accuracy:.4f}"
        )
        logger.debug(
            f"Precision of the model on the {dataset_type} images: {precision:.4f}"
        )
        logger.debug(f"Recall of the model on the {dataset_type} images: {recall:.4f}")
        logger.debug(f"F1 Score of the model on the {dataset_type} images: {f1:.4f}")

    @staticmethod
    def train_and_evaluate_fold(trainer, images, labels, train_index, val_index, fold):
        train_images, val_images = images[train_index], images[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        trainer.train(train_images, train_labels, val_images, val_labels)
        metrics = trainer.evaluate(
            val_images, val_labels, dataset_type=f"fold_{fold+1}"
        )
        return metrics

    @staticmethod
    def log_cross_validation_metrics(metrics, k_folds, log_to_wandb):
        accuracy = metrics["average_accuracy"]
        precision = metrics["average_precision"]
        recall = metrics["average_recall"]
        f1 = metrics["average_f1"]

        logger.info(f"Cross-Validation with {k_folds} folds:")
        logger.info(f"Average Accuracy: {accuracy:.4f}")
        logger.info(f"Average Precision: {precision:.4f}")
        logger.info(f"Average Recall: {recall:.4f}")
        logger.info(f"Average F1 Score: {f1:.4f}")

        if log_to_wandb:
            wandb.log(
                {
                    "cv_average_accuracy": accuracy,
                    "cv_average_precision": precision,
                    "cv_average_recall": recall,
                    "cv_average_f1": f1,
                }
            )
