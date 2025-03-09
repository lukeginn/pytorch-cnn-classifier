import pandas as pd
import itertools
from src.model.classes.trainer import ModelTrainer
from src.model.classes.compiler import ModelCompiler
import logging as logger
from config.paths import Paths


class HyperparameterTuner:
    def __init__(self, config):
        self.config = config
        self.param_grid = config.cross_validation.param_grid
        self.metric_to_optimize = config.cross_validation.metric_to_optimize

    def tune(self, train_images, train_labels):
        logger.info("Starting hyperparameter tuning")

        best_score = float("inf")
        best_params = None
        params_list = []
        scores_list = []

        for params in self._param_combinations():
            logger.info(f"Training with params: {params}")

            self._set_config(params)
            score = self._compile_train_and_cross_validate(train_images, train_labels)
            score_to_compare = score[self.metric_to_optimize]

            params_list.append(params)
            scores_list.append(score)

            if score_to_compare < best_score:
                best_score = score_to_compare
                best_params = params

        results_table = self._create_results_table(params_list, scores_list)

        logger.info("Hyperparameter tuning completed")
        logger.info(f"Best params: {best_params}, Best score: {best_score}")
        self._set_config(best_params)

        return best_params, best_score, results_table, self.config

    def _param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def _set_config(self, params):
        self.config["model"]["shuffle"] = params["shuffle"]
        self.config["model"]["batch_size"] = params["batch_size"]
        self.config["model"]["epochs"] = params["epochs"]
        self.config["model"]["learning_rate"] = params["learning_rate"]
        self.config["model"]["optimizer"] = params["optimizer"]
        self.config["model"]["activation_function"] = params["activation_function"]
        for i, layer in enumerate(self.config["model"]["fc_layers"]):
            if layer["type"] == "Dropout":
                layer["p"] = params["dropout"]
        for i, layer in enumerate(self.config["model"]["deconv_layers"]):
            if layer["type"] == "Dropout2d":
                layer["p"] = params["dropout2d"]

    def _compile_train_and_cross_validate(self, train_images, train_labels):
        model = ModelCompiler(self.config)
        model.compile()
        trainer = ModelTrainer(model, self.config, False)
        return trainer.cross_validate(train_images, train_labels)

    def _create_results_table(self, params_list, scores_list):
        params_df = pd.DataFrame(params_list)
        scores_df = pd.DataFrame(scores_list)

        results_table = pd.concat([params_df, scores_df], axis=1)
        logger.info(f"Cross-validation results table: {results_table}")

        logger.info(
            f"Saving cross-validation results to {Paths.CROSS_VALIDATION_RESULTS_PATH.value}"
        )
        results_table.to_csv(Paths.CROSS_VALIDATION_RESULTS_PATH.value, index=False)

        return results_table
