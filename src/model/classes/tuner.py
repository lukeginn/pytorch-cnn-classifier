import pandas as pd
import itertools
from src.model.classes.trainer import ModelTrainer
from src.model.classes.compiler import ModelCompiler
import logging as logger

class HyperparameterTuner:
    def __init__(self, config):
        self.config = config
        self.param_grid = config.cross_validation.param_grid
        self.metric_to_optimize = config.cross_validation.metric_to_optimize

    def tune(self, train_images, train_labels):
        logger.info("Starting hyperparameter tuning")

        best_score = float('-inf')
        best_params = None
        params_list = []
        scores_list = []

        for params in self._param_combinations():
            logger.info(f"Training with params: {params}")

            # Set up the config
            self.config.model.shuffle = params['shuffle']
            self.config.model.batch_size = params['batch_size']
            self.config.model.epochs = params['epochs']
            self.config.model.learning_rate = params['learning_rate']
            self.config.model.optimizer = params['optimizer']
            self.config.model.activation_function = params['activation_function']
            for i, layer in enumerate(self.config['model']['fc_layers']):
                if 'dropout' in layer and i != len(self.config['model']['fc_layers']) - 1:
                    layer['dropout'] = params['dropout']

            model = ModelCompiler(self.config)
            model.compile()

            trainer = ModelTrainer(model, self.config, False)
            score = trainer.cross_validate(train_images, train_labels)
            score_to_compare = score[self.metric_to_optimize]

            params_list.append(params)
            scores_list.append(score)

            if score_to_compare > best_score:
                best_score = score_to_compare
                best_params = params

        # Convert results to a DataFrame
        params_df = pd.DataFrame(params_list)
        scores_df = pd.DataFrame(scores_list)
        results_table = pd.concat([params_df, scores_df], axis=1)

        logger.info("Hyperparameter tuning completed")
        logger.info(f"Best params: {best_params}, Best score: {best_score}")
        logger.info(f"Cross-validation results table: {results_table}")

        # Set the best params in the config
        self.config.model.shuffle = best_params['shuffle']
        self.config.model.batch_size = best_params['batch_size']
        self.config.model.epochs = best_params['epochs']
        self.config.model.learning_rate = best_params['learning_rate']
        self.config.model.optimizer = best_params['optimizer']
        self.config.model.activation_function = best_params['activation_function']
        for i, layer in enumerate(self.config['model']['fc_layers']):
            if 'dropout' in layer and i < len(self.config['model']['fc_layers']) - 1:
                layer['dropout'] = best_params['dropout']

        return best_params, best_score, results_table, self.config

    def _param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))