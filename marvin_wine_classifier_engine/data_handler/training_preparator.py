#!/usr/bin/env python
# coding=utf-8

"""TrainingPreparator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['TrainingPreparator']


logger = get_logger('training_preparator')


class TrainingPreparator(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(TrainingPreparator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import SMOTE

        cols = [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        ]

        dataset = {}

        for wine_type in params["wine_types"]:
            X = self.marvin_initial_dataset[wine_type][cols]
            y = self.marvin_initial_dataset[wine_type]["quality_label"]

            X_resampled, y_resampled = SMOTE(random_state=params["random_state"]).fit_sample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=params["test_size"],
                random_state=params["random_state"])

            dataset[wine_type] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test
            }

        self.marvin_dataset = dataset

