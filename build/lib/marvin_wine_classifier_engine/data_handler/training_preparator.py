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

        X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
            self.marvin_initial_dataset["red"][cols], self.marvin_initial_dataset["red"]["quality_label"],
            test_size=params["test_size"], random_state=params["random_state"])

        X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(
            self.marvin_initial_dataset["white"][cols], self.marvin_initial_dataset["white"]["quality_label"],
            test_size=params["test_size"], random_state=params["random_state"])

        self.marvin_dataset = {
            "red": {
                "X_train": X_train_red,
                "X_test": X_test_red,
                "y_train": y_train_red,
                "y_test": y_test_red
            },
            "white": {
                "X_train": X_train_white,
                "X_test": X_test_white,
                "y_train": y_train_white,
                "y_test": y_test_white
            }
        }

