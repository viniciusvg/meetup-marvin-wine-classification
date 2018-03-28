#!/usr/bin/env python
# coding=utf-8

"""Trainer engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['Trainer']


logger = get_logger('trainer')


class Trainer(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        from sklearn.ensemble import RandomForestClassifier

        models = {}

        for wine_type in params["wine_types"]:
            clf = RandomForestClassifier(random_state=params['random_state'])
            clf.fit(self.marvin_dataset[wine_type]["X_train"], self.marvin_dataset[wine_type]["y_train"])

            models[wine_type] = {
                "clf": clf
            }

        self.marvin_model = models

