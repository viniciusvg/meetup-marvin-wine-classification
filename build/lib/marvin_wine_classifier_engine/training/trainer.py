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
        clf_red = RandomForestClassifier()
        clf_red.fit(self.marvin_dataset["red"]["X_train"], self.marvin_dataset["red"]["y_train"])

        clf_white = RandomForestClassifier()
        clf_white.fit(self.marvin_dataset["white"]["X_train"], self.marvin_dataset["white"]["y_train"])

        self.marvin_model = {
            "red": {
                "clf": clf_red
            },
            "white": {
                "clf": clf_white
            }
        }

