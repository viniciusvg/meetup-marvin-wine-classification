#!/usr/bin/env python
# coding=utf-8

"""Predictor engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBasePrediction

__all__ = ['Predictor']


logger = get_logger('predictor')


class Predictor(EngineBasePrediction):

    def __init__(self, **kwargs):
        super(Predictor, self).__init__(**kwargs)

    def execute(self, input_message, params, **kwargs):
        wine_type = input_message["wine_type"][0]
        features = input_message.drop("wine_type", axis=1)

        if wine_type not in params["wine_types"]:
            pred = None
        else:
            pred = self.marvin_model[wine_type]["clf"].predict(features)[0]

        final_prediction = pred

        return final_prediction
