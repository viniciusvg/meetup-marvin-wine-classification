#!/usr/bin/env python
# coding=utf-8

"""MetricsEvaluator engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseTraining

__all__ = ['MetricsEvaluator']


logger = get_logger('metrics_evaluator')


class MetricsEvaluator(EngineBaseTraining):

    def __init__(self, **kwargs):
        super(MetricsEvaluator, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        from sklearn.metrics import classification_report

        for wine_type in params["wine_types"]:
            prediction = self.marvin_model[wine_type]["clf"].predict(self.marvin_dataset[wine_type]["X_test"])
            metrics = classification_report(self.marvin_dataset[wine_type]["y_test"], prediction)

            print("Wine type: ", wine_type)
            print(metrics)

        self.marvin_metrics = metrics

