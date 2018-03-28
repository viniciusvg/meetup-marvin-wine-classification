#!/usr/bin/env python
# coding=utf-8

"""AcquisitorAndCleaner engine action.

Use this module to add the project main code.
"""

from .._compatibility import six
from .._logging import get_logger

from marvin_python_toolbox.engine_base import EngineBaseDataHandler

__all__ = ['AcquisitorAndCleaner']


logger = get_logger('acquisitor_and_cleaner')


class AcquisitorAndCleaner(EngineBaseDataHandler):

    def __init__(self, **kwargs):
        super(AcquisitorAndCleaner, self).__init__(**kwargs)

    def execute(self, params, **kwargs):
        import pandas as pd

        white_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")
        red_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

        white_wine["wine_type"] = "white"
        red_wine["wine_type"] = "red"

        labes = ["low", "medium", "high"]
        bins = [0, 5, 7, 10]

        white_wine["quality_label"] = pd.cut(white_wine["quality"], bins, labels=labes)
        red_wine["quality_label"] = pd.cut(red_wine["quality"], bins, labels=labes)

        self.marvin_initial_dataset = {
            "red": red_wine,
            "white": white_wine
        }

