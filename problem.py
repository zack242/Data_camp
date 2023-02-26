import os
import pandas as pd
import rampwf as rw
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from rampwf.score_types.base import BaseScoreType


problem_title = "Quality of Air"
Predictions = rw.prediction_types.make_regression()
workflow = rw.workflows.Estimator()


class R2(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    name = "R2"

    def __init__(self, precision=2):
        self.precision = precision

    def __call__(self, y_true, y_pred):
        r_2 = r2_score(y_true, y_pred)
        return r_2


# Symmetric Mean Absolute Percentage Error (SMAPE)
class SMAPE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0
    name = "SMAPE"

    def __init__(self, precision=2):
        self.precision = precision

    def __call__(self, y_true, y_pred):
        smape = (
            100
            / len(y_true)
            * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        )
        return smape


score_types = [
    R2(),
    SMAPE(),
]


def get_cv(X, y):
    cv = TimeSeriesSplit(n_splits=5)
    return cv.split(X, y)


def _read_data(path, f_name):

    _target_column_name = "PM2.5"

    data = pd.read_csv(os.path.join(path, "data", f_name))
    y = data[_target_column_name].values
    X = data.drop([_target_column_name], axis=1)

    # for the "quick-test" mode, use less data
    test = os.getenv("RAMP_TEST_MODE", 0)
    if test:
        N_small = 35000
        X = data[:N_small]
        y = y[:N_small]

    return X, y


def get_train_data(path="."):
    f_name = "train.csv"
    return _read_data(path, f_name)


def get_test_data(path="."):
    f_name = "test.csv"
    return _read_data(path, f_name)
