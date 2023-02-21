import os
import pandas as pd
import rampwf as rw
from data_cleaning import *
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from rampwf.score_types.base import BaseScoreType


problem_title = "Quality of Air"
Predictions = rw.prediction_types.make_regression()
workflow = rw.workflows.Regressor()


class R2(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    name = "R2"

    def __init__(self, precision=4):
        self.precision = precision

    def __call__(self, y_true, y_pred):
        r_2 = r2_score(y_true, y_pred)
        return r_2


score_types = [
    R2(),
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)


def _read_data(path, type_):
    data_path = os.path.join("Data", "merged_data.csv")
    _target_column_name = "PM2.5"

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if type_ == "train":
        data_train, _ = preprocess_df(data)
        X, y = get_labels(data_train, _target_column_name)

    elif type_ == "test":
        _, data_test = preprocess_df(data)
        X, y = get_labels(data_test, _target_column_name)

    # for the "quick-test" mode, use less data
    test = os.getenv("RAMP_TEST_MODE", 0)
    if test:
        N_small = 35000
        X = data[:N_small]
        y = y[:N_small]

    return X, y


def get_train_data():
    return _read_data("train")


def get_test_data():
    return _read_data("test")
