import os
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, recall_score, precision_score
import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.workflows.sklearn_pipeline import SKLearnPipeline
from rampwf.workflows.sklearn_pipeline import Estimator
from preprocess import preprocess_df, get_labels

problem_title = "Quality of Air"
data_path = os.path.join("Data", "merged_data.csv")

# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)


def _read_data(path, type_):

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if type_ == "train":
        data_train, _ = preprocess_df(data)
        X, y = get_labels(data_train, "PM2.5")

    elif type_ == "test":
        _, data_test = preprocess_df(data)
        X, y = get_labels(data_test, "PM2.5")

    # for the "quick-test" mode, use less data
    test = os.getenv("RAMP_TEST_MODE", 0)
    if test:
        N_small = 35000
        X = data[:N_small]
        y = y[:N_small]

    return X, y


def get_train_data(path="."):
    return _read_data(path, "train")


def get_test_data(path="."):
    return _read_data(path, "test")
