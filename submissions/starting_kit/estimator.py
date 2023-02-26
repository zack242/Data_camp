from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator


class RegressorTransform(BaseEstimator):

    """
    A scikit-learn estimator that transforms the categorical and numerical features of a dataset using
    one-hot encoding and standard scaling, respectively, before fitting a random forest regression model.

    Parameters
    ----------
    cat_cols : list of str
        The names of the categorical columns in the input data.
    num_cols : list of str
        The names of the numerical columns in the input data.
    " keep_cols" : list of str
        The names of the columns to keep

    Attributes
    ----------
    _processor : ColumnTransformer
        The transformer that applies one-hot encoding and standard scaling to the input data.
    _model : Pipeline
        The pipeline that applies the transformer and fits a random forest regression model to the transformed data.

    Methods
    -------
    fit(X, y)
        Fits the pipeline to the input data and target values.
    predict(X)
        Predicts the target values for new input data using the fitted model.

    """

    def __init__(self, cat_cols=['station'], num_cols=["month", "day", "hour",
                                                       "TEMP", "PRES", "DEWP", "RAIN", "WSPM"],
                 keep_cols=["wd"]) -> None:
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.keep_cols = keep_cols

        self._processor = ColumnTransformer(
            [
                ("1hot_encoder", OneHotEncoder(), self.cat_cols),
                ("scaler", StandardScaler(), self.num_cols),
                ("keep", "passthrough", self.keep_cols)
            ]
        )

        self._model = Pipeline(
            [
                ("processor", self._processor),
                ("model", RandomForestRegressor(max_depth=10, random_state=42))
            ]
        )

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)


def get_estimator():
    regressor = RegressorTransform()
    return regressor._model
