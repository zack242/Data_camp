from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator


def select_feature(df, feature):
    return df[feature].copy()


class FeatureSelector(BaseEstimator):
    def __init__(self, features) -> None:
        super().__init__()
        self.features = features

    def fit(self, X, y):
        return self

    def transform(self, X):
        return select_feature(X, self.features)


def get_estimator():

    # Variables to consider
    category = ["station"]
    numeric = ["TEMP", "PRES", "DEWP", "RAIN"]
    # Model
    model = RandomForestRegressor()
    # Transformer
    transf = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), category),
            ("numeric", StandardScaler, numeric),
        ],
        remainder="drop",
    )

    # Pipeline
    pipe = Pipeline([("transf", transf), ("model", model)])
    pipe = make_pipeline(transf, model)

    return model


def get_estimator():

    # Columns to select
    numeric_features = ["TEMP", "PRES", "DEWP", "RAIN"]
    # Model
    model = RandomForestRegressor(n_jobs=-1)
    # Custom transformer
    feature_selector = FeatureSelector(features=numeric_features)
    # Pipeline
    pipe = make_pipeline(feature_selector, StandardScaler(),  model)

    return pipe
