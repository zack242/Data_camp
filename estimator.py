from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_estimator():

    # Variables to consider
    category = ["station"]
    numeric = ['TEMP', 'PRES', 'DEWP', 'RAIN']
    # Model
    model = RandomForestRegressor()
    # Transformer
    transf = ColumnTransformer(transformers=[("cat", OneHotEncoder(
    ), category), ("numeric", StandardScaler, numeric)], remainder='drop')

    # Pipeline
    pipe = Pipeline(
        [
            ("transf", transf),
            ("model", model)
        ]
    )
    return pipe
