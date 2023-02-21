import numpy as np
import pandas as pd

direction_to_degree = {
    "N": np.random.uniform(348.75, 360)
    if np.random.uniform(0, 1) < 0.5
    else np.random.uniform(0, 11.25),
    "NNE": np.random.uniform(11.25, 33.75),
    "NE": np.random.uniform(33.75, 56.25),
    "ENE": np.random.uniform(56.25, 78.75),
    "E": np.random.uniform(78.75, 101.25),
    "ESE": np.random.uniform(101.25, 123.75),
    "SE": np.random.uniform(123.75, 146.25),
    "SSE": np.random.uniform(146.25, 168.75),
    "S": np.random.uniform(168.75, 191.25),
    "SSW": np.random.uniform(191.25, 213.75),
    "SW": np.random.uniform(213.75, 236.25),
    "WSW": np.random.uniform(236.25, 258.75),
    "W": np.random.uniform(258.75, 281.25),
    "WNW": np.random.uniform(281.25, 303.75),
    "NW": np.random.uniform(303.75, 326.25),
    "NNW": np.random.uniform(326.25, 348.75),
}


def convert_wd(df: pd.DataFrame):
    # convert the string wind directions to angles
    for wd in direction_to_degree.keys():
        df.loc[df["wd"] == wd, "wd"] = float(direction_to_degree[wd])
    df["wd"] = pd.to_numeric(df["wd"])
    return df


def splitting(df: pd.DataFrame):
    # exact split with our dataframe
    df_train = df[: int(len(df) * 2 / 3)]
    df_test = df[int(len(df) * 2 / 3) :]
    return (df_train, df_test)


def cleaning(df_train, df_test):
    # remove nan values
    columns = [
        "PM2.5",
        "PM10",
        "SO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "WSPM",
        "wd",
    ]
    stations = df_train["station"].unique()
    for station in stations:
        for col in columns:
            value = df_train[df_train["station"] == station][col].median()
            df_train.loc[df_train["station"] == station, col] = df_train[
                df_train["station"] == station
            ][col].fillna(value)
            df_test.loc[df_test["station"] == station, col] = df_test[
                df_test["station"] == station
            ][col].fillna(value)
        value = df_train[df_train["station"] == station][col].mean()
        df_train.loc[df_train["station"] == station, "NO2"] = df_train[
            df_train["station"] == station
        ]["NO2"].fillna(value)
        df_test.loc[df_test["station"] == station, "NO2"] = df_test[
            df_test["station"] == station
        ]["NO2"].fillna(value)
    return (df_train, df_test)


def get_datetime(df: pd.DataFrame):
    # return a dataframe indexed with datetime
    df["dateInt"] = (
        df["year"].astype(str)
        + df["month"].astype(str).str.zfill(2)
        + df["day"].astype(str).str.zfill(2)
        + df["hour"].astype(str).str.zfill(2)
    )
    df["Date"] = pd.to_datetime(df["dateInt"], format="%Y%m%d%H")
    df["dateInt"] = (
        df["year"].astype(str)
        + df["month"].astype(str).str.zfill(2)
        + df["day"].astype(str).str.zfill(2)
    )
    df["Date-time"] = pd.to_datetime(df["dateInt"], format="%Y%m%d")
    df.drop(["dateInt", "Date"], axis=1, inplace=True)
    df.set_index("Date-time", inplace=True)
    return df


def monthly_grouped(df: pd.DataFrame):
    # to get a dataframe by stations monthly averaged
    df.drop(["year", "month", "day", "hour"], axis=1, inplace=True)
    return df.groupby("station").resample("D").mean()


def preprocess_df(df: pd.DataFrame):
    df = convert_wd(df)
    df_train, df_test = splitting(df)
    df_train, df_test = cleaning(df_train, df_test)
    df_train = get_datetime(df_train)
    df_test = get_datetime(df_test)
    return (df_train, df_test)
