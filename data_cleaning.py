import numpy as np
import pandas as pd
import datetime
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

def angle2dir(angle):
    #usefull after groupby for instance to go back to str directions
    #use df['wd'] = df.apply(lambda col: get_value(col['wd']), axis=1)
    if 11.25 < angle < 33.75:
        return 'NNE'
    elif 33.75 < angle < 56.25:
        return 'NE'
    elif 56.25 < angle < 78.75:
        return 'ENE'
    elif 78.75 < angle < 101.25:
        return 'E'
    elif 101.25 < angle < 123.75:
        return 'ESE'
    elif 123.75 < angle < 146.25:
        return 'SE'
    elif 146.25 < angle < 168.75:
        return 'SSE'
    elif 168.75 < angle < 191.25:
        return 'S'
    elif 191.25 < angle < 213.75:
        return 'SSW'
    elif 213.75 < angle < 236.25:
        return 'SW'
    elif 236.25 < angle < 258.75:
        return 'WSW'
    elif 258.75 < angle < 281.25:
        return 'W'
    elif 281.25 < angle < 303.75:
        return 'WNW'
    elif 303.75 < angle < 326.25:
        return 'NW'
    elif 326.25 < angle < 348.75:
        return 'NNW'
    else : 
        return 'N'
    

class dir2degree(BaseEstimator, TransformerMixin):  
    
    def fit(self,df, y=None):
        self.direction_to_degree = {'N':np.random.uniform(348.75, 360) if np.random.uniform(0, 1)<0.5 else np.random.uniform(0, 11.25),
                       'NNE': np.random.uniform(11.25, 33.75),
                       'NE': np.random.uniform(33.75, 56.25),
                       'ENE': np.random.uniform(56.25, 78.75),
                       'E': np.random.uniform(78.75, 101.25),
                       'ESE': np.random.uniform(101.25, 123.75),
                       'SE': np.random.uniform(123.75, 146.25),
                       'SSE': np.random.uniform(146.25, 168.75),
                       'S': np.random.uniform(168.75, 191.25),
                       'SSW': np.random.uniform(191.25, 213.75),
                       'SW': np.random.uniform(213.75, 236.25),
                       'WSW': np.random.uniform(236.25, 258.75),
                       'W': np.random.uniform(258.75, 281.25),
                       'WNW': np.random.uniform(281.25, 303.75),
                       'NW': np.random.uniform(303.75, 326.25),
                       'NNW': np.random.uniform(326.25, 348.75)
                      }
        self.degree_to_direction =  {degree: direction for direction, degree in self.direction_to_degree.items()}
        
        return self
    def transform(self, df, y=None):
        for wd in self.direction_to_degree.keys():
            df.loc[df['wd']==wd, 'wd'] = float(self.direction_to_degree[wd])
        df['wd'] = pd.to_numeric(df['wd'])
        return df
    
    def inverse_transform(self, df, y=None):
        for wd in self.degree_to_direction.keys():
            df.loc[df['wd']==wd, 'wd'] = self.degree_to_direction[wd]
        return df
        
pipe = Pipeline(
    steps=[
        ("direction_to_degree", dir2degree())
    ]
)


def splitting(df:pd.DataFrame):
    #exact split with our dataframe 
    df_train = df[:int(len(df)*2/3)]
    df_test = df[int(len(df)*2/3):]
    return(df_train, df_test)


def cleaning(df_train, df_test):
    #remove nan values
    columns = ['PM2.5', 'PM10', 'SO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP',
       'RAIN', 'WSPM','wd']
    stations = df_train['station'].unique()
    for station in stations:
        for col in columns:
            value = df_train[df_train['station']==station][col].median()
            df_train.loc[df_train['station']==station, col] = df_train[df_train['station']==station][col].fillna(value)
            df_test.loc[df_test['station']==station, col] = df_test[df_test['station']==station][col].fillna(value)
        value = df_train[df_train['station']==station][col].mean()
        df_train.loc[df_train['station']==station, 'NO2'] = df_train[df_train['station']==station]['NO2'].fillna(value)
        df_test.loc[df_test['station']==station, 'NO2'] = df_test[df_test['station']==station]['NO2'].fillna(value)
    return(df_train, df_test)


def get_datetime(df:pd.DataFrame):
    # return a dataframe indexed with datetime
    df_copy = df.copy()
    df_copy.loc[:, "Date-time"] = df_copy.apply(lambda x: datetime.datetime(
    x['year'], x['month'], x['day'], x['hour']), axis=1)
    df_copy.set_index("Date-time", inplace=True)
    df_copy = df_copy.drop(['No'], axis=1)
    return(df_copy)

def monthly_grouped(df:pd.DataFrame, period="D"):
    #to get a dataframe by stations monthly averaged
    df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)
    return(df.groupby('station').resample(period).mean("numeric_only"))

def main(df:pd.DataFrame):
    pipe.fit_transform(df)
    df_train, df_test = splitting(df)
    df_train, df_test = cleaning(df_train, df_test)
    df_train = get_datetime(df_train)
    df_test = get_datetime(df_test)
    return(df_train, df_test)