import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
import datetime

def get_poly_feat(data: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """
    Generates polynomial features from the dataframe numerical columns.
    
    Args: 
        data (pd.DataFrame): The dataframe.
        degree (int): degree of polynomial features.
        
    Returns:
        pd.DataFrame: dataframe with the required polynomial features
    """
    # polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    # fit/transform data
    poly_data = poly.fit_transform(data)
    # feature names from columns
    columns = poly.get_feature_names(data.columns)
    return pd.DataFrame(poly_data, columns=columns, index=data.index)

def get_date_feat(data: pd.DataFrame, date_col: list) -> pd>DataFrame:
    for col in date_col:
        data[col + '_year'] = data[col].dt.year
        data[col + '_month'] = data[col].dt.month
        data[col + '_day'] = data[col].dt.day
    # return the modified DataFrame with date features
    return data