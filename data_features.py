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

def get_date_feat(data: pd.DataFrame, date_col: list) -> pd.DataFrame:
    for col in date_col:
        data[col + '_year'] = data[col].dt.year
        data[col + '_month'] = data[col].dt.month
        data[col + '_day'] = data[col].dt.day
    # return the modified DataFrame with date features
    return data

def get_statistical_feat(data):

    frequency_column = [col for col in data.columns if 'LL_' in col or 'RL_' in col]
    data["mean"] = data[frequency_column].mean(axis=1)
    data["std"] = data[frequency_column].std(axis=1)

    return data

def normalize(data):

    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = (data[num_cols] - data[num_cols].mean()) / data[num_cols].std()
    
    return data

def get_encoded_feat(data: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:

    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    return data

def add_derived_feat(data: pd.DataFrame) -> pd.DataFrame:
    """
    TODO This might not be necessary yet
    Derive features from existing columns.
    """
    
    # Ratio between seizure and LPD vote
    if 'seizure_vote' in data.columns and 'lpd_vote' in data.columns:
        # Add 1 to avoid division by zero
        data['seizure_to_lpd_ratio'] = data['seizure_vote'] / (data['lpd_vote'] + 1)
    
    # Difference between seizure and GPD vote
    if 'seizure_vote' in data.columns and 'gpd_vote' in data.columns:
        data['seizure_to_gpd_difference'] = data['seizure_vote'] - data['gpd_vote']
    
    # Sum of seizure and other vote
    if 'seizure_vote' in data.columns and 'other_vote' in data.columns:
        data['seizure_and_other_sum'] = data['seizure_vote'] + data['other_vote']

    # Interaction between frequency columns
    freq_cols = [col for col in data.columns if 'LL_' in col or 'RL_' in col]
    if freq_cols:
        # Calculate the product of the first two frequency bands
        data['freq_interaction'] = data[freq_cols[0]] * data[freq_cols[1]]
    
    # Frequency range
    if 'time' in data.columns:
        data['time_range'] = data['time'].max() - data['time'].min()

    return data

