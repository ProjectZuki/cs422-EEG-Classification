"""
Test implementation for training model
"""

import pandas as pd
import numpy as np
import sys
import os
import logging

# import tensorflow as tf
# import tensorflow.keras.layers as layers
# from tensorflow.keras.models import Model
# import tensorflow.keras.optimizers as optimizers
# import tensorflow.keras.backend as K

from sklearn.impute import KNNImputer

# user defined modules
import data_load as dl
import data_info as di

def impute_null(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using KNN to determine the value of the missing data.

    Args:
        data (pd.DataFrame): The DataFrame to impute.

    Returns:
        pd.DataFrame: The DataFrame with imputed values.
    """

    if data.empty:
        return data

    # manually define column names
    columns = data.columns

    # exclude non-numeric columns (i.e. expert_concensus column)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data_numeric = data[numeric_columns]

    # use KNN to impute values
    impute = KNNImputer(n_neighbors=5)
    imputed_values = impute.fit_transform(data_numeric)

    df = pd.DataFrame(imputed_values, columns=numeric_columns, index=data_numeric.index)

    # combine non-numeric columns with imputed values
    for col in columns:
        if col not in numeric_columns:
            df[col] = data[col]

    return df



def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    # handle missing values
    train_data = impute_null(train_data)
    test_data = impute_null(test_data)

    return train_data, test_data

def main():

    if len(sys.argv) < 3:
        print("Error: Expected 3 arguments, recieved", len(sys.argv))
        print("Usage: python", sys.argv[0], "<train_file> <test_file>")
        print("Exiting ...")
        exit(1)

    train_data : pd.DataFrame
    test_data : pd.DataFrame
    train_data, test_data = dl.read_data(sys.argv[1], sys.argv[2])

    di.get_EDA(train_data)
    di.get_EDA(test_data)

    # =========================== Data Preprocessing ===========================
    train_data, test_data = preprocess_data(train_data, test_data)
    # ============================ Categorical Data ============================
    # TODO
    # ========================== Feature Engineering ===========================
    # TODO

if __name__ == "__main__":
    main()


"""
    TODO
    
    Data Loading: x
    Exploratory Data Analysis (EDA): x
    Data Preprocessing: x
    Feature Engineering:
    Model Training:
    Model Evaluation:
    Prediction:
"""