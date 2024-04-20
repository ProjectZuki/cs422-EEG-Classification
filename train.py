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
from concurrent.futures import ThreadPoolExecutor      # MOAR POWER

# user defined modules
import data_load as dl
import data_info as di
import data_features as df

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

def preprocess_data(metadata_file: str, spectrograms_dir: str) -> pd.DataFrame:
    """
    Preprocesses the metadata file and merges with the spectrogram data.
    """

    # read metadata
    metadata = pd.read_csv("eeg-data/" + f"{metadata_file}")

    # thread worker function
    def process_row(row):
        spectrogram_id = row['spectrogram_id']

        print("Preprocessing:", spectrogram_id)

        spectrogram_file = os.path.join(spectrograms_dir, f"{spectrogram_id}.parquet")
        spectrogram_data = pd.read_parquet(spectrogram_file)
        merge_row = pd.concat([pd.Series(row), spectrogram_data], axis=0)
        return merge_row

    with ThreadPoolExecutor() as executor:
        print("preprocessing:", metadata_file, "| Thread Count:", executor._max_workers)
        # process task
        futures = [executor.submit(process_row, row) for _, row in metadata.iterrows()]
        # results from task completion
        merged_data = pd.concat([future.result() for future in futures], ignore_index=True)

        # Shutdown the ThreadPoolExecutor to ensure proper cleanup
        executor.shutdown()

    # # merge files
    # merged_data = pd.DataFrame()

    # for index, row in metadata.iterrows():
    #     # get metadata rows (relevant)
    #     spectrogram_id = row['spectrogram_id']        # spectrogram_id from metadata
    #     # patient_id = row['patient_id']              # patient_id if necessary

    #     # read spectrogram data
    #     print("Preprocessing:", spectrogram_id)
    #     spectrogram_file = os.path.join(spectrograms_dir, f"{spectrogram_id}.parquet")
    #     spectrogram_data = pd.read_parquet(spectrogram_file)

    #     # merge metadata with spectrogram data
    #     merge_row = pd.concat([pd.Series(row), spectrogram_data], axis=0)
    #     # if utilizing optional patient_id
    #     # merged_row = pd.concat([pd.Series(row), pd.Series({'patient_id': patient_id}), spectrogram_data], axis=0)
    #     merged_data = merged_data._append(merge_row, ignore_index=True)

    return merged_data

def main():

    if len(sys.argv) < 3:
        print("Error: Expected 3 arguments, recieved", len(sys.argv))
        print("Usage: python", sys.argv[0], "<train_file> <test_file>")
        print("Exiting ...")
        exit(1)

    train_metadata : pd.DataFrame
    test_metadata : pd.DataFrame
    train_metadata, test_metadata = dl.read_data(sys.argv[1], sys.argv[2])

    di.get_EDA(train_metadata)
    di.get_EDA(test_metadata)

    # =========================== Data Preprocessing ===========================
    # preprocess train/test metadata files, impute missing values
    train_metadata = impute_null(train_metadata)
    test_metadata  = impute_null(test_metadata)

    # merge with spectrogram data
    train_metadata_file = sys.argv[1]
    print("Attempting")
    spectrograms_dir = "eeg-data/train_spectrograms/"
    merged_data = preprocess_data(train_metadata_file, spectrograms_dir)
    print("finished")

    """
    NOTE: Spectrogram data contains time values, as well as the following:
        LL - Left Lateral
        RL - Right Lateral
        LP - Left Parasagittal
        RP - Right Parasagittal

        eeg_data/train_spectrograms/ directory contains {spectrogram_id}.parquet files
        spectrogram_id is provided by train.csv metadata
    """
    # preprocess spectrogram data
    merged_train_data = preprocess_data(merged_data)

    # ========================== Feature Engineering ===========================
    # # generate polynomial features from data
    # train_metadata_poly = df.get_poly_feat(train_metadata)
    # test_metadata_poly = df.get_poly_feat(test_metadata)

    # # manually set columns
    # date_cols = ['date_col_1', 'date_col_2']
    # # date features
    # train_metadata = df.get_date_feat(train_metadata, date_cols)
    # test_metadata = df.get_date_feat(test_metadata, date_cols

    # ============================ Model Training =============================


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