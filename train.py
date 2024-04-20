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
import data_process as dp
import data_info as di
import data_features as df



def main():

    if len(sys.argv) < 3:
        print("Error: Expected 3 arguments, recieved", len(sys.argv))
        print("Usage: python", sys.argv[0], "<train_file> <test_file>")
        print("Exiting ...")
        exit(1)

    train_metadata : pd.DataFrame
    test_metadata : pd.DataFrame
    train_metadata, test_metadata = dp.read_data(sys.argv[1], sys.argv[2])

    di.get_EDA(train_metadata)
    di.get_EDA(test_metadata)

    # =========================== Data Preprocessing ===========================
    # preprocess train/test metadata files, impute missing values
    train_metadata = dp.impute_null(train_metadata)
    test_metadata  = dp.impute_null(test_metadata)

    # merge with spectrogram data
    train_metadata_file = sys.argv[1]
    print("Attempting")
    spectrograms_dir = "eeg-data/train_spectrograms/"
    merged_data = dp.preprocess_data(train_metadata_file, spectrograms_dir)
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
    merged_train_data = dp.preprocess_data(merged_data)

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