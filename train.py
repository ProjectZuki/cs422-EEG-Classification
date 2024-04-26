"""
Test implementation for training model
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
import time

# import tensorflow as tf
# import tensorflow.keras.layers as layers
# from tensorflow.keras.models import Model
# import tensorflow.keras.optimizers as optimizers
# import tensorflow.keras.backend as K

from sklearn.impute import KNNImputer
from concurrent.futures import ThreadPoolExecutor      # MOAR POWER
from sklearn.model_selection import train_test_split

# user defined modules
import data_process as dp
import data_info as di
import data_features as df

def get_time(start_time):
    # end time
    end_time = time.time()
    print()
    print("-" * 65)
    execution_time = end_time - start_time

    # convert time to HH:MM:SS.mmm format
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int((execution_time % 60) // 1)
    milliseconds = int((execution_time % 1) * 1000)

    # format to 2 digits (3 for milliseconds) with leading zeros
    formatted_hours = f"{hours:02}"
    formatted_minutes = f"{minutes:02}"
    formatted_seconds = f"{seconds:02}"
    formatted_milliseconds = f"{milliseconds:02}"

    # print execution time
    print("Execution time (HH:MM:SS.mmm): \033[92m{}:{}:{}.{}\033[0m".format(formatted_hours, formatted_minutes, formatted_seconds, formatted_milliseconds))
    print("-" * 65)
    print()

def main():

    # check args

    if len(sys.argv) < 3:
        print("\n\033[91mError:\033[0m Expected 3 arguments, recieved", len(sys.argv))
        print("Usage: python", sys.argv[0], "<train_file> <test_file>")
        print("For multithreading, add -m flag as 4th arg")
        print("\033[91m\nExiting ...\n\033[0m")
        exit(1)

    if len(sys.argv) == 4:
        if sys.argv[3] == "-m":
            dp.MULTITHREAD = True
            print("Running with multithreading. CPU count:\033[92m", os.cpu_count(), "\033[0m")
        else:
            print("\n\033[91mError:\033[0m", sys.argv[3], "is not a valid flag")
            print("Usage: python", sys.argv[0], "<train_file> <test_file>")
            print("For multithreading, add -m flag as 4th arg")
            print("\033[91m\nExiting ...\n\033[0m")
            exit(1)

    train_metadata : pd.DataFrame
    test_metadata : pd.DataFrame
    train_metadata, test_metadata = dp.read_data(sys.argv[1], sys.argv[2])

    # di.get_EDA(train_metadata)
    # di.get_EDA(test_metadata)

    # =========================== Data Preprocessing ===========================
    # preprocess train/test metadata files, impute missing values
    train_metadata = dp.impute_null(train_metadata)
    test_metadata  = dp.impute_null(test_metadata)

    # merge with spectrogram data
    train_metadata_file = sys.argv[1]

    print()
    print("==================== Preprocessing data ... =====================")
    spectrograms_dir = "eeg-data/train_spectrograms/"
    merged_data = dp.preprocess_data(train_metadata_file, spectrograms_dir)
    print("========================== Finished =============================")

    # # preprocess spectrogram data
    # merged_train_data = dp.preprocess_data(merged_data)

    # ======================== Split train/validation ==========================
    # target_col = 'expert_consensus'
    # train_set, validation_set = train_test_split(merged_data, 
    #                                             test_size=0.2,
    #                                             stratify=merged_data[target_col],   # handle imbalanced data
    #                                             random_state=42)                    # for reproducibility

    # print("Train Set:\n", train_set.head())
    # print("Validation Set:\n", validation_set.head())

if __name__ == "__main__":
    # set logging level
    logging.basicConfig(level=logging.INFO)

    # start timing
    start_time = time.time()

    main()

    get_time(start_time)

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