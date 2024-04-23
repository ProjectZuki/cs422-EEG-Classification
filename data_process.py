"""
Loads the data from the training and testing sets.
"""

import pandas as pd
import numpy as np
import os

from sklearn.impute import KNNImputer
from concurrent.futures import ThreadPoolExecutor      # MOAR POWER

MULTITHREAD : bool = False  # multithreading flag
TEST_FILE : str = None      # change to filename for testing

def read_data(train_file, test_file):
    """
    Read data from train/test sets.

    Args:
        train_file (str): The path to the training data file.
        test_file (str): The path to the testing data file.

    Returns:
        pd.DataFrame: The contents of the training data file.
        pd.DataFrame: The contents of the testing data file.
    """
    try:
        train_data = load_data('eeg-data/' + train_file)
        test_data = load_data('eeg-data/' + test_file)
    except FileNotFoundError as e:
        print("File not found: ", e)
        print("Exiting ...")
        exit(1)

    return train_data, test_data

def load_data(file):
    """
    Load training data
    
    Args:
        file (str): Path to the file containing training data.
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded training data.
    """
    # Check file type, raise exception if not supported
    if file.endswith('.csv'):
        return pd.read_csv(file)
    elif file.endswith('.parquet'):
        return pd.read_parquet(file)
    else:
        raise ValueError("Unsupported file format. Only .csv and .parquet files are supported.")

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

    # for testing with single file
    if TEST_FILE is not None:
        spectrogram_id = TEST_FILE
        print("Preprocssing:", spectrogram_id)
        spectrogram_file = os.path.join(spectrograms_dir, f"{spectrogram_id}.parquet")
        spectrogram_data = pd.read_parquet(spectrogram_file)
        merge_row = pd.concat([pd.Series({'spectrogram_id': spectrogram_id}), spectrogram_data.mean(axis=0)], axis=0)

        return pd.DataFrame([merge_row])

    # ========================= multi-threaded method ==========================
    if MULTITHREAD:
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

    # ============================= Single-threaded ============================
    else:
        # merge files
        merged_data = pd.DataFrame()

        for index, row in metadata.iterrows():
            # get metadata rows (relevant)
            spectrogram_id = row['spectrogram_id']        # spectrogram_id from metadata
            # patient_id = row['patient_id']              # patient_id if necessary

            # read spectrogram data
            print("Preprocessing:", spectrogram_id)
            spectrogram_file = os.path.join(spectrograms_dir, f"{spectrogram_id}.parquet")
            spectrogram_data = pd.read_parquet(spectrogram_file)

            # merge metadata with spectrogram data
            merge_row = pd.concat([pd.Series(row), spectrogram_data], axis=0)
            # if utilizing optional patient_id
            # merged_row = pd.concat([pd.Series(row), pd.Series({'patient_id': patient_id}), spectrogram_data], axis=0)
            merged_data = merged_data._append(merge_row, ignore_index=True)
    # ==========================================================================
    return merged_data