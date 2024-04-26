"""
Loads the data from the training and testing sets.
"""

import pandas as pd
import numpy as np
import os

from sklearn.impute import KNNImputer
from concurrent.futures import ThreadPoolExecutor      # MOAR POWER

MULTITHREAD : bool  = False  # multithreading flag
TEST_FILE   : str   = None   # change to filename for testing
TEST_MODE   : bool  = True  # testing mode flag
BATCH_SIZE  : int   = 1    # batch size for testing

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
    metadata = pd.read_csv("eeg-data/" + metadata_file)

    if TEST_MODE:
        batch = metadata[:BATCH_SIZE]
    else:
        batch = metadata

    # helper function to process single row of data
    def process_file(spectrogram_id):
            print("Processing spectrogram:", spectrogram_id)

            # Load the corresponding spectrogram data
            spectrogram_file = os.path.join(spectrograms_dir, f"{spectrogram_id}.parquet")
            spectrogram_data = pd.read_parquet(spectrogram_file)

            # Find the corresponding metadata for this spectrogram
            metadata_row = metadata[metadata['spectrogram_id'] == spectrogram_id]

            # return empty dataframe if no match
            if metadata_row.empty:
                print(f"Warning: No metadata found for spectrogram_id {spectrogram_id}")
                return pd.DataFrame()
        
            # Get the first (and presumably only) row
            metadata_row = metadata_row.iloc[0]

            # Ensure 'expert_consensus' exists in metadata
            if 'expert_consensus' not in metadata_row.index:
                print(f"Error: 'expert_consensus' is missing for spectrogram_id {spectrogram_id}")
                return pd.DataFrame()  # Return an empty DataFrame to avoid errors
            
            # Add 'expert_consensus' as the target label to spectrogram data
            spectrogram_data['label'] = metadata_row['expert_consensus']

            return spectrogram_data
    
    merged_data = pd.DataFrame()
    # ========================= multi-threaded method ==========================
    if MULTITHREAD:
        with ThreadPoolExecutor() as executor:
            # Process the batch in parallel
            futures = [executor.submit(process_file, row['spectrogram_id']) for _, row in batch.iterrows()]

            # Collect results and concatenate
            results = [future.result() for future in futures if not future.result().empty]  # Exclude empty results
            merged_data = pd.concat(results, ignore_index=True)

    # ============================= Single-threaded ============================
    else:
        for _, row in batch.iterrows():
            spectrogram_id = row['spectrogram_id']
            processed_data = process_file(spectrogram_id)
            if not processed_data.empty:
                merged_data = pd.concat([merged_data, processed_data], ignore_index=True)
    # ==========================================================================

    if 'label' in merged_data.columns:
        print("Success: 'expert_consensus' has been merged as 'label'")
        print("Label: " + str(merged_data['label'].unique()))
    else:
        print("Error: 'label' (expert_consensus) is missing in merged data")

    return merged_data