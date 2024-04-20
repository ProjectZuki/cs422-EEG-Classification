"""
Loads the data from the training and testing sets.
"""

import pandas as pd

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

# def load_test_data(test_file):
#     """
#     Load test data
    
#     Args:
#         test_file (str): Path to the CSV file containing test data.
        
#     Returns:
#         pd.DataFrame: DataFrame containing the loaded test data.
#     """
#     if 
#     return pd.read_csv(test_file)

