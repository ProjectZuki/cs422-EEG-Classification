"""
Reads the data from the training and testing sets.

Output:
    Contents of the data files
"""

import pandas as pd
import sys

def get_info(train_file, test_file):
    """
    Reads the data from the training and testing sets.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        pd.DataFrame: The contents of the data file.
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Print the contents of the data files
    print ("Test Data:")
    print(test_df.head())       # print first rows of test DataFrame
    print(test_df.info())       # print information about test DataFrame

    print("\n")
    print ("Train Data:")
    print(train_df.head())      # print first rows of train DataFrame
    print(train_df.info())      # print information about train DataFrame

def get_EDA(data):
    # Exploratory Data Analysis
    print(data.head())
    print("-" * 50)
    print(data.info())
    print("-" * 50)
    print(data.describe())
    print("-" * 50)
    