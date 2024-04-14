"""
Reads the data from the training and testing sets.

Output:
    Contents of the data files
"""

import pandas as pd
import sys

def read_data(train_file, test_file):
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

def main():
    if len(sys.argv) < 3:
        print("Usage: python", sys.argv[0], "<train_file> <test_file>")
        exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    read_data(train_file, test_file)

if __name__ == "__main__":
    main()