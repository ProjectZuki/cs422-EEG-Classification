"""
Test implementation for training model
"""

import pandas as pd
import numpy as np
import sys
import os

# user defined modules
import data_load as dl

# import tensorflow as tf
# import tensorflow.keras.layers as layers
# from tensorflow.keras.models import Model
# import tensorflow.keras.optimizers as optimizers
# import tensorflow.keras.backend as K

from sklearn.model_selection import KFold

def main():

    if len(sys.argv) < 3:
        print("Usage: python", sys.argv[0], "<train_file> <test_file>")
        print("Exiting ...")
        exit(1)

    train_data : pd.DataFrame
    test_data : pd.DataFrame
    train_data, test_data = dl.read_data(sys.argv[1], sys.argv[2])

    # Exploratory Data Analysis
    print(train_data.head())
    print("-" * 50)
    print(train_data.info())
    print("-" * 50)
    print(train_data.describe())
    print("-" * 50)

if __name__ == "__main__":
    main()


"""
    TODO
    
    Data Loading: x
    Exploratory Data Analysis (EDA): 
    Data Preprocessing:
    Feature Engineering:
    Model Training:
    Model Evaluation:
    Prediction:
"""