"""
data_process.py: Helper function for test.py.
    Loads the data from the training and testing sets. Includes data preprocessing.
"""

__author__ = "Willie Alcaraz"
__credits__ = ["Arian Izadi", "Yohan Dizon"]
__license__ = "MIT License"
__email__ = "willie.alcaraz@gmail.com"

import pandas as pd
import numpy as np
import os
import joblib

# suppress tensorflow specific warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tqdm import tqdm
import tensorflow as tf

from sklearn.impute import KNNImputer
from concurrent.futures import ThreadPoolExecutor      # MOAR POWER


MULTITHREAD : bool  = False  # multithreading flag
TEST_FILE   : str   = None   # change to filename for testing
TEST_MODE   : bool  = True  # testing mode flag
BATCH_SIZE  : int   = 15    # batch size for testing

N_THREADS   : int = os.cpu_count() - 1

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
        train_data = load_data(train_file)
        test_data = load_data(test_file)
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

    # Extract the numeric columns that need imputation
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    # If there are no numeric columns, return the original DataFrame
    if numeric_columns.empty:
        return data

    # Instantiate KNNImputer with a specified number of neighbors
    impute = KNNImputer(n_neighbors=5)

    # Apply KNN imputation only to the numeric columns
    imputed_values = impute.fit_transform(data[numeric_columns])

    # Create a new DataFrame with imputed values
    df_imputed = pd.DataFrame(imputed_values, columns=numeric_columns, index=data.index)

    # Merge non-numeric columns back into the imputed DataFrame
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
    df_final = df_imputed.join(data[non_numeric_columns])

    return df_final

def preprocess_data(metadata_file: str, spectrograms_dir: str) -> pd.DataFrame:
    """
    Preprocesses the metadata file and merges with the spectrogram data.

    Args:
        metadata_file (str): The path to the metadata file.
        spectrograms_dir (str): The directory containing the spectrogram data.

    Returns:
        pd.DataFrame: merged DataFrame
    """


    # read metadata
    metadata = pd.read_csv(metadata_file)

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

def load_and_preprocess_image(file, label, image_size):
    """
    Load and preprocess the image data.
    
    Args:
        file (str): The path to the image file.
        label (int): The label for the image.
        image_size (tuple): The desired size of the image.
    
    Returns:
        image : tf.Tensor: image data
        label : int: image label
    """
    image = tf.io.read_file(file)  # Read the file as a byte array
    image = tf.io.decode_raw(image, tf.float32)  # Decode the raw data into a float32 array
    image = tf.reshape(image, [image_size[0] * image_size[1]])  # Reshape to desired size
    # Adjust the size if required (pad or trim)
    if tf.size(image) < image_size[0] * image_size[1]:
        image = tf.pad(image, [[0, image_size[0] * image_size[1] - tf.size(image)]], constant_values=0)
    else:
        image = image[:image_size[0] * image_size[1]]
    image = tf.reshape(image, [image_size[0], image_size[1], 1])  # Add a channel dimension
    # Normalize the image
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    return image, label


def process_npy(file):
    """
    Process the .npy file by reshaping it to the correct dimensions.
    
    Args:
        file (str): The path to the .npy file.
        
    Return:
        N/A
    """
    try:
        data = np.load(file)
        data = data.ravel()

        needed_elements = 400 * 300
        current_elements = data.size
        if current_elements < needed_elements:
            padding_needed = needed_elements - current_elements
            data = np.pad(data, (0, padding_needed), mode='constant', constant_values=0)
        elif current_elements > needed_elements:
            data = data[:needed_elements]

        data = data.reshape(400, 300)
        np.save(file, data)
        # print(f"***PROCESSED {file}: NEW SHAPE {data.shape}***")
    except Exception as e:
        print(f"\n\033[91mError:\033[0m processing {file}: {e}")

def all_npy(dir):
    """
    Process all .npy files in the given directory.
    
    Args:
        dir (str): The directory containing the .npy files.
    
    Return:
        N/A
    """
    # Corrected list comprehension to get all .npy files in the given directory
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npy')]

    # Use joblib for parallel processing
    _ = joblib.Parallel(n_jobs=N_THREADS, backend='loky')(
        joblib.delayed(process_npy)(file) for file in tqdm(files, total=len(files))
    )

def process_spec(spec_id):
    """
    Preprocess spectrogram data
    
    Args:
        spec_id (str): The spectrogram ID.
    
    Return:
        N/A
    """
    try: 
        spec_path = f'train_spectrograms/{spec_id}.parquet'
        spec = pd.read_parquet(spec_path)

        # find missing values, use KNN to impute
        spec = impute_null(spec)

        spec = spec.fillna(0).values[:, 1:].T
        spec = spec.astype('float32')
        np.save(f'train_spectrograms/{spec_id}.npy', spec)
    except Exception as e:
        print(f"\n\033[91mError:\033[0m processing: {spec_id}: {e}")
    np.save(f'train_spectrograms/{spec_id}.npy', spec)