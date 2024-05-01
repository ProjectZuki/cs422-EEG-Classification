import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import warnings
import sys

# suppress tensorflow specific warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from sklearn.model_selection import KFold

# custom imports
import helper       as hp
import model        as md
import data_process as dp

# Set GPU memory growth to avoid allocating all GPU memory at once
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


# Data Processing Functions
class Config:
    def __init__(self):
        self.image_size = [400, 300]
        self.epochs = 1
        self.batch_size = 128
        self.classes = 6
        self.fold = 1
        self.class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
        self.verbose = 1
        self.label2name = dict(enumerate(self.class_names))
        self.name2label = {name: i for i, name in enumerate(self.class_names)}

def main():
    config = Config()
    df = pd.read_csv('train.csv')
    df['spec2_path'] = 'train_spectrograms/' + df['spectrogram_id'].astype(str) + '.npy'
    df['class_label'] = df['expert_consensus'].map(config.name2label)

    print("CPU count: %d" % dp.N_THREADS)

    # Process the existing .npy files
    print("***PROCESSING NPY FILES (WATCH DISK)***")
    dp.all_npy('train_spectrograms')

    # convert .parquet files to .npy
    print("***PROCESSING SPECTROGRAMS (WATCH CPU)***")
    spec_ids = df['spectrogram_id'].drop_duplicates().values
    _ = joblib.Parallel(n_jobs=dp.N_THREADS, backend="loky")(
        joblib.delayed(dp.process_spec)(spec_id) for spec_id in tqdm(spec_ids, total=len(spec_ids))
    )

    # kf = KFold(n_splits=config.fold, shuffle=True, random_state=42)
    # for i, (train_index, val_index) in enumerate(kf.split(df)):
    #     print(f"Processing Fold {i + 1}")
    #     train_df = df.iloc[train_index]
    #     val_df = df.iloc[val_index]
    #     train_ds = md.build_ds(
    #         train_df, config.batch_size, config.image_size, shuffle=True, augment=True
    #     )
    #     val_ds = md.build_ds(
    #         val_df, config.batch_size, config.image_size, shuffle=False, augment=False
    #     )
    #     model = md.create_model(
    #         (config.image_size[0], config.image_size[1], 1), config.classes
    #     )
    #     history = model.fit(train_ds, epochs=config.epochs)
    #     test_loss, test_acc = model.evaluate(val_ds, verbose=config.verbose)
    #     print(f"Fold {i + 1} completed")
    #     print(f"test accuracy: {test_acc}")
    #     print(f"test loss: {test_loss}")
    #     predictions = model.predict(val_ds)
    #     print(predictions)

    # training and validation datasets
    train_ds = md.build_ds(df, config.batch_size, config.image_size, shuffle=True, augment=True)
    val_ds = md.build_ds(df, config.batch_size, config.image_size, shuffle=False, augment=False)

    # create, train model
    model = md.create_model((config.image_size[0], config.image_size[1], 1), config.classes)
    history = model.fit(train_ds, epochs=config.epochs, verbose=config.verbose)

    # analyze model
    test_loss, test_acc = model.evaluate(val_ds, verbose=config.verbose)
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Loss: {test_loss}")

    # get predictions
    predictions = model.predict(val_ds)
    print("Predictions:", predictions)



if __name__ == '__main__':
    start_time = time.time()  # Start the timer

    main()  # Run the main function

    # show execution time
    hp.exec_time(start_time)