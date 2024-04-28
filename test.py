import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
import time

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
        self.batch_size = 256
        self.classes = 6
        self.fold = 1
        self.class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
        self.verbose = 1
        self.label2name = dict(enumerate(self.class_names))
        self.name2label = {name: i for i, name in enumerate(self.class_names)}

def process_spec(spec_id):
    spec_path = f'train_spectrograms/{spec_id}.parquet'
    spec = pd.read_parquet(spec_path)
    spec = spec.fillna(0).values[:, 1:].T
    spec = spec.astype('float32')
    np.save(f'train_spectrograms/{spec_id}.npy', spec)

def process_npy(file):
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
        print(f"***ERROR PROCESSING {file}: {e}***")

def all_npy(dir):
    # Corrected list comprehension to get all .npy files in the given directory
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npy')]

    # Use joblib for parallel processing
    _ = joblib.Parallel(n_jobs=-1, backend='loky')(
        joblib.delayed(process_npy)(file) for file in tqdm(files, total=len(files))
    )

# Model-related Functions
def load_and_preprocess_image(file, label, image_size):
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

def set_shapes(img, label, img_shape):
    img.set_shape(img_shape)  # Set static shape for TensorFlow optimizations
    label.set_shape([])  # Set shape for label (scalar)
    return img, label

def build_ds(df, batch_size, image_size, shuffle=True, augment=True):
    paths = df['spec2_path'].values
    labels = df['class_label'].values
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda file, label: load_and_preprocess_image(file, label, image_size),
        num_parallel_calls=AUTOTUNE  # Use parallel processing
    )
    img_shape = (image_size[0], image_size[1], 1)  # Define the image shape
    dataset = dataset.map(lambda x, y: set_shapes(x, y, img_shape))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)  # Shuffle with buffer for optimized I/O
    if augment:
        dataset = dataset.map(augmentation, num_parallel_calls=AUTOTUNE)  # Apply augmentation
    return dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)  # Prefetch to improve throughput

def augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

# Main Function to Train and Evaluate the Model
def main():
    config = Config()
    df = pd.read_csv('train.csv')
    df['spec2_path'] = 'train_spectrograms/' + df['spectrogram_id'].astype(str) + '.npy'
    df['class_label'] = df['expert_consensus'].map(config.name2label)

    # Process the existing .npy files
    print("***PROCESSING NPY FILES (WATCH DISK)***")
    all_npy('train_spectrograms')

    # convert .parquet files to .npy
    print("***PROCESSING SPECTROGRAMS (WATCH CPU)***")
    spec_ids = df['spectrogram_id'].drop_duplicates().values
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_spec)(spec_id) for spec_id in tqdm(spec_ids, total=len(spec_ids))
    )

    # training and validation datasets
    train_ds = build_ds(df, config.batch_size, config.image_size, shuffle=True, augment=True)
    val_ds = build_ds(df, config.batch_size, config.image_size, shuffle=False, augment=False)

    # create, train model
    model = create_model((config.image_size[0], config.image_size[1], 1), config.classes)
    history = model.fit(train_ds, epochs=config.epochs, verbose=config.verbose)

    # analyze model
    test_loss, test_acc = model.evaluate(val_ds, verbose=config.verbose)
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Loss: {test_loss}")

    # get predictions
    predictions = model.predict(val_ds)
    print("Predictions:", predictions)

def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    start_time = time.time()  # Start the timer

    main()  # Run the main function

    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed Time: {elapsed_time} seconds")  # Print the elapsed time
