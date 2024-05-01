"""
model.py: Helper file for test.py that includes functions for building TensorFlow dataset
    and creating CNN model.
"""

__author__ = "Willie Alcaraz"
__credits__ = ["Arian Izadi", "Yohan Dizon"]
__license__ = "MIT License"
__email__ = "willie.alcaraz@gmail.com"

import numpy as np
import pandas as pd
import os

# suppress tensorflow specific warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE

import data_process as dp

TEST_MODE = False

def set_shapes(img, label, img_shape):
    """
    Sets shapes for image and label tensors.
    
    Args:
        img (tf.Tensor): The image tensor.
        label (tf.Tensor): The label tensor.
        img_shape (tuple): The shape of the image tensor.
        
    Returns:
        tf.Tensor: The image tensor with the specified shape.
        tf.Tensor: The label tensor with the specified shape.
    """
    img.set_shape(img_shape)  # Set static shape for TensorFlow optimizations
    label.set_shape([])  # Set shape for label (scalar)
    return img, label

def augmentation(image, label):
    """
    Apply augmentation to the image tensor.

    Args:
        image (tf.Tensor): The image tensor.
        label (tf.Tensor): The label tensor.
    
    Returns:
        tf.Tensor: The augmented image tensor.
        tf.Tensor: The label tensor.
    """

    image = tf.image.random_flip_left_right(image)
    return image, label

def build_ds(df, batch_size, image_size, shuffle=True, augment=True):
    """
    Builds a TensorFlow dataset from the input DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        batch_size (int): The batch size.
        image_size (tuple): The image size.
        shuffle (bool): Whether to shuffle the dataset.
        augment (bool): Whether to apply augmentation.
    
    Returns:
        tf.data.Dataset: TensorFlow dataset.
    """
    paths = df['spec2_path'].values
    labels = df['class_label'].values
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda file, label: dp.load_and_preprocess_image(file, label, image_size),
        num_parallel_calls=AUTOTUNE  # Use parallel processing
    )
    img_shape = (image_size[0], image_size[1], 1)  # Define the image shape
    dataset = dataset.map(lambda x, y: set_shapes(x, y, img_shape))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)  # Shuffle with buffer for optimized I/O
    if augment:
        dataset = dataset.map(augmentation, num_parallel_calls=AUTOTUNE)  # Apply augmentation
    return dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)  # Prefetch to improve throughput

def create_model(input_shape, num_classes):
    """
    Creates model for convolutional neural network.
    
    Args:
        input_shape (tuple): The input shape for the model.
        num_classes (int): The number of classes.
    
    Returns:
        tf.keras.Sequential: The model.
    """

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