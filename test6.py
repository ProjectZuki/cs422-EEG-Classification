import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

class Config:
    def __init__(self):
        self.image_size = [400, 300]
        self.epochs = 1
        self.batch_size = 64
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
    # data = np.load(file)
    # if data.shape[1] < 300:
    #     print("file name/id: ", file , "shape: ", data.shape)
    #     padding = ((0, 0), (0, 300 - data.shape[1]))
    #     data = np.pad(data, padding, mode = 'constant', constant_values=0)
    # elif data.shape[1] > 300:
    #     data = data[:, :300]
    # data = data[:400, :300]
    # np.save(file, data)

    data = np.load(file)
    current_elements = data.size
    needed_elements = 400 * 300
    print(f"***PROCESSING {file}: {current_elements} ELEMENTS FOUND***")
    if current_elements < needed_elements:
        print(f"***SKIPPING {file} INSUFFICIENT DATA SIZE ({current_elements} < {needed_elements})***")
        return
    elif current_elements > needed_elements:
        data = data.ravel()[:needed_elements]
        print(f"***TRUNCATING {file} TO {needed_elements} ELEMENTS***")
    try:
        data = data.reshape(400, 300)
    except ValueError as e:
        print(f"***RESHAPE ERROR ON {file}: {str(e)}***")
        return
    np.save(file, data)
    print(f"***SAVED {file} POST RESHAPE***")

def all_npy(dir):
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npy')]
    _ = joblib.Parallel(n_jobs=-1, backend='loky')(
        joblib.delayed(process_npy)(file) for file in tqdm(files, total=len(files))
    )

def load_and_preprocess_image(file, label, image_size):
    # image = tf.io.read_file(file)
    # image = tf.io.decode_raw(image, tf.float32)
    # image = tf.reshape(image, [image_size[0], image_size[1], 1])
    # image = tf.image.resize(image, [image_size[0], image_size[1]])  
    # image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    # image.set_shape([image_size[0], image_size[1], 1])  
    
    # return image, label

    #delete below if error
    # image = tf.io.read_file(file)
    # image = tf.io.decode_raw(image, tf.float32)
    # original = [400, -1, 1]
    # image = tf.reshape(image, original)
    # image = tf.image.resize(image, [image_size[0], image_size[1]])
    # image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    # image.set_shape([image_size[0], image_size[1], 1])
    # return image, label

    image = tf.io.read_file(file)
    image = tf.io.decode_raw(image, tf.float32)
    image = tf.reshape(image, [400, 300, 1])
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    return image, label


def set_shapes(img, label, img_shape):
    img.set_shape(img_shape)
    label.set_shape([])
    return img, label

def build_ds(df, batch_size, image_size, shuffle=True, augment=True):
    paths = df['spec2_path'].values
    labels = df['class_label'].values
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda file, label: load_and_preprocess_image(file, label, image_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    img_shape = (image_size[0], image_size[1], 1)
    dataset = dataset.map(lambda x, y: set_shapes(x, y, img_shape))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1024)
    if augment:
        dataset = dataset.map(augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

def main():
    config = Config()
    df = pd.read_csv('train.csv')
    df['spec2_path'] = 'train_spectrograms/' + df['spectrogram_id'].astype(str) + '.npy'
    df['class_label'] = df['expert_consensus'].map(config.name2label)
    dimensions = np.load('train_spectrograms/353733.npy')
    print(dimensions.shape)
    # print(dimensions)

    # PROCESSING ALL NPY FILES TO CONVERT TO PROPER FORMAT
    print("***PROCESSING NPY FILES (WATCH DISK)***")
    all_npy('train_spectrograms')

    print("***PROCESSING SPECTROGRAMS (WATCH CPU)***")
    spec_ids = df['spectrogram_id'].drop_duplicates().values
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_spec)(spec_id) for spec_id in tqdm(spec_ids, total=len(spec_ids))
    )

    train_ds = build_ds(df, config.batch_size, config.image_size, shuffle=True, augment=True)
    val_ds = build_ds(df, config.batch_size, config.image_size, shuffle=False, augment=False)

    model = create_model((config.image_size[0], config.image_size[1], 1), config.classes)
    history = model.fit(train_ds, epochs=config.epochs)
    test_loss, test_acc = model.evaluate(val_ds, verbose=config.verbose)
    print(f"test accuracy: {test_acc}")
    print(f"test loss: {test_loss}")
    predictions = model.predict(val_ds)
    print (predictions)

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
    main()
