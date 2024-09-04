# Author: Aidan Elm | 2024-08-02

""" Check documentation for compatible versions. """

import sys # Exit
import os
import argparse
import pathlib  # File operations
from typing import List # Type hints
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio  # DICOM support
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Normalize a DICOM image based on window center and width
def window_image(image: tf.Tensor, window_center: int, window_width: int) -> tf.Tensor:

    """
    @param image: tf.Tensor - The input image tensor.
    @param window_center: int - The center of the window for normalization.
    @param window_width: int - The width of the window for normalization.
    @return: tf.Tensor - The normalized image tensor.
    """

    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    image = tf.clip_by_value(image, clip_value_min=img_min, clip_value_max=img_max)
    return image

# Decode a DICOM file into a normalized image tensor
def decode_dicom(file_path: str, window_center: int, window_width: int, image_height: int,
                    image_width: int) -> tf.Tensor:

    """
    @param file_path: str - Path to the DICOM file.
    @param window_center: int - The center of the window for normalization.
    @param window_width: int - The width of the window for normalization.
    @param image_height: int - Desired height of the output image.
    @param image_width: int - Desired width of the output image.
    @return: tf.Tensor - The decoded and normalized image tensor.
    """

    # Ensure file exists
    if not os.path.isfile(file_path):
        print('Error: File does not exist.')
        sys.exit()

    # Decode image
    try:
        image_bytes = tf.io.read_file(file_path)
        image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
        image = window_image(image, window_center, window_width)
        image = tf.image.resize(image, [image_height, image_width])
        image = tf.cast(image, tf.float32) / 65535.0  # Normalize to [0, 1]
        return image
    except tf.errors.InvalidArgumentError as e:
        print(f'Error decoding DICOM file {file_path}: {e}')
        sys.exit()

# Train a model on DICOM images
def train(
    dcm_dir: str,
    num_epochs: int,
    batch_size: int,
    class_names: List[str],
    window_center: int,
    window_width: int,
    image_height: int,
    image_width: int,
    model_path: str,
) -> None:

    """
    @param dcm_dir: str - Directory containing DICOM images for training.
    @param num_epochs: int - Number of training epochs.
    @param batch_size: int - Batch size for training.
    @param class_names: List[str] - List of class names for prediction.
    @param window_center: int - The center of the window for normalization.
    @param window_width: int - The width of the window for normalization.
    @param image_height: int - Desired height of the input images.
    @param image_width: int - Desired width of the input images.
    @param model_path: str - Path to save the trained model.
    @return: None - Saves the trained model.
    """

    # Load directory of DICOM images
    if not os.path.isdir(dcm_dir):
        print('Error: Training directory does not exist.')
        sys.exit()
    data_dir = pathlib.Path(dcm_dir)

    # List files in dataset
    image_paths = list(data_dir.glob('*/*.dcm'))
    image_count = len(image_paths)

    # Shuffle and split dataset
    np.random.shuffle(image_paths)
    split_index = int(image_count * 0.8) # 80/20 split
    train_image_paths = image_paths[:split_index]
    val_image_paths = image_paths[split_index:]

    # Create training dataset
    train_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)
    train_ds = train_ds.map(
        lambda x: tf.py_function(func=decode_dicom,
                                  inp=[x, window_center, window_width,
                                       image_height, image_width], Tout=tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Cache, batch, and buffer training dataset
    train_ds = (
        train_ds.cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Create validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices(val_image_paths)
    val_ds = val_ds.map(
        lambda x: tf.py_function(func=decode_dicom,
                                  inp=[x, window_center, window_width,
                                       image_height, image_width], Tout=tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Cache, batch, and buffer validation dataset
    val_ds = (
        val_ds.cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Create Sequential model
    model = Sequential(
        [
            layers.InputLayer(input_shape=(image_height, image_width, 1)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(class_names)),
        ]
    )

    # Compile and fit the model - adam optimizer with sparse categorical cross-entropy
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    model.fit(train_ds, validation_data = val_ds, epochs=num_epochs)

    # Save the model
    try:
        model.save(model_path)
    except OSError as e:
        print(f'Error: Model could not be saved. Details: {e}')
        sys.exit()

# Predict the class of an image using a trained model
def predict(
    dcm_path: str,
    model_path: str,
    class_names: List[str],
    window_center: int,
    window_width: int,
    image_height: int,
    image_width: int,
) -> None:

    """
    @param dcm_path: str - Path to the DICOM image for prediction.
    @param model_path: str - Path to the saved model.
    @param class_names: List[str] - List of class names for prediction.
    @param window_center: int - The center of the window for normalization.
    @param window_width: int - The width of the window for normalization.
    @param image_height: int - Desired height of the input image.
    @param image_width: int - Desired width of the input image.
    @return: None - Prints the class and confidence of the prediction.
    """

    # Load saved model
    try:
        model = tf.keras.models.load_model(model_path)
    except OSError as e:
        print(f'Error loading model. Details: {e}')
        sys.exit()

    # Load image
    try:
        image_bytes = tf.io.read_file(dcm_path)
    except OSError as e:
        print(f'Error loading image. Details: {e}')
        sys.exit()

    # Decode and format the image
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
    image = window_image(image, window_center, window_width)
    image = tf.image.resize(image, [image_height, image_width])
    image = tf.cast(image, tf.float32) / 65535.0  # Normalize to [0, 1]
    image = tf.expand_dims(image, 0)  # Create a batch dimension

    # Print prediction
    prediction = tf.nn.softmax(model.predict(image)[0])
    print(f'Class: {class_names[np.argmax(prediction)]}\
            | Confidence: {100 * np.max(prediction):.2f}%')

# Call training or prediction function
if __name__ == '__main__':

    # Create argument parser
    parser = argparse.ArgumentParser(description='Train or predict DICOM images.')

    # Add arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict'],
        default='train',
        help='Mode to run the script in: train or predict.',
    )
    parser.add_argument(
        '--dcm_dir',
        type=str,
        default='images_dcm/',
        help='Directory containing DICOM images for training.',
    )
    parser.add_argument(
        '--epochs', type=int, default=2, help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1, help='Batch size for training.'
    )
    parser.add_argument(
        '--class_names',
        nargs='+',
        default=['Brain', 'Lung'],
        help='Space-separated list of class names for prediction (e.g., --class_names Brain Lung).',
    )
    parser.add_argument(
        '--dcm_path',
        type=str,
        default='dcm_image.dcm',
        help='Path to a DICOM image for prediction.',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='saved_model.keras',
        help='Path to save or load the model.',
    )
    parser.add_argument(
        '--window_center',
        type=int,
        default=50,
        help='Window center for DICOM image normalization.',
    )
    parser.add_argument(
        '--window_width',
        type=int,
        default=350,
        help='Window width for DICOM image normalization.',
    )
    parser.add_argument(
        '--image_height', type=int, default=512, help='Height of the images.'
    )
    parser.add_argument(
        '--image_width', type=int, default=512, help='Width of the images.'
    )

    args = parser.parse_args() # Parse arguments

    # Call function
    if args.mode == 'train':
        train(
            args.dcm_dir,
            args.epochs,
            args.batch_size,
            args.class_names,
            args.window_center,
            args.window_width,
            args.image_height,
            args.image_width,
            args.model_path,
        )
    elif args.mode == 'predict':
        predict(
            args.dcm_path,
            args.model_path,
            args.class_names,
            args.window_center,
            args.window_width,
            args.image_height,
            args.image_width,
        )
    else:
        print('Error: Mode must be \'train\' or \'predict\'.')
