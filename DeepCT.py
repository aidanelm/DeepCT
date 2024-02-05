# Import all necessary modules
import os
import argparse
import numpy as np
import shutil # File operations
import pathlib # File operations
import tensorflow as tf
import tensorflow_io as tfio # DICOM support
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Normalize DICOM images
def window_image(image, window_center, window_width):

    # Assuming image is a TensorFlow tensor
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    image = tf.clip_by_value(image, clip_value_min=img_min, clip_value_max=img_max)

    return image


# Train a model
def train(
    dcm_dir,
    epochs,
    batch_size,
    class_names,
    window_center,
    window_width,
    image_height,
    image_width,
    model_path,
):

    # Check if directory exists, exit if it does not
    if not os.path.isdir(dcm_dir):
        print("Error: Training directory does not exist.")
        return

    # Directory of input images
    data_dir = pathlib.Path(dcm_dir)

    # Decode DICOM image
    def decode_dicom(file_path, window_center, window_width, image_height, image_width):
        if not os.path.isfile(file_path):
            print("Error: File does not exist.")
            return None

        try:

            # Read file and decode DICOM
            image_bytes = tf.io.read_file(file_path)
            image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)

            # Resize and normalize the image
            image = window_image(image, window_center, window_width)
            image = tf.image.resize(image, [image_height, image_width])
            image = tf.cast(image, tf.float32) / 65535.0  # Normalize to [0, 1]

            return image

        except tf.errors.InvalidArgumentError as e:
            print(f"Error decoding DICOM file {file_path}: {e}")
            return None
        except Exception as e:  # General exception catch
            print(f"Unexpected error processing file {file_path}: {e}")
            return None

    # Count the number of images in the directory
    image_count = len(list(data_dir.glob("*/*.dcm")))

    # Create a TensorFlow dataset of file paths by listing all files in the specified directory and its subdirectories
    list_ds = tf.data.Dataset.list_files(str(data_dir / "*/*"), shuffle=False)

    # Shuffle the dataset while maintaining the order of elements across epochs
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    # Prepare the training dataset
    train_ds = list_ds.map(
        lambda x: tf.py_function(func=decode_dicom, inp=[x], Tout=tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Configure the dataset for performance
    train_ds = (
        train_ds.cache()
        .shuffle(1000)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Create the model (adjust layers as necessary)
    model = Sequential(
        [
            layers.InputLayer(input_shape=(image_height, image_width, 1)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(len(class_names)),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train the model
    try:
        model.fit(train_ds, epochs=epochs)
    except:
        print(
            "Error: Training failed. Please ensure there are no issues with the provided model file path."
        )

    # Save the model
    try:
        model.save(model_path)
    except:
        print(
            "Error: Model could not be saved. Please ensure there are no issues with the provided model file path."
        )


# Predict from DICOM image
def predict(
    dcm_path,
    model_path,
    class_names,
    window_center,
    window_width,
    image_height,
    image_width,
):

    # Check if model exists, load if it does
    if not os.path.isfile(model_path):
        print("Error: Model does not exist.")
        return
    else:
        model = tf.keras.models.load_model(model_path)

    # Check if DICOM image exists
    if not os.path.isfile(dcm_path):
        print("Error: DICOM image does not exist.")
        return

    # Read file and decode DICOM
    image_bytes = tf.io.read_file(dcm_path)
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)

    # Resize and normalize the image
    image = window_image(image, window_center, window_width)
    image = tf.image.resize(image, [image_height, image_width])
    image = tf.cast(image, tf.float32) / 65535.0  # Normalize to [0, 1]
    image = tf.expand_dims(image, 0)  # Create a batch dimension

    # Predict which class the image belongs to
    try:
        predictions = model.predict(image)
    except:
        print(
            "Error: Prediction failed. Please ensure there are no issues with the provided model."
        )
        return

    # Print results with score
    score = tf.nn.softmax(predictions[0])
    print(
        "Class: {} | Confidence: {:.2f}%".format(
            class_names[np.argmax(score)], 100 * np.max(score)
        )
    )


# Main function
def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train or predict DICOM images.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        default="train",
        help="Mode to run the script in: train or predict.",
    )
    parser.add_argument(
        "--dcm_dir",
        type=str,
        default="images_dcm/",
        help="Directory containing DICOM images for training.",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument(
        "--class_names",
        nargs="+",
        default=["Brain", "Lung"],
        help="Space-separated list of class names for prediction (e.g., --class_names Brain Lung).",
    )
    parser.add_argument(
        "--dcm_path",
        type=str,
        default="dcm_image.dcm",
        help="Path to a DICOM image for prediction.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="saved_model.keras",
        help="Path to save or load the model.",
    )
    parser.add_argument(
        "--window_center",
        type=int,
        default=50,
        help="Window center for DICOM image normalization.",
    )
    parser.add_argument(
        "--window_width",
        type=int,
        default=350,
        help="Window width for DICOM image normalization.",
    )
    parser.add_argument(
        "--image_height", type=int, default=512, help="Height of the images."
    )
    parser.add_argument(
        "--image_width", type=int, default=512, help="Width of the images."
    )

    # Parse arguments
    args = parser.parse_args()

    # Proceed with training or prediction
    if args.mode == "train":
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
    elif args.mode == "predict":
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
        print("Error: Mode must be 'train' or 'predict'.")


# Call main function when run directly from Python interpreter (change this if running from another program)
if __name__ == "__main__":
    main()
