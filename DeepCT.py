# Import all necessary modules
import os
import cv2 # OpenCV
import pydicom # DICOM
import numpy as np
import shutil # File operations
import pathlib # File operations
import PIL # Images
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Change these variables as is necessary or convenient
# Training
dcm_dir = 'images_dcm/' # Input directory containing DICOM images
png_dir = 'images_png/' # Output directory to save converted PNG images
epochs = 2
batch_size = 1
# Prediction
class_names = ['Brain', 'Lung'] # Alphabetized list of possible classes (for prediction only - should correspond with training classes)
dcm_path = 'dcm_image.dcm' # image path for prediction
# Both/Misc
mode = 'train' # train or predict
model_path = 'saved_model.keras'
window_center = 50 # Window center for image normalization
window_width = 350 # Window width for image normalization
image_height = 512
image_width = 512

# Helper function to normalize DICOM images
def window_image(png_image, window_center, window_width, intercept, slope):
    # Rescale the image
    png_image = (png_image * slope + intercept)

    # Define minimum and maximum values based on window center and width
    png_image_min = window_center - window_width
    png_image_max = window_center + window_width

    # Clip values outside the window range
    png_image[png_image < png_image_min] = png_image_min
    png_image[png_image > png_image_max] = png_image_max

    return png_image

# Convert single DICOM to PNG
def convert_image(dcm_path):

    # Read the DICOM file
    try:
        dicom_ds = pydicom.read_file(dcm_path)
    except FileNotFoundError:
        print('Error: DICOM file not found. Please ensure the path contained in dcm_path is correct.')
        exit()

    # Get the image data
    png_image = dicom_ds.pixel_array

    # Get the intercept and slope for image rescaling, use default values if they are not provided
    intercept = dicom_ds.RescaleIntercept if 'RescaleIntercept' in dicom_ds else -1024
    slope = dicom_ds.RescaleSlope if 'RescaleSlope' in dicom_ds else 1

    # Normalize the image
    png_image = window_image(png_image, window_center, window_width, intercept, slope)

    # Convert image data type to 8-bit unsigned integer
    png_image = png_image.astype(np.uint8)
    return png_image

# Convert multiple DICOM to PNG
def convert_batch_images(subdirectory):

    # Get the list of all files in the DICOM directory
    dcm_dir_files = [dcm_file for dcm_file in os.listdir(dcm_dir + subdirectory)]

    # Process each file
    for dcm_file in dcm_dir_files:

        # Process only DICOM files
        if dcm_file.lower().endswith('.dcm'):

            # Get PNG image
            png_image = convert_image(dcm_dir + subdirectory + dcm_file)

            # Save the image in PNG format, replace original file extension '.dcm' with '.png'
            cv2.imwrite(png_dir + subdirectory + dcm_file.lower().replace('.dcm', '.png'), png_image)

# Train a model
def train():

    # Create the PNG directory if it does not exist
    try:
        os.mkdir(png_dir)

    # If it exists, remove all files and directories inside
    except FileExistsError:
        shutil.rmtree(png_dir) # Remove the directory and all its contents
        os.mkdir(png_dir) # Recreate the directory

    # For each top-level directory (but not further subdirectory) in dcm_dir, convert all DICOM images
    try:
        for item in os.scandir(dcm_dir):
            if item.is_dir():
                subdirectory = str(item.path).split('/', 1)[-1] + '/' # Remove dcm_dir from path and add slash
                os.mkdir(png_dir + subdirectory)
                convert_batch_images(subdirectory) # Convert all DICOM in subdirectory to PNG
    except FileNotFoundError:
        print("Error: Please ensure that the directory in dcm_dir exists.")
        shutil.rmtree(png_dir) # Delete created directory
        exit()

    # TensorFlow standard is data_dir for input images
    data_dir = pathlib.Path(png_dir)

    # Create training and validation datasets (80/20 split)
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split = 0.2, subset = 'training', seed = 123, image_size = (image_height, image_width), batch_size = batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split = 0.2, subset = 'validation', seed = 123, image_size = (image_height, image_width), batch_size = batch_size)
    class_names = train_ds.class_names # Override global variable

    # Configure the dataset for performance
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    # Standardize the data
    normalization_layer = layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    # Create the model
    model = Sequential([
        layers.Rescaling(1./255, input_shape = (image_height, image_width, 3)),
        layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(len(class_names))
    ])

    # Compile and train the model
    model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
    history = model.fit(train_ds, validation_data = val_ds, epochs = epochs)

    # Save the model
    model.save(model_path)

    # Delete all PNG files
    shutil.rmtree(png_dir)

# Predict from DICOM image
def predict():

    # Convert DCM to PNG
    png_image = convert_image(dcm_path)

    # Save the image in PNG format
    png_path = dcm_path.lower().replace('.dcm', '.png')
    cv2.imwrite(png_path, png_image)

    # Load a saved model
    try:
        model = tf.keras.models.load_model(model_path)
    except IOError:
        print('I/O error. Please ensure that the file in model_path exists.')
        os.remove(png_path) # Delete created PNG
        exit()

    # Load the PNG image into TensorFlow
    img = tf.keras.utils.load_img(png_path, target_size = (image_height, image_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Predict which class the image belongs to
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    try:
        print('Class: {} | Confidence: {:.2f}%'.format(class_names[np.argmax(score)], 100 * np.max(score)))
    except IndexError:
        print('Error: Please ensure that the values in class_names correspond with the classes from the trained model.')
        os.remove(png_path) # Delete created PNG
        exit()

    # Delete created PNG
    os.remove(png_path)

# Main function
def main():
    if mode == "train":
        train()
    elif mode == "predict":
        predict()
    else:
        print("Error: Mode must be 'train' or 'predict'.")
        exit() # Redundant (for now)

# Call main function when run directly from Python interpreter (change this if running from another program)
if __name__ == '__main__':
	main()
