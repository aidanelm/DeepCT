# DeepCT
A simple DICOM (CT, MRI, etc.) image classification framework for TensorFlow in Python. Based on [this](https://www.tensorflow.org/tutorials/images/classification) TensorFlow tutorial.

# Dependencies
Note: Versions more recent than those specified have not been tested with this software.

#### Python
The latest version tested is Python 3.11.2.

#### TensorFlow
```pip3 install tensorflow==2.13.0```

The latest tested version is 2.13.0. Many other dependencies, such as NumPy, should be installed along with TensorFlow. GPU support is also included if CUDA drivers are installed.

#### TensorFlow I/O
```pip3 install tensorflow-io==0.36.0```

Used for working with DICOM images. The latest tested version is 0.36.0.

#### OpenGL API
```apt install libgl1 libglib2.0-0```

On Debian 12 (Linux Kernel 6.1), these OpenGL packages were required in the past.

# How to Use

### Training the Model

1. **Prepare Your Data**: Organize your DICOM images in a directory structure where each subdirectory name represents a class name, and contains the DICOM images belonging to that class. For example:

```
images_dcm/
    Brain/
        image1.dcm
        image2.dcm
    Lung/
        image3.dcm
        image4.dcm
```

2. **Run the Training**: Use the command line to navigate to the directory containing the script and run it with the necessary arguments for training. Here's an example:

```bash
python DeepCT.py --mode train --dcm_dir path_to_images_dcm/ --epochs 10 --batch_size 32 --class_names Brain Lung --model_path path_to_saved_model.keras --window_center 50 --window_width 350 --image_height 512 --image_width 512
```

Replace `path_to_images_dcm/` with the path to your DICOM images directory and `path_to_saved_model.keras` with where you want to save your trained model.

### Predicting with the Model

1. **Ensure Your Model is Trained**: Make sure you have a trained model saved at a known location.

2. **Run the Prediction**: Use the command line to navigate to the directory containing the script and run it with the necessary arguments for prediction. Here's an example:

```bash
python DeepCT.py --mode predict --dcm_path path_to_image.dcm --model_path path_to_saved_model.keras --class_names Brain Lung --window_center 50 --window_width 350 --image_height 512 --image_width 512
```

Replace `path_to_image.dcm` with the DICOM image you want to predict and `path_to_saved_model.keras` with the path to your trained model.

### Notes:

- The `--window_center` and `--window_width` arguments are used for normalizing the DICOM images. Adjust these based on the type of images you're working with for optimal results.
- The `--image_height` and `--image_width` should match the input shape that the model expects. Adjust these based on your specific needs.
- This script assumes a binary or multi-class classification problem. You should adjust the `class_names` argument based on the classes represented in your dataset.

This framework can be adapted or extended depending on specific project needs, such as modifying the CNN architecture, changing the optimization parameters, or adding additional preprocessing steps for the DICOM images.
