# DeepCT
A simple DICOM (CT, MRI, etc.) image classification framework for Python that uses PyDicom and TensorFlow. Based on [this](https://www.tensorflow.org/tutorials/images/classification) TensorFlow tutorial.

# How to Use
Default variables are located beneath the imports at the top of DeepCT.py. These variables can be changed to fit your environment.

To train a model, store classes of .dcm images in the directory associated with the dcm_dir variable. (The default is 'images_dcm'.) The classes must be in different subdirectories - for example, 'images_dcm/brain' would be one and 'images_dcm/lung' would be another. Batch size, number of epochs, etc. can be changed from the default variables. The model will be saved in the location associated with the model_path variable. (The default is 'saved_model.keras'. Note: .keras files are recommended over the legacy .h5 files.) The program uses a supervised learning process with 80% of images used for training and 20% used for validation.

To predict the class of an image, change the "mode" variable from "train" to "predict." The path of the .dcm image to be classified should be stored in the dcm_path variable.

# Dependencies
Note: Versions more recent than those specified have not been tested with this software.

#### Python
The latest version tested is Python 3.11.2.

#### pydicom
```pip3 install pydicom```

Used for working with .dcm/.dicom images. Tested version is 2.4.2.

#### TensorFlow
```pip3 install tensorflow```

The latest tested version is 2.13.0. Many other dependencies, such as NumPy, should be installed along with TensorFlow. GPU support is also included if CUDA drivers are installed.

#### PIL
```pip3 install pillow```

Used for working with .png images. Tested version is 10.0.0.

#### Misc
```pip3 install pyyaml h5py```

Used for working with model storage files other than the default .keras.

#### OpenGL API
```apt install libgl1 libglib2.0-0```

On Debian 12 (Linux Kernel 6.1), these OpenGL packages were required.
