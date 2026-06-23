# Tkinter import
from tkinter import *
from tkinter import filedialog # Needed for Pyinstaller to work

# TensorFlow import
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pydicom
import os
import PIL

# Window
main = Tk()
main.title('DeepCT')

# Variables for status of image and model
is_image = False
is_model = False
global model_from_upload # Whether the model was uploaded or created
model_from_upload = True

# Check status of whether there are images
global image_status_text
image_status_text = StringVar(main)

def check_image_status():

    if (is_image == False):
        image_status_text.set("Uploaded Image: False")
    else:
        image_status_text.set("Uploaded Image: True")

# Check status of whether there is an uploaded model
global model_status_text
model_status_text = StringVar(main)

def check_model_status():

    if (is_model == False):
        model_status_text.set("Uploaded Model: False")
    else:
        model_status_text.set("Uploaded Model: True")

# Upload DICOM image
def upload_image():

    # Upload the DICOM
    global import_image_filename
    import_image_filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("DICOM Image","*.dcm"),("all files","*.*")))

    # Set image uploaded to true
    global is_image
    is_image = True

    # Convert the dicom into .png
    try:
        ImageFile = pydicom.dcmread(import_image_filename)
        plt.imsave(str(import_image_filename + ".png"), ImageFile.pixel_array, cmap=plt.cm.gray, vmin=1, vmax=2500)
    except FileNotFoundError:
        is_image = False # In case no file is uploaded

    # Update image status
    check_image_status()

# Upload trained model
def upload_model():

    # Upload the model
    global import_model_filename
    import_model_filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("ML Model (.h5)","*.h5"),("all files","*.*")))

    # Set model to true
    global is_model
    is_model = True

    check_model_status()

# Load and prepare the uploaded image
def load_image(filename):
    # Load the image
    img = load_img(filename, target_size=(150, 150))
    # Convert to array
    img = img_to_array(img)
    # Reshape into a single sample with 3 channels
    img = img.reshape(1, 150, 150, 3)
    # Center pixel data
    img = img.astype("float32")
    return img

# Load an image and predict its class
def run_program():
    # Load the image
    img = load_image(import_image_filename + ".png")
    
    global model_filename

    # Load the model
    if(model_from_upload == False):
        model_filename = "generated_model.h5"
    else:
        model_filename = import_model_filename

    model = load_model(model_filename)

    # Predict the class
    result = model.predict(img)
    result_val = result[0]

    # Initialize result text variable
    global result_text
    # Print result based on numeric value
    if (result_val == 0):
        result_text = "Cancer"
    else:
        result_text = "Healthy"

# Display the value in Entry box
def display_result():
    # Run the program while catching 1. No Image/Model uploaded 2. model uploaded not a .h5 file
    try:
        run_program()
        blank.delete(0, 'end')
        blank.insert(0, result_text)
    except NameError:
        messagebox.showerror("Error", "No Image/Model uploaded. Please try again.")
    except OSError:
        messagebox.showerror("Error", "Model Not .h5. Please import another model or create one.")

def open_healthy_train_dir():
    global healthy_train_dir_path
    healthy_train_dir_path = filedialog.askdirectory()
    print(healthy_train_dir_path)

def open_cancer_validate_dir():
    global cancer_validate_dir_path
    cancer_validate_dir_path = filedialog.askdirectory()
    print(cancer_validate_dir_path)

def open_healthy_validate_dir():
    global healthy_validate_dir_path
    healthy_validate_dir_path = filedialog.askdirectory()
    print(healthy_validate_dir_path)

# Create a new window for training
def create_train_window():
    train_window = Toplevel(main)
    train_window.title("Train a Model")

    # Variables for existence of directories
    cancer_train_dir_exist = False
    healthy_train_dir_exist = False
    cancer_validate_dir_exist = False
    healthy_validate_dir_exist = False

    # Cancer Training Directory
    global cancer_train_dir_text
    cancer_train_dir_text = StringVar(train_window)

    def check_cancer_train_dir_status():

        if(cancer_train_dir_exist == False):
            cancer_train_dir_text.set("Cancer Training Directory: False")
        else:
            cancer_train_dir_text.set("Cancer Training Directory: True")


    def open_cancer_train_dir():
        global cancer_train_dir_path
        cancer_train_dir_path = filedialog.askdirectory()

        nonlocal cancer_train_dir_exist
        cancer_train_dir_exist = True

        check_cancer_train_dir_status()

    check_cancer_train_dir_status()

    Label(train_window, textvariable = cancer_train_dir_text).grid(row=0)
    Button(train_window, text="Select", command=open_cancer_train_dir).grid(row=0, column=1)

    # Healthy Training Directory
    global healthy_train_dir_text
    healthy_train_dir_text = StringVar(train_window)

    def check_healthy_train_dir_status():

        if(healthy_train_dir_exist == False):
            healthy_train_dir_text.set("Healthy Training Directory: False")
        else:
            healthy_train_dir_text.set("Healthy Training Directory: True")

    def open_healthy_train_dir():
        global healthy_train_dir_path
        healthy_train_dir_path = filedialog.askdirectory()

        nonlocal healthy_train_dir_exist
        healthy_train_dir_exist = True

        check_healthy_train_dir_status()

    check_healthy_train_dir_status()

    Label(train_window, textvariable = healthy_train_dir_text).grid(row=1)
    Button(train_window, text="Select", command=open_healthy_train_dir).grid(row=1, column=1)

    # Cancer Validation Directory
    global cancer_validate_dir_text
    cancer_validate_dir_text = StringVar(train_window)

    def check_cancer_validate_dir_status():

        if(cancer_validate_dir_exist == False):
            cancer_validate_dir_text.set("Cancer Validation Directory: False")
        else:
            cancer_validate_dir_text.set("Cancer Validation Directory: True")

    def open_cancer_validate_dir():
        global cancer_validate_dir_path
        cancer_validate_dir_path = filedialog.askdirectory()

        nonlocal cancer_validate_dir_exist
        cancer_validate_dir_exist = True

        check_cancer_validate_dir_status()

    check_cancer_validate_dir_status()

    Label(train_window, textvariable = cancer_validate_dir_text).grid(row=2)
    Button(train_window, text="Select", command=open_cancer_validate_dir).grid(row=2, column=1)

    # Healthy Validation Directory
    global healthy_validate_dir_text
    healthy_validate_dir_text = StringVar(train_window)

    def check_healthy_validate_dir_status():

        if(healthy_validate_dir_exist == False):
            healthy_validate_dir_text.set("Healthy Validation Directory: False")
        else:
            healthy_validate_dir_text.set("Healthy Validation Directory: True")

    def open_healthy_validate_dir():
        global healthy_validate_dir_path
        healthy_validate_dir_path = filedialog.askdirectory()

        nonlocal healthy_validate_dir_exist
        healthy_validate_dir_exist = True

        check_healthy_validate_dir_status()

    check_healthy_validate_dir_status()

    Label(train_window, textvariable = healthy_validate_dir_text).grid(row=3)
    Button(train_window, text="Select", command=open_healthy_validate_dir).grid(row=3, column=1)

    # ML model variables
    global num_epochs
    target_size = 150

    Label(train_window, text="Number of Epochs:").grid(row=4)
    num_epochs_entry = Entry(train_window)
    num_epochs_entry.grid(row=4, column=1) # So that .get() works and does not select .grid()

    def create_model():

        # Catches number of epochs being zero or a decimal
        try:
            num_epochs = int(num_epochs_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Number of epochs cannot be zero and must be a whole number.")

        # Catches number of epochs being less than one
        try: 
            if (num_epochs < 1):
                messagebox.showerror("Error", "Number of epochs cannot be less than one.")
        except UnboundLocalError:
            pass # So that there is no error when num_epochs has no value

        # More variables

        try:
            num_cancer_tr = len(os.listdir(cancer_train_dir_path))
            num_healthy_tr = len(os.listdir(healthy_train_dir_path))

            num_cancer_val = len(os.listdir(cancer_validate_dir_path))
            num_healthy_val = len(os.listdir(healthy_validate_dir_path))

            total_train = num_cancer_tr + num_healthy_tr
            total_val = num_cancer_val + num_healthy_val
            print(total_val)
        except NameError:
            messagebox.showerror("Error", "At least one of the selected directories is invalid.") # If no directory is selected, or other error

        # Initializing the CNN
        classifier = Sequential()

        # Convolution
        classifier.add(Convolution2D(32, 3, 3, input_shape = (target_size, target_size, 3), activation = 'relu'))

        # Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second convolutional layer
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Flattening
        classifier.add(Flatten())

        # Full connection
        classifier.add(Dense(128, activation = 'relu'))
        classifier.add(Dense(1, activation = 'sigmoid'))

        # Compile the CNN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Fit the CNN to the images
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1./255)

        try:
            training_set = train_datagen.flow_from_directory(os.path.abspath(os.path.join(cancer_train_dir_path, os.pardir)), # Get training parent directory
                                                             target_size = (target_size, target_size),
                                                             batch_size = total_train // 1000,
                                                             class_mode = 'binary')

            validation_set = test_datagen.flow_from_directory(os.path.abspath(os.path.join(cancer_validate_dir_path, os.pardir)), # Get validation parent directory
                                                    target_size = (target_size, target_size),
                                                    batch_size = total_val // 1000,
                                                    class_mode = 'binary')
        
            history = classifier.fit_generator(training_set,
                                        steps_per_epoch = total_train // 500,
                                        epochs = num_epochs,
                                        validation_data = validation_set,
                                        validation_steps = total_val // 500)

            # Visualize training results
            acc = history.history['acc']
            val_acc = history.history['val_acc']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(num_epochs)

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show(block=False)

            # Save the trained model
            classifier.save('generated_model.h5')
            print()
            print("Trained model saved.")
            # Set model to true
            global is_model
            is_model = True
            # Check status of model exist
            check_model_status()
            # Declare that the program should use the generated model
            global model_from_upload
            model_from_upload = False

        except ZeroDivisionError: # In final build, remove FileNotFoundError to except all
            pass # In case directory is not found

    Button(train_window, text="Create Model", command=create_model).grid(row=5)

    # Directions for directory setup
    Label(train_window, text="Individual directories should be one level below total train/validate directories.").grid(row=0, column=2)

# Menu
menubar = Menu(main)

# create a pulldown menu, and add it to the menu bar
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Select Image", command=upload_image) # Uploads the image
menubar.add_cascade(label="File", menu=filemenu)

# Training menu
train_menu = Menu(menubar, tearoff=0)
train_menu.add_command(label="Import Trained Model", command=upload_model)
train_menu.add_command(label="Train a Model", command=create_train_window)
menubar.add_cascade(label="Train Model", menu=train_menu)

# Display the menu
main.config(menu=menubar)

# Display image and model
check_image_status() # Update status of image upload
check_model_status() # Update status of model upload

image_label = Label(main, textvariable = image_status_text)
model_label = Label(main, textvariable = model_status_text)

# Display result
blank = Entry(main)

# Button to run the program
run_button = Button(main, text='Classify', command=display_result)

# Positioning of elements in main window
image_label.grid(row=0)
model_label.grid(row=1)
run_button.grid(row=2)
blank.grid(row=2, column=1)

main.mainloop()

# Delete the created .png image
os.remove(import_image_filename + ".png")