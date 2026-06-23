# Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Other imports
import matplotlib.pyplot as plt
import os

# Locations of datasets
cwd = os.getcwd()
PATH = cwd+'\dataset'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cancer_dir = os.path.join(train_dir, 'cancer_png')
train_healthy_dir = os.path.join(train_dir, 'healthy_png')
validation_cancer_dir = os.path.join(validation_dir, 'cancer_png')
validation_healthy_dir = os.path.join(validation_dir, 'healthy_png')

# Variables
num_epochs = 2
target_size = 150

num_cancer_tr = len(os.listdir(train_cancer_dir))
num_healthy_tr = len(os.listdir(train_healthy_dir))

num_cancer_val = len(os.listdir(validation_cancer_dir))
num_healthy_val = len(os.listdir(validation_healthy_dir))

total_train = num_cancer_tr + num_healthy_tr
total_val = num_cancer_val + num_healthy_val

# Initialising the CNN
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

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = (target_size, target_size),
                                                 batch_size = total_train // 1000,
                                                 class_mode = 'binary')

validation_set = test_datagen.flow_from_directory(validation_dir,
                                            target_size = (target_size, target_size),
                                            batch_size = total_val // 1000, # Change to 1000 in final build
                                            class_mode = 'binary')

history = classifier.fit_generator(training_set,
                         steps_per_epoch = total_train // 500, # Change to 500 in final build
                         epochs = num_epochs,
                         validation_data = validation_set,
                         validation_steps = total_val // 500) # Change to 500 in final build

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
plt.show()

# Save the trained model
classifier.save('test_model.h5')
print()
print("Trained model saved.")
