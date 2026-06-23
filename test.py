# Tkinter import
from tkinter import *
from tkinter import filedialog # Needed for Pyinstaller to work

# Helper libraries
import pydicom
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os

# Window
main = Tk()
main.title('DeepCT')

# Change the directory
def select_directory():
    path = filedialog.askdirectory()
    os.chdir(path)

# Display the model
def show_scan():

    # load the DICOM files
    files = []

    for fname in glob.glob("*.dcm", recursive=False):
        files.append(pydicom.dcmread(fname))

    # Skip files without a SliceLocation (Scout Views)
    slices = []
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            continue

    # Order files correctly
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # pixel aspects, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]

    # create 3D array
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d

    # Plot axial slice
    axial = plt.subplot(2, 2, 1)
    axial.set_title("Axial")
    plt.imshow(img3d[:, :, img_shape[2]//2])
    axial.set_aspect(ax_aspect)
    plt.set_cmap("bone")
    
    # Plot sagittal slice
    sagittal = plt.subplot(2, 2, 2)
    sagittal.set_title("Sagittal")

    # Rotate the image correctly (by 90 degrees)
    sag_original = img3d[:, img_shape[1]//2, :]
    sag_rotated = ndimage.rotate(sag_original, 90)
    plt.imshow(sag_rotated)
    sagittal.set_aspect(sag_aspect)
    plt.set_cmap("bone")

    # Plot coronal slice
    coronal = plt.subplot(2, 2, 3)
    coronal.set_title("Coronal")
    plt.imshow(img3d[img_shape[0]//2, :, :].T, origin="lower") # Aligns the scan correctly
    coronal.set_aspect(cor_aspect)
    plt.set_cmap("bone")

    # Ensure no overlap
    plt.tight_layout()

    plt.show()

# Menu
menubar = Menu(main)

# create a pulldown menu, and add it to the menu bar
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Select Directory", command=select_directory) # Selects the DICOM directory
menubar.add_cascade(label="File", menu=filemenu)

# Model menu
model_menu = Menu(menubar, tearoff=0)
model_menu.add_command(label="Display the Scan", command=show_scan)
menubar.add_cascade(label="Train Model", menu=model_menu)

# Display the menu
main.config(menu=menubar)

main.mainloop()