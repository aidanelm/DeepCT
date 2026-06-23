import pydicom
import matplotlib.pyplot as plt
import os

directory_in_str = "dataset/validation/cancer"

directory = os.fsencode(directory_in_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename_plus_dir = directory_in_str + "/" + filename
    print(filename_plus_dir)
    ImageFile = pydicom.read_file(filename_plus_dir)
    plt.imsave(str("dataset/validation/cancer_png/" + filename + ".png"), ImageFile.pixel_array, cmap=plt.cm.gray, vmin=1, vmax=250)