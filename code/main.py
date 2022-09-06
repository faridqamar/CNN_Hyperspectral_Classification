#!/usr/bin/python3
"""
Main driver code for the paper titled "Pixel-Wise Classification of 
High-Resolution Ground-Based Urban Hyperspectral Images with 
Convolutional Neural Networks"

Author: Farid Qamar
Organizartion: Urban Observatory, University of Delaware

"""

# -- import python libraries
import sys
import pandas as pd
import numpy as np

# -- import local libraries
import params as prm
import functions as fn
import hyss_util as hu

print("TensorFlow Version:", tf. __version__)
print("Python Version:", version)

# -- read parameters
transfer = prm.transfer
if transfer == True:
       include_spatial = False
else:
       include_spatial = prm.nclude_spatial


if str(prm.scene) == "1-a":
       scan = "108"
       fname = "../../image_files/veg_00"+scan+".raw"
elif str(prm.scene) == "1-b":
       scan = "000"
       fname = "../../image_files/veg_00"+scan+".raw"
else:
       fname = "../../scan1_slow_roof_VNIR.hdr"


# -- read the HSI cube from .raw file into float array
fname = "../../image_files/veg_00"+scan+".raw"
cube  = hu.read_hyper(fname)

# -- reshape cube from (wavelength, row, col) to shape (row*col, wavelength)
cube_reshaped = cube.data.transpose(1, 2, 0).reshape((cube_sub0.shape[1] * cube_sub0.shape[2]), cube_sub0.shape[0])

# -- standardize the cube to have mean=0 and standard deviation=1
cube_standard = (cube_reshaped - cube_reshaped.mean(1, keepdims=True)) / cube_reshaped.std(1, keepdims=True)

# -- if reduce_resolution = True, the spectra are averaged into bins to simulate reduced resolution
if prm.reduce_resolution:
    bin_ind = []

    for i in range(0, num_of_bins):
        low_ind = int(i*int(cube_sub.shape[0]/num_of_bins))
        upp_ind = int(low_ind + int(cube_sub.shape[0]/num_of_bins))
        bin_ind.append([low_ind, upp_ind])
    bin_ind[-1][-1] = cube_sub.shape[0]

    cube_binned = np.zeros(shape=(cube_standard.shape[0], num_of_bins))
    for i in range(num_of_bins):
        cube_binned[:, i] = cube_standard[:, bin_ind[i][0]:bin_ind[i][1]].mean(1)

    cube_standard = cube_binned
    
# -- reshape standardized cube to (row, col, wavelength)
cube_std_3d = cube_standard.reshape(cube_sub.shape[1], cube_sub.shape[2], cube_sub.shape[0])

# -- create position array from coordinates and normalize
xy = fn.coords(cube_sub.shape[1], cube_sub.shape[2])
xy = xy/xy.max()


cube_train, cube_train_labels, xy_train, cube_test, cube_test_labels, xy_test = get_train_test("108", cube_std_3d, xy)

# -- create and compile CNN model
cnn = fn.CNN_Model(cube_std_3d.shape[2], spatial=include_spatial, prm.filtersize, prm.conv1, prm.dens1)
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])

# -- fit model
if include_spatial:
    CNNmodel = cnn.fit({"spectra":cube_train, "spatial":xy_train}, cube_train_labels, 
                         validation_data=({"spectra":cube_test, "spatial":xy_test}, cube_test_labels),
                         epochs=prm.EPOCHS, batch_size=prm.BATCH_SIZE)
else:
    CNNmodel = cnn.fit({"spectra":cube_train}, cube_train_labels, 
                         validation_data=({"spectra":cube_test}, cube_test_labels),
                         epochs=prm.EPOCHS, batch_size=prm.BATCH_SIZE)


# -- evaluate model
# model accuracy on training and testing sets

if include_spatial:
    train_loss, train_acc = cnn.evaluate({"spectra":cube_train, "spatial":xy_train}, cube_train_labels)
    test_loss, test_acc = cnn.evaluate({"spectra":cube_test, "spatial":xy_test}, cube_test_labels)
else:
    train_loss, train_acc = cnn.evaluate({"spectra":cube_train}, cube_train_labels)
    test_loss, test_acc = cnn.evaluate({"spectra":cube_test}, cube_test_labels)
    
print(train_acc, test_acc)


# -- predict pixel classification on entire images

# -- predict on scene 1-a
cube_standard_1 = cube_standard.reshape(cube_standard.shape[0], cube_standard.shape[1], 1)
xy_2d = xy.reshape((xy.shape[0] * xy.shape[1]), xy.shape[2])
start_time = time.time()
if include_spatial:
    probCube = cnn.predict({"spectra":cube_standard_1, "spatial":xy_2d})
else:
    probCube = cnn.predict({"spectra":cube_standard_1})
predictCube = probCube.argmax(axis=-1)
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# -- evaluation metrics for Scene 1-a



