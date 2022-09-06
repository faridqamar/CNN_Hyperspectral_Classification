"""
Supporting functions

Author: Farid Qamar
Organizartion: Urban Observatory, University of Delaware

"""

# import python libraries
import sys
import pandas as pd
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt


# import local libraries
import params as prm


def point_from_string(text):
    """
	Takes in text representing the indices, splits 
	it into row index and column index and returns 
	them as integers
    """

    items = text.strip("\n").split(" ")
    rind = int(items[0])
    cind = int(items[1])
    
    return rind, cind


def coords(row, col):
	"""
	Transforms individual row and column numbers into array
	"""

    return np.array(list(np.ndindex((row, col)))).reshape(row, col, 2)


def read_labelled_coordinates(kind, scan):
	"""
	Read the manually classified pixels coordinated given
	the type of pixels (kind) and image identifier (scan)
	"""

    file = open("../data/{0}_coordinates_{1}.txt".format(kind, scan), "r")
    coords = file.readlines()
    file.close()
    coords = np.array([point_from_string(line) for line in coords])
    
    return coords


def split_train_test_indices(coords, seed, trrat, kind):
	"""
	Split the manually classified pixel coordinates into 
	training and testing sets fiven a split ratio (trrat)
	"""

    ind = np.arange(coords.shape[0])
    random.Random(seed).shuffle(ind)
    lim_ind = int(len(ind)*trrat)
    train_ind = ind[:lim_ind]
    test_ind = ind[lim_ind:]
    
    return train_ind, test_ind


def CNN_Model(nwaves, spatial, filtersize, conv1, dens1):  
	"""
	Create the CNN model using tensorflow.keras and the 
	selected hyperparamters
	"""      

    Inputs_1 = keras.Input(shape=(nwaves, 1), name="spectra")
    Conv1D_1 = layers.Conv1D(conv1, kernel_size=(filtersize), padding="same", activation="relu")(Inputs_1)
    MaxPool_1 = layers.MaxPooling1D((2), strides=2)(Conv1D_1)
    Conv1D_2 = layers.Conv1D(conv1*2, kernel_size=(filtersize), padding="same", activation="relu")(MaxPool_1)
    MaxPool_2 = layers.MaxPooling1D((2), strides=2)(Conv1D_2)
    Dropout_1 = layers.Dropout(0.5)(MaxPool_2)
    Flatten_1 = layers.Flatten()(Dropout_1)
    
    if spatial:
        Inputs_2 = keras.Input(shape=(2,), name="spatial")
        Concat = layers.concatenate([Flatten_1, Inputs_2])
        Dense_1 = layers.Dense(dens1, activation="relu")(Concat)
        Output = layers.Dense(9, activation="softmax")(Dense_1)

        model = keras.Model(inputs=[Inputs_1, Inputs_2],
                            outputs=[Output],)
    else:
        Dense_1 = layers.Dense(dens1, activation="relu")(Flatten_1)
        Output = layers.Dense(9, activation="softmax")(Dense_1)

        model = keras.Model(inputs=[Inputs_1],
                            outputs=[Output],)
    
    return model


def plot_loss_history(model):
	"""
	Plot the loss history and accuracy of the CNN
	"""

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12, 10))
    ax1.plot(model.history['loss'])
    ax1.plot(model.history['val_loss'])
    ax1.set_title('CNN Model Loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.locator_params(nbins=13, axis='x')
    ax1.legend(['train', 'test'], loc='center right')
    ax2.plot(model.history['acc'])
    ax2.plot(model.history['val_acc'])
    ax2.set_title('CNN Model Accuracy')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epoch')
    ax2.locator_params(nbins=12, axis='x')
    ax2.legend(['train', 'test'], loc='center right')
    ax3.plot(model.history['loss'])
    ax3.plot(model.history['val_loss'])
    ax3.set_ylabel('log(loss)')
    ax3.set_xlabel('epoch')
    ax3.locator_params(nbins=13, axis='x')
    ax3.legend(['train', 'test'], loc='center right')
    ax3.set_yscale('log')
    ax4.plot(model.history['acc'])
    ax4.plot(model.history['val_acc'])
    ax4.set_ylabel('log(accuracy)')
    ax4.set_xlabel('epoch')
    ax4.locator_params(nbins=12, axis='x')
    ax4.legend(['train', 'test'], loc='center right')
    ax4.set_yscale('log')
    f.savefig("../output/loss_history.png")

def get_train_test(scan, cube_std_3d, xy):
	"""
	Read hand labeled data for each class and create arrays for training
	and testing the model
	"""

	# -- Read hand labeled data of each class
	sky_coords = read_labelled_coordinates("1_sky", scan)
	clouds_coords = read_labelled_coordinates("2_clouds", scan)
	veg_coords = read_labelled_coordinates("3_vegetation", scan)
	wtr_coords = read_labelled_coordinates("4_water", scan)
	blt_coords = read_labelled_coordinates("5_buildings", scan)
	windows_coords = read_labelled_coordinates("6_windows", scan)
	rds_coords = read_labelled_coordinates("7_roads", scan)
	cars_coords = read_labelled_coordinates("8_cars", scan)
	mtl_coords = read_labelled_coordinates("9_metal", scan)

	# -- get the coordinates of training and testing sets for each class
	sky_train_ind, sky_test_ind = split_train_test_indices(sky_coords, 0, prm.trrat, "sky")
	clouds_train_ind, clouds_test_ind = split_train_test_indices(clouds_coords, 1, prm.trrat, "clouds")
	veg_train_ind, veg_test_ind = split_train_test_indices(veg_coords, 2, prm.trrat, "veg")
	wtr_train_ind, wtr_test_ind = split_train_test_indices(wtr_coords, 3, prm.trrat, "wtr")
	blt_train_ind, blt_test_ind = split_train_test_indices(blt_coords, 4, prm.trrat, "blt")
	windows_train_ind, windows_test_ind = split_train_test_indices(windows_coords, 5, prm.trrat, "windows")
	rds_train_ind, rds_test_ind = split_train_test_indices(rds_coords, 6, prm.trrat, "rds")
	cars_train_ind, cars_test_ind = split_train_test_indices(cars_coords, 7, prm.trrat, "cars")
	mtl_train_ind, mtl_test_ind = split_train_test_indices(mtl_coords, 8, prm.trrat, "mtl")

	# -- create the training and testing arrays for each class from the coordinates
	cube_sky_train = cube_std_3d[sky_coords[sky_train_ind[:], 0], sky_coords[sky_train_ind[:], 1], :]
	cube_sky_test = cube_std_3d[sky_coords[sky_test_ind[:], 0], sky_coords[sky_test_ind[:], 1], :]
	sky_xy_train = xy[sky_coords[sky_train_ind[:], 0], sky_coords[sky_train_ind[:], 1], :]
	sky_xy_test = xy[sky_coords[sky_test_ind[:], 0], sky_coords[sky_test_ind[:], 1], :]
	print("sky ", cube_sky_train.shape, cube_sky_test.shape)

	cube_clouds_train = cube_std_3d[clouds_coords[clouds_train_ind[:], 0], clouds_coords[clouds_train_ind[:], 1], :]
	cube_clouds_test = cube_std_3d[clouds_coords[clouds_test_ind[:], 0], clouds_coords[clouds_test_ind[:], 1], :]
	clouds_xy_train = xy[clouds_coords[clouds_train_ind[:], 0], clouds_coords[clouds_train_ind[:], 1], :]
	clouds_xy_test = xy[clouds_coords[clouds_test_ind[:], 0], clouds_coords[clouds_test_ind[:], 1], :]
	print("clouds ", cube_clouds_train.shape, cube_clouds_test.shape)

	cube_veg_train = cube_std_3d[veg_coords[veg_train_ind[:], 0], veg_coords[veg_train_ind[:], 1], :]
	cube_veg_test = cube_std_3d[veg_coords[veg_test_ind[:], 0], veg_coords[veg_test_ind[:], 1], :]
	veg_xy_train = xy[veg_coords[veg_train_ind[:], 0], veg_coords[veg_train_ind[:], 1], :]
	veg_xy_test = xy[veg_coords[veg_test_ind[:], 0], veg_coords[veg_test_ind[:], 1], :]
	print("vegetation ", cube_veg_train.shape, cube_veg_test.shape)

	cube_wtr_train = cube_std_3d[wtr_coords[wtr_train_ind[:], 0], wtr_coords[wtr_train_ind[:], 1], :]
	cube_wtr_test = cube_std_3d[wtr_coords[wtr_test_ind[:], 0], wtr_coords[wtr_test_ind[:], 1], :]
	wtr_xy_train = xy[wtr_coords[wtr_train_ind[:], 0], wtr_coords[wtr_train_ind[:], 1], :]
	wtr_xy_test = xy[wtr_coords[wtr_test_ind[:], 0], wtr_coords[wtr_test_ind[:], 1], :]
	print("water ", cube_wtr_train.shape, cube_wtr_test.shape)

	cube_blt_train = cube_std_3d[blt_coords[blt_train_ind[:], 0], blt_coords[blt_train_ind[:], 1], :]
	cube_blt_test = cube_std_3d[blt_coords[blt_test_ind[:], 0], blt_coords[blt_test_ind[:], 1], :]
	blt_xy_train = xy[blt_coords[blt_train_ind[:], 0], blt_coords[blt_train_ind[:], 1], :]
	blt_xy_test = xy[blt_coords[blt_test_ind[:], 0], blt_coords[blt_test_ind[:], 1], :]
	print("built ", cube_blt_train.shape, cube_blt_test.shape)

	cube_windows_train = cube_std_3d[windows_coords[windows_train_ind[:],0], windows_coords[windows_train_ind[:],1],:]
	cube_windows_test = cube_std_3d[windows_coords[windows_test_ind[:], 0], windows_coords[windows_test_ind[:], 1], :]
	windows_xy_train = xy[windows_coords[windows_train_ind[:], 0], windows_coords[windows_train_ind[:], 1], :]
	windows_xy_test = xy[windows_coords[windows_test_ind[:], 0], windows_coords[windows_test_ind[:], 1], :]
	print("windows ", cube_windows_train.shape, cube_windows_test.shape)

	cube_rds_train = cube_std_3d[rds_coords[rds_train_ind[:], 0], rds_coords[rds_train_ind[:], 1], :]
	cube_rds_test = cube_std_3d[rds_coords[rds_test_ind[:], 0], rds_coords[rds_test_ind[:], 1], :]
	rds_xy_train = xy[rds_coords[rds_train_ind[:], 0], rds_coords[rds_train_ind[:], 1], :]
	rds_xy_test = xy[rds_coords[rds_test_ind[:], 0], rds_coords[rds_test_ind[:], 1], :]
	print("roads ", cube_rds_train.shape, cube_rds_test.shape)

	cube_cars_train = cube_std_3d[cars_coords[cars_train_ind[:], 0], cars_coords[cars_train_ind[:], 1], :]
	cube_cars_test = cube_std_3d[cars_coords[cars_test_ind[:], 0], cars_coords[cars_test_ind[:], 1], :]
	cars_xy_train = xy[cars_coords[cars_train_ind[:], 0], cars_coords[cars_train_ind[:], 1], :]
	cars_xy_test = xy[cars_coords[cars_test_ind[:], 0], cars_coords[cars_test_ind[:], 1], :]
	print("cars ", cube_cars_train.shape, cube_cars_test.shape)

	cube_mtl_train = cube_std_3d[mtl_coords[mtl_train_ind[:], 0], mtl_coords[mtl_train_ind[:], 1], :]
	cube_mtl_test = cube_std_3d[mtl_coords[mtl_test_ind[:], 0], mtl_coords[mtl_test_ind[:], 1], :]
	mtl_xy_train = xy[mtl_coords[mtl_train_ind[:], 0], mtl_coords[mtl_train_ind[:], 1], :]
	mtl_xy_test = xy[mtl_coords[mtl_test_ind[:], 0], mtl_coords[mtl_test_ind[:], 1], :]
	print("metal ", cube_mtl_train.shape, cube_mtl_test.shape)

	# -- concatenate the training arrays into training data and labels
	cube_train = np.concatenate((cube_sky_train, cube_clouds_train, cube_veg_train, cube_wtr_train, cube_blt_train,
	                            cube_windows_train, cube_rds_train, cube_cars_train, cube_mtl_train), axis=0)
	cube_train2 = cube_train.reshape(cube_train.shape[0], cube_train.shape[1], 1)
	cube_train_labels = [0]*cube_sky_train.shape[0] + [1]*cube_clouds_train.shape[0] \
	                    + [2]*cube_veg_train.shape[0] + [3]*cube_wtr_train.shape[0] \
	                    + [4]*cube_blt_train.shape[0] + [5]*cube_windows_train.shape[0] \
	                    + [6]*cube_rds_train.shape[0] + [7]*cube_cars_train.shape[0] + [8]*cube_mtl_train.shape[0]
	cube_train_labels = np.array(cube_train_labels)
	xy_train = np.concatenate((sky_xy_train, clouds_xy_train, veg_xy_train, wtr_xy_train, blt_xy_train,
	                             windows_xy_train, rds_xy_train, cars_xy_train, mtl_xy_train), axis=0)

	# -- concatenate the testing arrays into testing data and labels
	cube_test = np.concatenate((cube_sky_test, cube_clouds_test, cube_veg_test, cube_wtr_test, cube_blt_test,
	                            cube_windows_test, cube_rds_test, cube_cars_test, cube_mtl_test), axis=0)
	cube_test2 = cube_test.reshape(cube_test.shape[0], cube_test.shape[1], 1)
	cube_test_labels = [0]*cube_sky_test.shape[0] + [1]*cube_clouds_test.shape[0] \
	                    + [2]*cube_veg_test.shape[0] + [3]*cube_wtr_test.shape[0] \
	                    + [4]*cube_blt_test.shape[0] + [5]*cube_windows_test.shape[0] \
	                    + [6]*cube_rds_test.shape[0] + [7]*cube_cars_test.shape[0] + [8]*cube_mtl_test.shape[0]
	cube_test_labels = np.array(cube_test_labels)
	xy_test = np.concatenate((sky_xy_test, clouds_xy_test, veg_xy_test, wtr_xy_test, blt_xy_test,
	                             windows_xy_test, rds_xy_test, cars_xy_test, mtl_xy_test), axis=0)

	return cube_train2, cube_train_labels, xy_train, cube_test2, cube_test_labels, xy_test
