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
import tensorflow as tf

# -- import local libraries
import params as prm
import functions as fn


print("TensorFlow Version:", tf. __version__)
print("Python Version:", sys.version)

# -- read parameters
transfer = prm.transfer
if transfer == True:
       include_spatial = False
else:
       include_spatial = prm.include_spatial

if str(prm.scene) == "1-a":
       scan = "108"
elif str(prm.scene) == "1-b":
       scan = "000"
else:
       scan = "north"


# -- read and prep data
print("Preparing data cube...")
cube_std_3d, xy = fn.prep_data(prm.scene)

print("acquiring train/test sample...")
cube_train, cube_train_labels, xy_train, cube_test, cube_test_labels, xy_test = fn.get_train_test(scan, cube_std_3d, xy)

# -- create and compile CNN model
print("Instantiating the model...")
cnn = fn.CNN_Model(cube_std_3d.shape[2], include_spatial, prm.filtersize, prm.conv1, prm.dens1)

print("Compiling the model...")
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])

# -- fit model
if include_spatial:
       print("Fitting the model (with spatial included)...")
       CNNmodel = cnn.fit({"spectra":cube_train, "spatial":xy_train}, cube_train_labels, 
                         validation_data=({"spectra":cube_test, "spatial":xy_test}, cube_test_labels),
                         epochs=prm.EPOCHS, batch_size=prm.BATCH_SIZE)
else:
       print("Fitting the model (without spatial)...")
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
    
print("Training Accuracy = ", train_acc)
print("Testing Accuracy  = ", test_acc)


# -- predict pixel classification on each scene
if transfer == True:
       fn.evaluate_model("1-a", cnn, include_spatial)
       fn.evaluate_model("1-b", cnn, include_spatial)
       fn.evaluate_model("2", cnn, include_spatial)
else:
       fn.evaluate_model(scan, cnn, include_spatial)


