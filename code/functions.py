"""
Supporting functions

Author: Farid Qamar
Organizartion: Urban Observatory, University of Delaware

"""

# import python libraries
import sys
import pandas as pd
import numpy as np

# import local libraries
import params


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

    file = open("../manual_classified_pixels/{0}_coordinates_{1}.txt".format(kind, scan), "r")
    coords = file.readlines()
    file.close()
    coords = np.array([point_from_string(line) for line in coords])
    print("{0}: {1}".format(kind,coords.shape))
    
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
    
    print("{0}: {1} {2}".format(kind, len(train_ind), len(test_ind)))
    
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

