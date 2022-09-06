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

# -- read the HSI cube from .raw file into float array
fname = "../../image_files/veg_00108.raw"
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








