# CNN_Hyperspectral_Classification
Pixel-Wise Classification of Urban Hyperspectral Images with Convolutional Neural Networks


This repo contains the code used in the publication ["Pixel-Wise Classification of High-Resolution Ground-Based Urban Hyperspectral Images with Convolutional Neural Networks"](https://doi.org/10.3390/rs12162540), published in MDPI Remote Sensing in 2020.

For this work we used ground-based, remote hyperspectral images of an urban scene in New York City captured by the Urban Observatory in VNIR (0.4 - 1.0 microns) in ~850 spectral channels. We use a 1D-CNN model to classify pixels from their spectra as the following classes: Sky, Clouds, Vegetation, Water, Building facades, Windows, Roads, Cars, and Metal Structures.

The model architecture is summarized as follows:

![CNN Model Architecture](https://github.com/faridqamar/UO_Hyperspectral_Classification/blob/main/output/cnn_model.png)

