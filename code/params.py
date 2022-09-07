"""
Parameter Configurations

Author: Farid Qamar
Organizartion: Urban Observatory, University of Delaware

"""

# #####################
# logistics
# #####################

# -- training image
scene = "1-a"  # options are "1-a", "1-b", and "2"

# -- transfer model (test on) to other scenes
transfer = False



# #####################
# main model parameters
# #####################
include_spatial   = True  # will be forced to False if transfer==True
reduce_resolution = False
num_of_bins = 848  # change to desired resolution if reduce_resolution==True

# training sample split ratio
trrat = 0.8



# #####################
# Hyperparameters
# #####################

# CNN
filtersize = 50
conv1      = 16
dens1      = 512

EPOCHS     = 110 
BATCH_SIZE = 512
