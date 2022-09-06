"""
Parameter Configurations

Author: Farid Qamar
Organizartion: Urban Observatory, University of Delaware
"""

# #####################
# main model parameters
# #####################
include_spatial   = True 
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

EPOCHS   = 
PATIENCE = 
