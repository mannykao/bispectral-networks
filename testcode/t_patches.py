#!/usr/bin/env python
# coding: utf-8

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import sys
sys.path.append("../")
import os
import torch
import numpy as np
from bispectral_networks.logger import load_checkpoint

# Dataset Imports
from bispectral_networks.data.datasets import MNISTExemplars, TransformDataset
from bispectral_networks.data.transforms import SO2, CircleCrop, CenterMean, UnitStd, Ravel
from skimage.transform import resize

# Plotting imports
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
#get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=1.5)
sns.set_style('dark')

kPlotWeights=False

# # Bispectral Neural Networks - Rotation Experiment
# 
# This notebook reproduces the plots for the rotation experiment. It also allows the user to test the network on datasets generated with different random seeds, to examine the generality of the results. We examine the properties of the network with respect to three criteria:
# - Invariance and Equivariance
# - Generalization
# - Robustness

def plotWeights(weights, save_dir):
	from bispectral_networks.analysis.plotting import image_grid

	# **Real Components**
	image_grid(weights.real, cmap="Greys_r", shape=(16, 16), figsize=(15, 15), share_range=False, save_name=os.path.join(save_dir, "W_real.pdf"))

	# **Imaginary Components**
	image_grid(weights.imag, cmap="Greys_r", shape=(16, 16), figsize=(15, 15), share_range=False, save_name=os.path.join(save_dir, "W_imag.pdf"))



if __name__ == '__main__':
	save_dir = "figs/rotation/"
	os.makedirs(save_dir, exist_ok=True) 

	# ### Load Checkpoint
	log_path = "../logs/rotation_model/"
	checkpoint, config, weights = load_checkpoint(log_path)

	#checkpoint
	print(weights.dtype)
	print(weights.shape)

	patch_size = config["dataset"]["pattern"]["params"]["patch_size"]

	# ### Visualize Weights
	if kPlotWeights:
		plotWeights(weights, save_dir)

	# ## Evaluate Model
	# 
	# For all analyses, we use an out-of-distribution test dataset of never-before-seen images. While the network was trained on natural image patches from the Van Hateren dataset, we use exemplars from the MNIST dataset for all model evaluation analyses, to test the generality of the learned map. 

	# ### Invariance / Equivariance Analysis
	# We first perform a qualitative analysis of the invariance and equivariance properties of the network. For this analysis, we randomly draw exemplars from the MNIST dataset, using one exemplar per digit and generating 360 rotations (in 1 degree increments) of each exemplar to form the image orbits. We then pass the data through the model, and examine the output of the first linear term $Wx$ and the output of the full network.

	# **Generate Dataset**

	torch.random.seed()

	# 1 exemplar per digit is randomly selected
	pattern = MNISTExemplars(path="../datasets/mnist/mnist_train.csv", n_exemplars=1)

	print(len(pattern.data), patch_size)

	# Resize the images to the size of the patches the network was trained on
	#resized = torch.stack([torch.tensor(resize(x, (patch_size, patch_size))) for x in pattern.data])
	#pattern.data = resized
