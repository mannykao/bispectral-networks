 #!/usr/bin/env python
# coding: utf-8

#
# Adapted and heavily refactored from bispectral_networks.notebooks.translation_experiment_analysis
#
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union
from pathlib import Path
import time, timeit
from mkpyutils.testutil import time_spent

import sys
#sys.path.append("../")
import os
import torch
import numpy as np
from bispectral_networks.logger import load_checkpoint
from bispectral_networks.analysis.plotting import image_grid

# Dataset Imports
from bispectral_networks.data.datasets import MNISTExemplars, TransformDataset
from bispectral_networks.data.transforms import CyclicTranslation2D, CenterMean, UnitStd, Ravel
from skimage.transform import resize

# Plotting imports
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
#get_ipython().run_line_magic('matplotlib', 'inline')

from mk_mlutils.utils import torchutils

from bispectral_networks import projconfig
from bispectral_networks.MNIST_inv_eq import MNISTExemplars, MNIST_train_inv_eq_set
import bispectral_networks.MNIST_inv_eq as MNIST_inv_eq
from bispectral_networks.experiments.rotation_experiment import Weights, Robustness
#from rotation_experiment_analysis import Weights, Equivariance, Invariance, Generalization, Robustness

kMNIST_path = None	#"datasets/mnist/mnist_train.csv"
kLog_path = projconfig.getLogsFolder()/"translation_model/"
kSave_dir = Path("notebooks/figs/translation/")

kWeights=False
kImageGrid=False
kEquivariance=True
kInvariance=True
kGeneralization=True
kRobustness=True

 
def Equivariance(
	checkpoint:torch.nn.Module,
	inv_eq_dataset,
	save_dir
):
	""" **First Linear Term (Equivariance)** 
		Output: save_dir/"equivariance-0|1|2.pdf"
	"""
	print("Equivariance")
	start = time.time()

	l_out = checkpoint.model.layers[0].forward_linear(inv_eq_dataset.data).detach().numpy()
	l_out = l_out.real + 1j * l_out.imag

	image_grid(inv_eq_dataset.data[::16][:8].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(l_out[::16][:16, [1, 13, 22, 3]].real,);
	plt.axis([-1, 16, -4.5, 4.5]);
	plt.savefig(save_dir/"equivariance-0.pdf")

	image_grid(inv_eq_dataset.data[::16][16:24].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(l_out[::16][16:32, [1, 13, 22, 3]].real,);
	plt.axis([-1, 16, -4.5, 4.5]);
	plt.savefig(save_dir/"equivariance-1.pdf")

	image_grid(inv_eq_dataset.data[::16][32:40].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(l_out[::16][32:48, [1, 13, 22, 3]].real,);
	plt.axis([-1, 16, -4.5, 4.5]);
	plt.savefig(save_dir/"equivariance-2.pdf")

	time_spent(start, f"Equivariance: ", count=1)

def Invariance(
	inv_eq_dataset:torch.utils.data.Dataset,
	out, 				#output of model (foreward)
	save_dir:Path,
):
	"""
		Output: save_dir/"invariance-0|1|2.pdf"
	"""
	print("Invariance")
	start = time.time()
	out = out.detach().numpy()

	image_grid(inv_eq_dataset.data[::16][:8].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(out[::16][:16].real,);
	plt.axis([-1, 16, -0.2, 0.3]);
	plt.savefig(save_dir/"invariance-0.pdf")

	image_grid(inv_eq_dataset.data[::16][16:24].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(out[::16][16:32].real,);
	plt.axis([-1, 16, -0.2, 0.3]);
	plt.savefig(save_dir/"invariance-1.pdf")

	image_grid(inv_eq_dataset.data[::16][32:40].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(out[::16][32:48].real,);
	plt.axis([-1, 16, -0.2, 0.3]);
	plt.savefig(save_dir/"invariance-2.pdf")

	time_spent(start, f"Invariance: ", count=1)

def Generalization(
	checkpoint:torch.nn.Module,
	patch_size:int, 
	k:int,				#k in knn - i.e. number of clusters
	save_dir:Path,
):
	# We next quantify the generalization performance of the network on the test data. 
	#Here, we use a smaller subset of the rotations, to make it computationally feasible to compute pairwise distances between the network outputs for all datapoints. We then examine the k-nearest neighbors of each datapoint and examine the fraction of k that are correctly classified as within-orbit. K is set to the number of elements in the orbit. Here, since 10% of the orbit was selected (fraction_transforms parameter in CyclicTranslation2D), the orbit has 25 elements. A model that perfectly collapses orbits should achieve 100% classification accuracy on this metric.
	print("Generalization")
	from bispectral_networks.analysis.knn import knn_analysis
	start = time.time()
	#torchutils.initSeeds(0)		#torch.random.seed()

	# 1 exemplar per digit is randomly selected

	# Apply transformations
	knn_dataset = MNIST_train_inv_eq_set(
		path=kMNIST_path, 
		n_exemplars=1,	# 1 exemplar per digit is randomly selected
		patch_size=16,
		fraction_transforms=0.1,
		xformkind="cyclicxlat",
	)	
	print(f" {knn_dataset.data.shape=}")

#	k = 25
	embeddings, distance_matrix, knn_scores = knn_analysis(checkpoint.model, knn_dataset, k)
	start = time_spent(start, f"knn_analysis: ", count=1)

	print(" The model successfully classified {:.2f}% of the orbit on average.".format(knn_scores[0] * 100))
	print(" The model misclassified {:.2f}% of the orbit on average.".format(knn_scores[1] * 100))

	plt.figure(figsize=(15, 15))
	im = plt.imshow(distance_matrix)
	plt.colorbar(im, fraction=0.046, pad=0.04)
	plt.savefig(save_dir/"test_distance_matrix.pdf")

	time_spent(start, f"Generalization: ", count=1)


def main(
	save_dir=kSave_dir,
	log_path=kLog_path,
	kWeights=False,
	kImageGrid=False,
	kEquivariance=False,
	kInvariance=False,
	kGeneralization=True,
	kRobustness=True,	
):
	print("Bispectral Neural Networks - Translation Experiment..")

	sns.set(font_scale=1.5)
	sns.set_style('dark')

	SEED = 0
	device = torchutils.onceInit(kCUDA=True, seed=SEED)
 
	os.makedirs(save_dir, exist_ok=True)

	# ### Load Checkpoint
	# 
	# The default in this notebook is to use the pretrained model from the paper, which is located at `../logs/translation_model/`
	# 
	# Alternatively, the user can load a new model by changing the `log_path` below to the location of the log folder for that model.
	start = time.time()

	checkpoint, config, weights = load_checkpoint(log_path, device=device)
	print(f"{weights.dtype=}, {weights.shape=}")
	time_spent(start, f"load_checkpoint: ", count=1)

	patch_size = config["dataset"]["pattern"]["params"]["patch_size"]

	# ### Visualize Weights
	if kWeights:
		start = time.time()
		Weights(weights, save_dir)
		time_spent(start, f"Weights: ", count=1)

	# ## Evaluate Model
	# 
	# For all analyses, we use an out-of-distribution test dataset of never-before-seen images. While the network was trained on natural image patches from the Van Hateren dataset, we use exemplars from the MNIST dataset for all model evaluation analyses, to test the generality of the learned map. 

	# ### Invariance / Equivariance Analysis
	# We first perform a qualitative analysis of the invariance and equivariance properties of the network. For this analysis, we randomly draw exemplars from the MNIST dataset, using one exemplar per digit and generating all 256 integer translations of each exemplar to form the image orbits. We then pass the data through the model, and examine the output of the first linear term $Wx$ and the output of the full network.

	# **Generate Dataset**
	start = time.time()

	inv_eq_dataset = MNIST_train_inv_eq_set(
		path=kMNIST_path,
		n_exemplars=1,	# 1 exemplar per digit is randomly selected
		patch_size=16,
		fraction_transforms=1.0,
		xformkind="cyclicxlat",
	)
	print(inv_eq_dataset.data.shape)
	time_spent(start, f"MNIST_train_inv_eq_set: ", count=1)

	if kImageGrid:
		start = time.time()
		image_grid(inv_eq_dataset.data[::10][:144].reshape(-1, patch_size, patch_size), shape=(12, 12), figsize=(15, 15), cmap="Greys_r", save_name=os.path.join(save_dir, "test_examples.pdf"))
		time_spent(start, f"image_grid(inv_eq_dataset): ", count=1)

	# **Pass Data Through Model**
	start = time.time()
	out, _ = checkpoint.model(inv_eq_dataset.data)
	time_spent(start, f"checkpoint.model(inv_eq_dataset.data): ", count=1)

	# **First Linear Term (Equivariance)**
	# 
	# The plots below show the outputs of 3 neurons after the first linear term $Wx$ is computed, on the data below: a single digit swept linearly through a translation.  Each colored line represents a single neuron, the y-axis shows the neuron's response, and the x axis corresponds to the translation.
	if kEquivariance:
		Equivariance(checkpoint, inv_eq_dataset, save_dir)

	# **Invariance**
	# 
	# The plots below show the outputs of all neurons at the output of the network, computed on single digit swept linearly through a translation. Each colored line represents a single neuron, the y-axis shows the neuron's response, and the x axis corresponds to the translation.
	if kInvariance:
		Invariance(inv_eq_dataset, out, save_dir)

	# ## Robustness Analysis
	# 
	# We next quantify the generalization performance of the network on the test data. Here, we use a smaller subset of the rotations, to make it computationally feasible to compute pairwise distances between the network outputs for all datapoints. We then examine the k-nearest neighbors of each datapoint and examine the fraction of k that are correctly classified as within-orbit. K is set to the number of elements in the orbit. Here, since 10% of the orbit was selected (fraction_transforms parameter in CyclicTranslation2D), the orbit has 25 elements. A model that perfectly collapses orbits should achieve 100% classification accuracy on this metric.
	if kGeneralization:
		Generalization(checkpoint, patch_size=16, k=25, save_dir=save_dir)

	if kRobustness:
		Robustness(checkpoint, patch_size, save_dir, device=device)


if __name__ == '__main__':
	# # Bispectral Neural Networks - Translation Experiment
	# 
	# This notebook reproduces the plots for the translation experiment. It also allows the user to test the network on datasets generated with different random seeds, to examine the generality of the results. We examine the properties of the network with respect to three criteria:
	# - Invariance and Equivariance
	# - Generalization
	# - Robustness
	translation_experiment.main(
		log_path=kLog_path,
		save_dir=kSave_dir,
		kWeights=False,
		kImageGrid=False,
		kEquivariance=True,
		kInvariance=True,
		kGeneralization=True,
		kRobustness=True,
	)
