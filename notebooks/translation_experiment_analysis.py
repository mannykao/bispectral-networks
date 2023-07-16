 #!/usr/bin/env python
# coding: utf-8
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union
from pathlib import Path
import time, timeit
from mkpyutils.testutil import time_spent

import sys
sys.path.append("../")
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
sns.set(font_scale=1.5)
sns.set_style('dark')

from mk_mlutils.utils import torchutils

from rotation_experiment_analysis import Weights
#from rotation_experiment_analysis import Weights, Equivariance, Invariance, Generalization, Robustness

kWeights=False
kEquivariance=False
kInvariance=False
kGeneralization=True
kRobustness=True

def CyclicTranslation2DXform(fraction_transforms=1.0) -> List:
	transform1 = CyclicTranslation2D(fraction_transforms=fraction_transforms, sample_method="linspace")
	transform2 = CenterMean()
	transform3 = UnitStd()
	transform4 = Ravel()
	return [transform1, transform2, transform3, transform4]

def MNIST_train_inv_eq_set(
	path="../datasets/mnist/mnist_train.csv", 
	n_exemplars=1,	# 1 exemplar per digit is randomly selected
	patch_size:int=16,
	fraction_transforms=1.0,	#use subsampled rotation xform or full 
)  -> TransformDataset:
	torch.random.seed()
	
	pattern = MNISTExemplars(path=path, n_exemplars=n_exemplars)

	# Resize the images to the size of the patches the network was trained on
	resized = torch.stack([torch.tensor(resize(x.numpy(), (patch_size, patch_size))) for x in pattern.data])
	pattern.data = resized

	# Apply transformations
	transforms = CyclicTranslation2DXform(fraction_transforms=fraction_transforms)
	inv_eq_dataset = TransformDataset(pattern, transforms)
	return inv_eq_dataset

def Equivariance(
	inv_eq_dataset,
	out, 				#output of model (foreward)
	save_dir
):
	""" **First Linear Term (Equivariance)** 
		Output: save_dir/"equivariance-0|1|2.pdf"
	"""
	print("Equivariance")
	start = time.time()

	out = out.detach().numpy()
	l_out = checkpoint.model.layers[0].forward_linear(inv_eq_dataset.data).detach().numpy()
	l_out = l_out.real + 1j * l_out.imag

	image_grid(inv_eq_dataset.data[::16][:8].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(l_out[::16][:16, [1, 13, 22, 3]].real,);
	plt.axis([-1, 16, -4.5, 4.5]);
	plt.savefig(os.path.join(save_dir, "equivariance-0.pdf"))

	image_grid(inv_eq_dataset.data[::16][16:24].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(l_out[::16][16:32, [1, 13, 22, 3]].real,);
	plt.axis([-1, 16, -4.5, 4.5]);
	plt.savefig(os.path.join(save_dir, "equivariance-1.pdf"))

	image_grid(inv_eq_dataset.data[::16][32:40].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(l_out[::16][32:48, [1, 13, 22, 3]].real,);
	plt.axis([-1, 16, -4.5, 4.5]);
	plt.savefig(os.path.join(save_dir, "equivariance-2.pdf"))

	time_spent(start, f"Elapsed Time: ", count=1)

def Invariance(
	inv_eq_dataset:torch.utils.data.Dataset,
	out, 				#output of model (foreward)
	save_dir:str,
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
	plt.savefig(os.path.join(save_dir, "invariance-0.pdf"))

	image_grid(inv_eq_dataset.data[::16][16:24].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(out[::16][16:32].real,);
	plt.axis([-1, 16, -0.2, 0.3]);
	plt.savefig(os.path.join(save_dir, "invariance-1.pdf"))

	image_grid(inv_eq_dataset.data[::16][32:40].reshape(-1, 16, 16), shape=(1, 8), cmap="Greys_r")

	plt.figure(figsize=(7, 7))
	plt.plot(out[::16][32:48].real,);
	plt.axis([-1, 16, -0.2, 0.3]);
	plt.savefig(os.path.join(save_dir, "invariance-2.pdf"))

	time_spent(start, f"Elapsed Time: ", count=1)

def Generalization(
	checkpoint:torch.nn.Module,
	patch_size:int, 
	k:int,				#k in knn - i.e. number of clusters
	save_dir:str,
):
	# We next quantify the generalization performance of the network on the test data. 
	#Here, we use a smaller subset of the rotations, to make it computationally feasible to compute pairwise distances between the network outputs for all datapoints. We then examine the k-nearest neighbors of each datapoint and examine the fraction of k that are correctly classified as within-orbit. K is set to the number of elements in the orbit. Here, since 10% of the orbit was selected (fraction_transforms parameter in CyclicTranslation2D), the orbit has 25 elements. A model that perfectly collapses orbits should achieve 100% classification accuracy on this metric.
	print("Generalization")
	from bispectral_networks.analysis.knn import knn_analysis
	start = time.time()

	# 1 exemplar per digit is randomly selected

	# Apply transformations
	knn_dataset = MNIST_train_inv_eq_set(
		path="../datasets/mnist/mnist_train.csv", 
		n_exemplars=1,	# 1 exemplar per digit is randomly selected
		patch_size=16,
		fraction_transforms=0.1
	)	
	print(f" {knn_dataset.data.shape=}")

#	k = 25
	embeddings, distance_matrix, knn_scores = knn_analysis(checkpoint.model, knn_dataset, k)

	print(" The model successfully classified {:.2f}% of the orbit on average.".format(knn_scores[0] * 100))
	print(" The model misclassified {:.2f}% of the orbit on average.".format(knn_scores[1] * 100))

	plt.figure(figsize=(15, 15))
	im = plt.imshow(distance_matrix)
	plt.colorbar(im, fraction=0.046, pad=0.04)
	plt.savefig(save_dir + "test_distance_matrix.pdf")

	time_spent(start, f"Elapsed Time: ", count=1)


def Robustness(
	checkpoint:torch.nn.Module,
	patch_size:int,
	save_dir:str,
):
	# ## Robustness Analysis

	# Now, we analyze the robustness of the model through a simple adversarial example experiment. Here, we start with noise as the input, and optimize this input to yield a network output that is as close as possible to the network output for a target image. For this analysis, we select a single exemplar from the MNIST dataset as the target, and run the optimization starting from 100 different noise images to examine the range of results. Note that the BasicGradientDescent method often gets stuck in local minima that are farther from the target embedding than desired to be considered a meaningful "adversarial example." While such runs are interesting regardless, the approach should be rerun until the target reaches the margin. Fancier optimization methods could be used to escape these local minima, but here we use a vanilla model.
	# 
	# We find that, for this model, points that are close in embedding space will be (close to) equivalent up to the group action that the model has learned to be invariant to. In this notebook we use a margin of 0.1, which is larger than that used in the paper (since it tends to be difficult to drive the model lower). This means that the examples are more likely to look different from the target. Despite this, we find the same effect, which further demonstrates the robustness of this model.
	print("adversarial Robustness")
	start = time.time()

	from bispectral_networks.analysis.adversary import BasicGradientDescent
	from bispectral_networks.analysis.plotting import animated_video

	torch.random.seed()

	# 1 exemplar per digit is randomly selected
	pattern = MNISTExemplars(path="../datasets/mnist/mnist_train.csv", n_exemplars=1)

	# Resize the images to the size of the patches the network was trained on
	resized = torch.stack([torch.tensor(resize(x.numpy(), (patch_size, patch_size))) for x in pattern.data])
	pattern.data = resized

	# Apply transformations
	transform1 = CenterMean()
	transform2 = UnitStd()
	transform3 = Ravel()
	robustness_dataset = TransformDataset(pattern, [transform1, transform2, transform3])
	print(f" {robustness_dataset.data.shape=}")

	device = "cpu"
	target_idx = np.random.randint(len(robustness_dataset.data))
	target_image = robustness_dataset.data[target_idx].to(device)
	target_image_tiled = torch.tensor(np.tile(target_image, (100, 1)))

	plt.imshow(target_image.reshape(patch_size, patch_size), cmap="Greys_r")

	adversary = BasicGradientDescent(model=checkpoint.model, 
									 target_image=target_image_tiled,
									 margin=.1,
									 lr=0.1,
									 save_interval=10,
									 print_interval=100,
									 optimizer=torch.optim.Adam,
									 device=device)

	x_, target_embedding, embedding = adversary.train(max_iter=1000)

	history = np.array([x.reshape(-1, patch_size, patch_size) for x in adversary.history])
	history.shape

	image_grid(history[-1], shape=(10, 10), cmap="Greys_r", figsize=(20, 20))
	plt.savefig(os.path.join(save_dir, "adversary.pdf"))

	time_spent(start, f"Elapsed Time: ", count=1)

	# **Visualizing the Optimization Process**
	# 
	# The below video shows the optimization of a single example image through gradient steps.

	animated_video(history[:, 0], interval=100, cmap="Greys_r")


if __name__ == '__main__':
	# # Bispectral Neural Networks - Translation Experiment
	# 
	# This notebook reproduces the plots for the translation experiment. It also allows the user to test the network on datasets generated with different random seeds, to examine the generality of the results. We examine the properties of the network with respect to three criteria:
	# - Invariance and Equivariance
	# - Generalization
	# - Robustness
	print("Bispectral Neural Networks - Translation Experiment..")

	SEED = 0
	device = torchutils.onceInit(kCUDA=True, seed=SEED)
 
	save_dir = "figs/translation/"
	os.makedirs(save_dir, exist_ok=True)

	# ### Load Checkpoint
	# 
	# The default in this notebook is to use the pretrained model from the paper, which is located at `../logs/translation_model/`
	# 
	# Alternatively, the user can load a new model by changing the `log_path` below to the location of the log folder for that model.

	log_path = "../logs/translation_model/"
	checkpoint, config, weights = load_checkpoint(log_path)

	print(f"{weights.dtype=}, {weights.shape=}")

	patch_size = config["dataset"]["pattern"]["params"]["patch_size"]

	# ### Visualize Weights
	if kWeights:
		Weights(weights, save_dir)

	# ## Evaluate Model
	# 
	# For all analyses, we use an out-of-distribution test dataset of never-before-seen images. While the network was trained on natural image patches from the Van Hateren dataset, we use exemplars from the MNIST dataset for all model evaluation analyses, to test the generality of the learned map. 

	# ### Invariance / Equivariance Analysis
	# We first perform a qualitative analysis of the invariance and equivariance properties of the network. For this analysis, we randomly draw exemplars from the MNIST dataset, using one exemplar per digit and generating all 256 integer translations of each exemplar to form the image orbits. We then pass the data through the model, and examine the output of the first linear term $Wx$ and the output of the full network.

	# **Generate Dataset**

	inv_eq_dataset = MNIST_train_inv_eq_set(
		path="../datasets/mnist/mnist_train.csv", 
		n_exemplars=1,	# 1 exemplar per digit is randomly selected
		patch_size=16,
		fraction_transforms=1.0
	)
	print(inv_eq_dataset.data.shape)

	image_grid(inv_eq_dataset.data[::10][:144].reshape(-1, patch_size, patch_size), shape=(12, 12), figsize=(15, 15), cmap="Greys_r", save_name=os.path.join(save_dir, "test_examples.pdf"))


	# **Pass Data Through Model**

	out, _ = checkpoint.model(inv_eq_dataset.data)

	# **First Linear Term (Equivariance)**
	# 
	# The plots below show the outputs of 3 neurons after the first linear term $Wx$ is computed, on the data below: a single digit swept linearly through a translation.  Each colored line represents a single neuron, the y-axis shows the neuron's response, and the x axis corresponds to the translation.
	if kEquivariance:
		Equivariance(inv_eq_dataset, out, save_dir)

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
		Robustness(checkpoint, patch_size, save_dir)

