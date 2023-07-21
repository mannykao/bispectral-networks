"""
Title: Load the bsd500 into BSD500Dataset.
	
Created on Wed Jul 17 17:44:29 2023

@author: Manny Ko.
"""
from pathlib import Path
import time, timeit
import matplotlib.pyplot as plt
import numpy as np
import torch

from mkpyutils.testutil import time_spent

from bispectral_networks.analysis.plotting import image_grid
from bispectral_networks.MNIST_inv_eq import MNISTExemplars, MNIST_train_inv_eq_set


def Weights(
	weights,
	save_dir:Path,
	kPlot:bool=False
):
	# **Real Components**
	image_grid(weights.real, cmap="Greys_r", shape=(16, 16), figsize=(15, 15), share_range=False, save_name=save_dir/"W_real.pdf")
	# **Imaginary Components**
	image_grid(weights.imag, cmap="Greys_r", shape=(16, 16), figsize=(15, 15), share_range=False, save_name=save_dir/"W_imag.pdf")

	if kPlot: plt.show()

def Robustness(
	checkpoint:torch.nn.Module,
	patch_size:int,
	save_dir:str,
	device="cpu",
	kPlot:bool=False,
):
	# Now, we analyze the robustness of the model through a simple adversarial example experiment. 
	# Here, we start with noise as the input, and optimize this input to yield a network output that is as close as 
	# possible to the network output for a target image. For this analysis, we select a single exemplar from the
	# MNIST dataset as the target, 
	# and run the optimization starting from 100 different noise images to examine the range of results. 
	# Note that the BasicGradientDescent method often gets stuck in local minima that are farther from the target 
	# embedding than desired to be considered a meaningful "adversarial example." While such runs are interesting regardless, 
	# the approach should be rerun until the target reaches the margin. Fancier optimization methods could be used to escape these 
	# local minima, but here we use a vanilla model.
	# 
	# We find that, for this model, points that are close in embedding space will be (close to) equivalent up to the group action 
	# that the model has learned to be invariant to. In this notebook we use a margin of 0.1, which is larger than that used 
	# in the paper (since it tends to be difficult to drive the model lower). This means that the examples are more likely 
	# to look different from the target. Despite this, we find the same effect, which further demonstrates the robustness of this model.
	print("Adversarial Robustness")
	start = time.time()
	from bispectral_networks.analysis.adversary import BasicGradientDescent
	from bispectral_networks.analysis.plotting import animated_video, image_grid

	#torchutils.initSeeds(1)		#torch.random.seed()

	robustness_dataset = MNIST_train_inv_eq_set(
		path=None,	#kMNIST_path,
		n_exemplars=1,	# 1 exemplar per digit is randomly selected
		patch_size=16,
		fraction_transforms=1.0,
		xformkind="center",
		seed=1
	)
	print(f" {robustness_dataset.data.shape=}")

	cpudevice = "cpu"
	target_idx = np.random.randint(len(robustness_dataset.data))
	target_image = robustness_dataset.data[target_idx].to(cpudevice)
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
	plt.savefig(save_dir/"adversary.pdf")

	time_spent(start, f"Adversarial Robustness: ", count=1)

	if kPlot: plt.show()

	# **Visualizing the Optimization Process**
	# 
	# The below video shows the optimization of a single example image through gradient steps.

	animated_video(history[:, 0], interval=100, cmap="Greys_r")

