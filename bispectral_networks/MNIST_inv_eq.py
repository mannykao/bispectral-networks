"""
Title: Load the bsd500 into BSD500Dataset.
	
Created on Wed Feb 1 17:44:29 2023

@author: Manny Ko.
"""
import time, timeit
from pathlib import Path
import numpy as np
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union
from collections import Counter

import torch
from torch.utils.data import Dataset
import pandas as pd

from datasets.utils import projconfig
from datasets.mnist import mnist
from datasets import dataset_base
from mkpyutils.testutil import time_spent
from mk_mlutils.utils import torchutils

# Dataset Imports
from bispectral_networks.data.datasets import MNISTExemplars, TransformDataset
from bispectral_networks.data.transforms import SO2, CircleCrop, CenterMean, UnitStd, Ravel
from bispectral_networks.data.transforms import CyclicTranslation2D
from skimage.transform import resize


kMNIST_path = "../tdatasets/mnist/mnist_train.csv"

def load_mnistCSV(path) -> tuple:
	print("load_mnistCSV")
	mnist = np.array(pd.read_csv(path))
	labels = mnist[:, 0]
	images = mnist[:, 1:]
#	images = images.reshape((len(images), 28, 28))
	return images, labels

class MNISTExemplars(Dataset):
	"""
	Dataset object for the MNIST dataset.
	Takes the MNIST file path, then loads, standardizes, and saves it internally.
	"""
	def __init__(self, 
		path, 	#path to the .csv or None to use mnist in mldatasets
		digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
		n_exemplars=1
	):
		super().__init__()

		self.name = "mnist"
		self.dim = 28 ** 2
		self.img_size = (28, 28)
		self.digits = digits
		self.n_exemplars = n_exemplars

		if (type(path) is str) or (isinstance(path, Path)):
			images, labels = load_mnistCSV(path)
		else:
			#print("mnist.MNIST(split='train')")
			mnist_data = mnist.MNIST(split='train')
			images = np.asarray(getCoeffs(mnist_data))
			labels = dataset_base.getLabels(mnist_data)
			
		images = images / 255
		mean = images.mean(axis=1, keepdims=True)
		std  = images.std(axis=1, keepdims=True)

		mniimagesst = images - mean
		images = images / std
		images = images.reshape((len(images), 28, 28))

		label_idxs = {i: [j for j, x in enumerate(labels) if x == i] for i in range(10)}
		
		exemplar_data = []
		labels = []
		for d in digits:
			idxs = label_idxs[d]
			random_idxs = np.random.choice(idxs, size=self.n_exemplars, replace=False)
			for i in random_idxs:
				exemplar_data.append(np.asarray(images[i], dtype=np.float32))
				labels.append(d)
			
		#print(len(exemplar_data))	
		self.data = torch.tensor(np.array(exemplar_data))
		self.labels = torch.tensor(labels).long()

	def __getitem__(self, idx):
		x = self.data[idx]
		y = self.labels[idx]
		return x, y

	def __len__(self):
		return len(self.data)

def CenterXform(fraction_transforms=None) -> List:
	transform1 = CenterMean()
	transform2 = UnitStd()
	transform3 = Ravel()
	return [transform1, transform2, transform3]

def CyclicTranslation2DXform(fraction_transforms=1.0) -> List:
	transform1 = CyclicTranslation2D(fraction_transforms=fraction_transforms, sample_method="linspace")
	transform2 = CenterMean()
	transform3 = UnitStd()
	transform4 = Ravel()
	return [transform1, transform2, transform3, transform4]

def SO2Xform(fraction_transforms=1.0) -> List:
	transform1 = SO2(fraction_transforms=fraction_transforms, sample_method="linspace")
	transform2 = CircleCrop()
	transform3 = CenterMean()
	transform4 = UnitStd()
	transform5 = Ravel()
	return [transform1, transform2, transform3, transform4, transform5]

def MNIST_train_inv_eq_set(
	path=kMNIST_path,  	#path to the .csv or None to use mnist in mldatasets
	n_exemplars=1,		# 1 exemplar per digit is randomly selected
	patch_size:int=16,
	fraction_transforms=1.0,	#use subsampled rotation xform or full
	xformkind:str="SO2",		#'SO2|cyclicxlat|center'
	seed:int=0, 
)  -> TransformDataset:
	print(f" MNIST_train_inv_eq_set({fraction_transforms=}, {xformkind=})")
	torchutils.initSeeds(seed)		#torch.random.seed()
	
	pattern = MNISTExemplars(path=path, n_exemplars=n_exemplars)

	# Resize the images to the size of the patches the network was trained on
	resized = torch.stack([torch.tensor(resize(x.numpy(), (patch_size, patch_size))) for x in pattern.data])
	pattern.data = resized

	# Apply transformations
	dispatch = {
		'SO2':			SO2Xform,
		'cyclicxlat':	CyclicTranslation2DXform,
		'center':		CenterXform,
	}
	transforms = dispatch[xformkind](fraction_transforms=fraction_transforms)
	inv_eq_dataset = TransformDataset(pattern, transforms)
	return inv_eq_dataset

def getCoeffs(dataset):
	if isinstance(dataset, dataset_base.DataSet):
		coeffs = [item.coeffs.flatten() for item in dataset]
	else:
		coeffs = [item[0].flatten() for item in dataset]
	return coeffs

def verify_stats(path):
	images1, labels1 = load_mnistCSV(path)

	mean1 = images1.mean(axis=1, keepdims=False)
	std1  = images1.std(axis=1, keepdims=False)

	mnisttrain = mnist.MNIST(split='train')
	print(f"{mnisttrain=}, {len(mnisttrain)}")
	
	images2 = np.asarray(getCoeffs(mnisttrain))
	print(f"{images2.shape=}")
	mean2 = images2.mean(axis=1, keepdims=False)
	std2  = images2.std(axis=1, keepdims=False)

	print(f"{mean1.shape=} {mean2.shape=}")

	if np.equal(mean1, mean2).all():
		print("np.equal(mean1, mean2) = equal")
	else:
		#print(f"{mean1[0:10]}, {mean2[0:10]}")	
		print("two means are different")	

	if np.isclose(std1, std2).all():
		print("np.equal(std1, std2) = equal")
	else:
		#print(f"{std1[0:10]}, {std2[0:10]}")	
		print("two std are different")	

	labels2 = dataset_base.getLabels(mnisttrain)
	if np.equal(labels1, labels2).all():
		print("labels match")


if __name__ == '__main__':
	torchutils.initSeeds(0)		#torch.random.seed()

	#verify_stats(path=kMNIST_path)	

	start = time.time()
	mnisttrain = mnist.MNIST(split='train')
	time_spent(start, f"datasets.mnist: ", count=1)
	print(f"{mnisttrain=}, {len(mnisttrain)}")

	start = time.time()
	inv_eq_dataset = MNIST_train_inv_eq_set(
		#path=kMNIST_path,
		path=None, 
		n_exemplars=1,	# 1 exemplar per digit is randomly selected
		patch_size=16,
		fraction_transforms=1.0
	)
	time_spent(start, f"MNIST_train_inv_eq_set: ", count=1)

