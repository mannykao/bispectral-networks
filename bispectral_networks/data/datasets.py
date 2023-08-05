import torch
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd

from bispectral_networks.data.training.vanhateren import VanHateren

class TransformDataset:
	def __init__(self, dataset, transforms):
		"""
		Arguments
		---------
		dataset (obj):
			Object from patterns.natural or patterns.synthetic
		transforms (list of obj):
			List of objects from transformations. The order of the objects
			determines the order in which they are applied.
		"""
		self.dataset = dataset 	#stash original 'dataset' during refactoring to use mldatasets
		
		if type(transforms) != list:
			transforms = [transforms]
		self.transforms = transforms
		self.gen_transformations(dataset)
		if len(self.data.shape) == 3:
			self.img_size = tuple(self.data.shape[1:])
		else:
			self.dim = self.data.shape[-1]

	def gen_transformations(self, dataset):
		transform_dict = OrderedDict()
		transformed_data = dataset.data.clone()
		new_labels = dataset.labels.clone()
		for transform in self.transforms:
			transformed_data, new_labels, transform_dict, t = transform(
				transformed_data, new_labels, transform_dict
			)
			transform_dict[transform.name] = t
		self.data = transformed_data
		self.labels = new_labels
		self.transform_labels = transform_dict

	def __getitem__(self, idx):
		x = self.data[idx]
		y = self.labels[idx]
		return x, y

	def __len__(self):
		return len(self.data)
	
	
class MNISTExemplars(Dataset):
	"""
	Dataset object for the MNIST dataset.
	Takes the MNIST file path, then loads, standardizes, and saves it internally.
	"""

	def __init__(self, path, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], n_exemplars=1):

		super().__init__()

		self.name = "mnist"
		self.dim = 28 ** 2
		self.img_size = (28, 28)
		self.digits = digits
		self.n_exemplars = n_exemplars

		mnist = np.array(pd.read_csv(path))

		labels = mnist[:, 0]
		mnist = mnist[:, 1:]
		mnist = mnist / 255
		mnist = mnist - mnist.mean(axis=1, keepdims=True)
		mnist = mnist / mnist.std(axis=1, keepdims=True)
		mnist = mnist.reshape((len(mnist), 28, 28))
		
		label_idxs = {i: [j for j, x in enumerate(labels) if x == i] for i in range(10)}
		
		exemplar_data = []
		labels = []
		for d in digits:
			idxs = label_idxs[d]
			random_idxs = np.random.choice(idxs, size=self.n_exemplars, replace=False)
			for i in random_idxs:
				exemplar_data.append(np.asarray(mnist[i], dtype=np.float32))
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


if (__name__ == '__main__'):
	from mkpyutils.imgutils import show_imggrid
	#import datasets.utils.projconfig as projconfig

	vandir = "mldatasets/datasets/van_hateren/van_hateren/"  #projconfig.getVanHaterenFolder()
	print(vandir)

	#2: subset specified "select_imgs.txt" (from bispectral_networks)
	van_hateren = VanHateren(path=vandir, min_contrast=0.1, select_img_path="select_imgs.txt")
	print(van_hateren, len(van_hateren))		
