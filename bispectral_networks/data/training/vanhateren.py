"""
Title: Load the van Hateren dataset.
	
Created on Fri May 19 17:44:29 2023

@author: Manny Ko.
"""
#import os
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union
import numpy as np

import datasets.utils.projconfig as projconfig
from datasets.van_hateren.van_hateren import VanHateren as VanHaterenDS

import torch
from torch.utils.data import Dataset


class VanHateren(VanHaterenDS):
	def __init__(
		self,
		path=projconfig.getVanHaterenFolder(),
		normalize=True,
		select_img_path="select_imgs.txt",
		patches_per_image=10,
		patch_size=16,
		min_contrast=1.0,
	):
		#path = projconfig.getVanHaterenFolder() 	#TODO: always load from here for now
		print(f"VanHateren({path=}, {select_img_path=}, {patches_per_image}, {patch_size}, {min_contrast=})")

		super().__init__(
			path=path,
			normalize=normalize,
			select_img_path=select_img_path,
			patches_per_image=patches_per_image, 
			patch_size=patch_size, 
			min_contrast=min_contrast
		)
		full_images = self.images
		print(f"{self.min_contrast=}, {full_images.shape=}")
		self.data, self.labels = self.get_patches(full_images)
		print(f"{self.data.shape=}")
	
	def get_patches(self, full_images):
		data = []
		labels = []

		i = 0
		
		for img in full_images:
			for p in range(self.patches_per_image):
				low_contrast = True
				j = 0 
				while low_contrast and j < 100:
					start_x = np.random.randint(0, self.img_shape[1] - self.patch_size)
					start_y = np.random.randint(0, self.img_shape[0] - self.patch_size)
					patch = img[
						start_y : start_y + self.patch_size, start_x : start_x + self.patch_size
					]
					#print(f"{patch.std()},")
					if patch.std() >= self.min_contrast:
						low_contrast = False
						data.append(patch)
						labels.append(i)
					j += 1
				
				if j == 100 and not low_contrast:
					print("Couldn't find patch to meet contrast requirement. Skipping.")
					continue

				i += 1
		print(f"{len(data)=}, {i=}, {self.min_contrast=}")
		data = torch.tensor(np.array(data))
		labels = torch.tensor(np.array(labels))
		return data, labels
						
	def __getitem__(self, idx):
		x = self.data[idx]
		y = self.labels[idx]
		return x, y

	def __len__(self):
		return len(self.data)
#end of VanHateren

kImgGrid=True

if (__name__ == '__main__'):
	from mkpyutils.imgutils import show_imggrid

	root = projconfig.getRepoRoot()
	vandir = projconfig.getVanHaterenFolder()
	print(vandir)

	#2: subset specified "select_imgs.txt" (from bispectral_networks)
	van_hateren = VanHateren(path=vandir, min_contrast=0.1, select_img_path="select_imgs.txt")
	print(van_hateren, len(van_hateren))

	#2: full dataset
	#van_hateren_full = VanHateren(path=vandir, select_img_path=None)
	#print(van_hateren_full, len(van_hateren_full))

	if kImgGrid:	
		n_row, n_col = (2, 6) 

		imgs = [van_hateren[i].coeffs for i in range(n_row*n_col)]

		imgplot = show_imggrid(
			imgs=imgs, 
			n_row=n_row, n_col=n_col, 
			figsize=(12, 4 ), 
		)
