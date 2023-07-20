"""
Title: Load the van Hateren dataset.
	
Created on Fri May 19 17:44:29 2023

@author: Manny Ko.
"""
#import os
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union

import datasets.utils.projconfig as projconfig
from datasets.van_hateren.van_hateren import VanHateren

kImgGrid=True

if (__name__ == '__main__'):
	from mkpyutils.imgutils import show_imggrid

	root = projconfig.getRepoRoot()
	vandir = projconfig.getVanHaterenFolder()
	print(vandir)

	#2: subset specified "select_imgs.txt" (from bispectral_networks)
	van_hateren = VanHateren(path=vandir, select_img_path="select_imgs.txt")
	print(van_hateren, len(van_hateren))

	#2: full dataset
	van_hateren_full = VanHateren(path=vandir, select_img_path=None)
	print(van_hateren_full, len(van_hateren_full))

	#print(f"{van_hateren_full[0]}=")

	if kImgGrid:	
		n_row, n_col = (2, 6) 

		imgs = [van_hateren[i].coeffs for i in range(n_row*n_col)]

		imgplot = show_imggrid(
			imgs=imgs, 
			n_row=n_row, n_col=n_col, 
			figsize=(12, 4 ), 
		)
