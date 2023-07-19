"""
Title: Load the bsd500 into BSD500Dataset.
	
Created on Wed Jul 17 17:44:29 2023

@author: Manny Ko.
"""
from pathlib import Path
import matplotlib.pyplot as plt


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
