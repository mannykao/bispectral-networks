import sys
#sys.path.append('../')
import time
from mkpyutils.testutil import time_spent

from configs.translation_experiment import master_config
from bispectral_networks.data.utils import gen_dataset


if (__name__ == '__main__'):
	#import datasets.utils.projconfig as projconfig
	from bispectral_networks.data.datasets import VanHateren
	import torch
	import numpy as np

	seed = 1
	torch.manual_seed(seed)
	np.random.seed(seed)

	vandir = "mldatasets/datasets/van_hateren/van_hateren/"  #projconfig.getVanHaterenFolder()
	print(vandir)

	#1: subset specified "select_imgs.txt" (from bispectral_networks)
	van_hateren = VanHateren(path=vandir, patches_per_image=3, min_contrast=0.1, select_img_path="select_imgs.txt")
	print(van_hateren, len(van_hateren))	

	#2: use gen_dataset(master_config)
	dataset = gen_dataset(master_config["dataset"])

	data_loader = master_config["data_loader"].build()
	data_loader.load(dataset)

	print(len(data_loader.train.dataset.data))

	start1 = time.time()
	for x, labels in data_loader.train:
		print(".", end='')
	end1 = time_spent(start1, 'data_loader.train: ')
