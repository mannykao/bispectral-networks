import sys
#sys.path.append('../')
import time
from mkpyutils.testutil import time_spent

from configs.translation_experiment import master_config
from bispectral_networks.data.utils import gen_dataset

def cmp_datasets(ds1, ds2) -> bool:
	print(f"cmp_datasets -> {len(ds1)}, {len(ds2)}")
	result = (len(ds1) == len(ds2))
	if not result:
		print(f"cmp_datasets failed..")
		return result
	for i, entry1 in enumerate(ds1):
		entry2 = ds2[i]
		result &= np.array_equal(entry1[0], entry2[0])
		result &= entry1[1] == entry2[1]
		if not result:
			print(f"[{i}]:", entry1, entry2)
			break
	return result
		

if (__name__ == '__main__'):
	#import datasets.utils.projconfig as projconfig
	from bispectral_networks.data.datasets import VanHateren
	import torch
	import numpy as np

	dataset_config = master_config["dataset"]

	seed = dataset_config["seed"]
	torch.manual_seed(seed)
	np.random.seed(seed)

	vandir = "mldatasets/datasets/van_hateren/van_hateren/"  #projconfig.getVanHaterenFolder()
	print(vandir)

	#1: subset specified "select_imgs.txt" (from bispectral_networks)
	van_hateren = VanHateren(path=vandir, 
		patches_per_image=3, min_contrast=0.1, 
		select_img_path="select_imgs.txt",
	)
	print(van_hateren, len(van_hateren))	

	#2: use gen_dataset(master_config)
	dataset = gen_dataset(master_config["dataset"])

	print(f"{cmp_datasets(van_hateren, dataset.dataset)=}")

	data_loader = master_config["data_loader"].build()
	data_loader.load(dataset)

	print(len(data_loader.train.dataset.data))

	start1 = time.time()
	for x, labels in data_loader.train:
		print(".", end='')
	end1 = time_spent(start1, 'data_loader.train: ')
