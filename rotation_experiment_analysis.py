#!/usr/bin/env python
# coding: utf-8
import argparse
from pathlib import Path

from bispectral_networks import projconfig
from bispectral_networks.experiments import rotation_experiment

kMNIST_path = None	#"datasets/mnist/mnist_train.csv"
kLog_path = projconfig.getLogsFolder()/"rotation_model"		#"logs/rotation_model/"
kSave_dir = Path("notebooks/figs/rotation/")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--logs",
		type=str,
		help="checkpoints folder",
		default=kLog_path,
	)
	parser.add_argument("--device", type=int, help="device to run on, -1 for cpu", default=0)
	args = parser.parse_args()

	log_path = args.logs

	# # Bispectral Neural Networks - Rotation Experiment
	# 
	# This notebook reproduces the plots for the rotation experiment. It also allows the user to test the network on datasets generated with different random seeds, to examine the generality of the results. We examine the properties of the network with respect to three criteria:
	# - Invariance and Equivariance
	# - Generalization
	# - Robustness
	rotation_experiment.main(
		log_path=log_path,
		save_dir=kSave_dir,
		kWeights=False,
		kImageGrid=False,
		kEquivariance=True,
		kInvariance=True,
		kGeneralization=True,
		kRobustness=True,
	)
