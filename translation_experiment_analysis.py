 #!/usr/bin/env python
# coding: utf-8
from pathlib import Path

from bispectral_networks import projconfig
from bispectral_networks.experiments import translation_experiment

kLog_path = projconfig.getLogsFolder()/"translation_model/"
kSave_dir = Path("notebooks/figs/translation/")


if __name__ == '__main__':
	# # Bispectral Neural Networks - Translation Experiment
	# 
	# This notebook reproduces the plots for the translation experiment. It also allows the user to test the network on datasets generated with different random seeds, to examine the generality of the results. We examine the properties of the network with respect to three criteria:
	# - Invariance and Equivariance
	# - Generalization
	# - Robustness
	translation_experiment.main(
		kSave_dir, 
		kLog_path,
		kWeights=False,
		kImageGrid=False,
		kEquivariance=True,
		kInvariance=True,
		kGeneralization=True,
		kRobustness=True,
	)
