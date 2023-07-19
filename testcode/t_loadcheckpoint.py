from pathlib import Path

from bispectral_networks import projconfig
from bispectral_networks.logger import load_checkpoint
from bispectral_networks.config import Config

log_path = "../logs/rotation_model/"	#local checkpoint folder


if __name__ == '__main__':
	#checkpoint, config, weights = load_checkpoint(Path(log_path), device="cpu")
	#print(checkpoint, config)

	#1: get the checkpoints folder inside the installed package
	reffile = projconfig.getRefFile()
	print(projconfig.extractRepoRoot(reffile), reffile.name, reffile.parent)
	print(projconfig.getLogsFolder())

	checkpoint, config, weights = load_checkpoint(projconfig.getLogsFolder()/'rotation_model/')
	print(checkpoint)
	print(config.keys())
