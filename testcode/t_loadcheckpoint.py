from pathlib import Path

from bispectral_networks import projconfig
from bispectral_networks.logger import load_checkpoint
from bispectral_networks.config import Config

kLoadRef=False	#load the reference checkpoint from our package
log_path = "../logs/rotation_model/"	#local reference checkpoint folder


if __name__ == '__main__':
	from mk_mlutils.utils import torchutils
	device = torchutils.onceInit(kCUDA=True, seed=0)

	#checkpoint, config, weights = load_checkpoint(Path(log_path), device="cpu")
	#print(checkpoint, config)

	#reffile = projconfig.getRefFile()
	#print(projconfig.extractRepoRoot(reffile), reffile.name, reffile.parent)

	#1: get the checkpoints folder inside the installed package
	log_path = Path("../logs")

	#1: load the reference checkpoint from our package
	if kLoadRef:
		checkpoint_dir = projconfig.getLogsFolder()/'rotation_model/'
	else:
		checkpoint_dir = log_path/'xlat10E-0721'
	print(checkpoint_dir)

	checkpoint, config, weights = load_checkpoint(checkpoint_dir, device=device)
	print(checkpoint)
	print(torchutils.is_cuda(checkpoint.model))
	#print(config)
