import torch
import os
import datetime
import copy
from pathlib import Path

from bispectral_networks.logger import load_checkpoint
from bispectral_networks.config import Config

log_path = "../logs/rotation_model/"


if __name__ == '__main__':
	checkpoint, config, weights = load_checkpoint(Path(log_path), device="cpu")

	#out, _ = checkpoint.model(inv_eq_dataset.data)
