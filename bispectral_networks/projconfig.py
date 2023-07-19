"""
Title: projconfig.

Created on Wed July 19 17:44:29 2023

@author: Manny Ko &  Ujjawal.K.Panchal.
"""
#import re
from pathlib import Path, PurePosixPath
from typing import Tuple, Callable, Iterable, List, Any, Dict, Union

kOurRepo="bispectral_networks"	#<srcroot>/datasets
kOurVenv="venv4bisp"			#<srcroot/venv4bisp
#
# globals configuration for the root of our repo.
# anchor for all other files especially datasets, external packages (whl) etc.
#
kRepoName="bispectral_networks"		#the name of our package as defined in setup.py
#kRepoRoot="mldatasets"				#the root of our repo for the maintainer/developer
#from kRepoRoot to the sources
kToSrcRoot=""		#kRepoRoot/kToSrcRoot = "d:/Dev/SuperPixelSeg/mldatasets/"


def getRefFile() -> Path:
	return Path(__file__)

def extractRepoRoot(reffile:str=__file__, reporoot:str=kRepoName) -> Path:
	"""	
	reffile 'D:\\Dev\\ML\\SuperPixelSeg\\venv4seg\\lib\\site-packages\\datasets\\utils\\projconfig.py'
	"""
	#print(f"{reffile=}")
	parts = Path(reffile).parts
	index = 0

	ourroot = Path("")
	if (reporoot in parts):
#		print(f"found {reporoot} in {reffile}")
		for i, part in enumerate(parts):
#			if part == reporoot:
			if "lib" in part:			#look for 'venv4seg/lib/'
#				ourroot /=  part
				index = i
				break
			else:					
				ourroot /=  part
		ourroot = Path(*ourroot.parts[0:-1])	#remove 'venv4seg'				
		#ourroot = Path(*Path(reffile).parts[0:index-3])		#remove 'venv4seg/lib/site-package'
	return ourroot 		#"d:/Dev/SuperPixelSeg"

kOurRoot=extractRepoRoot(__file__, kRepoName) / kToSrcRoot

def getLogsFolder() -> Path:
	""" get the checkpoints/log folder inside the installed package """
	logsdir = getRefFile().parent / 'logs'		#assume we are at the root of the package 'bispectral_networks'
	return logsdir
