import sys, multiprocessing, logging
from enum import Enum


class ConfigError(Exception):
	pass


class Backend(Enum):
	cuda = 0
	hip = 1
	cpu = 2
	intel = 3


backend = Backend.cuda
deviceIdx = 0


allowMultiContext = False
systemLog = False
logger = None


libname = "PuzzleLib"


globalEvalMode = False
disableDtypeShapeChecks = False
disableModuleCompatChecks = False
verifyData = False
showWarnings = True


def isCPUBased(bnd):
	return bnd in {Backend.cpu, Backend.intel}


def shouldInit():
	return multiprocessing.current_process().name == "MainProcess" or allowMultiContext


def getLogger():
	global logger

	if logger is not None:
		return logger

	logger = logging.getLogger(libname)
	logger.setLevel(logging.DEBUG if systemLog else logging.INFO)

	handler = logging.StreamHandler(stream=sys.stdout)
	handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))

	logger.addHandler(handler)
	return logger
