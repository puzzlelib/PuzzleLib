import multiprocessing
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


libname = "PuzzleLib"
cachepath = None


globalEvalMode = False
disableDtypeShapeChecks = False
disableModuleCompatChecks = False
verifyData = False
showWarnings = True


def isCPUBased(bnd):
	return bnd in {Backend.cpu, Backend.intel}


def shouldInit():
	return multiprocessing.current_process().name == "MainProcess" or allowMultiContext
