from PuzzleLib import Config


embed = None
embedBackwardParams = None


def autoinit():
	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.opencl:
		initOpenCL()
	elif Config.isCPUBased(Config.backend):
		initCPU()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda.Kernels import Embedder

	global embed, embedBackwardParams
	embed = Embedder.embed
	embedBackwardParams = Embedder.embedBackwardParams


def initOpenCL():
	from PuzzleLib.OpenCL.Kernels import Embedder

	global embed, embedBackwardParams
	embed = Embedder.embed
	embedBackwardParams = Embedder.embedBackwardParams


def initCPU():
	pass


autoinit()
