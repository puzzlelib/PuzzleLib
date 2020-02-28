from PuzzleLib import Config


prelu = None
preluBackwardData = None
preluBackwardParams = None


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
	from PuzzleLib.Cuda.Kernels import PRelu

	global prelu, preluBackwardData, preluBackwardParams
	prelu = PRelu.prelu
	preluBackwardData = PRelu.preluBackwardData
	preluBackwardParams = PRelu.preluBackwardParams


def initOpenCL():
	from PuzzleLib.OpenCL.Kernels import PRelu

	global prelu, preluBackwardData, preluBackwardParams
	prelu = PRelu.prelu
	preluBackwardData = PRelu.preluBackwardData
	preluBackwardParams = PRelu.preluBackwardParams


def initCPU():
	pass


autoinit()
