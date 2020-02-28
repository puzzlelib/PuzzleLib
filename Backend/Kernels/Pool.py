from PuzzleLib import Config


maxpool2d = None
maxpool2dBackward = None
maxunpool2d = None
maxunpool2dBackward = None


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
	from PuzzleLib.Cuda.Kernels import Pool

	global maxpool2d, maxpool2dBackward, maxunpool2d, maxunpool2dBackward
	maxpool2d = Pool.maxpool2d
	maxpool2dBackward = Pool.maxpool2dBackward
	maxunpool2d = Pool.maxunpool2d
	maxunpool2dBackward = Pool.maxunpool2dBackward


def initOpenCL():
	from PuzzleLib.OpenCL.Kernels import Pool

	global maxpool2d, maxpool2dBackward, maxunpool2d, maxunpool2dBackward
	maxpool2d = Pool.maxpool2d
	maxpool2dBackward = Pool.maxpool2dBackward
	maxunpool2d = Pool.maxunpool2d
	maxunpool2dBackward = Pool.maxunpool2dBackward


def initCPU():
	pass


autoinit()
