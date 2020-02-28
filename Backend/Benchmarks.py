from PuzzleLib import Config


timeKernel = None


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
	from PuzzleLib.Cuda.Benchmarks import Utils

	global timeKernel
	timeKernel = Utils.timeKernel


def initOpenCL():
	from PuzzleLib.OpenCL.Benchmarks import Utils

	global timeKernel
	timeKernel = Utils.timeKernel


def initCPU():
	from PuzzleLib.CPU.Benchmarks import Utils

	global timeKernel
	timeKernel = Utils.timeKernel


autoinit()
