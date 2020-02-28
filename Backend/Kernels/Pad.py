from PuzzleLib import Config


reflectpad1d = None
reflectpad1dBackward = None

reflectpad2d = None
reflectpad2dBackward = None


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
	from PuzzleLib.Cuda.Kernels import Pad

	global reflectpad1d, reflectpad1dBackward, reflectpad2d, reflectpad2dBackward
	reflectpad1d = reflectpad2d = Pad.reflectpad
	reflectpad1dBackward = reflectpad2dBackward = Pad.reflectpadBackward


def initOpenCL():
	pass


def initCPU():
	from PuzzleLib.CPU.Kernels import Pad

	global reflectpad1d
	reflectpad1d = Pad.reflectpad1d

	global reflectpad2d
	reflectpad2d = Pad.reflectpad2d


autoinit()
