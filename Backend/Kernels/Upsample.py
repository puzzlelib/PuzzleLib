from PuzzleLib import Config


upsample2d = None
upsample2dBackward = None

upsample3d = None
upsample3dBackward = None


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
	from PuzzleLib.Cuda.Kernels import Upsample

	global upsample2d, upsample2dBackward
	upsample2d = Upsample.upsample2d
	upsample2dBackward = Upsample.upsample2dBackward

	global upsample3d, upsample3dBackward
	upsample3d = Upsample.upsample3d
	upsample3dBackward = Upsample.upsample3dBackward


def initOpenCL():
	from PuzzleLib.OpenCL.Kernels import Upsample2D, Upsample3D

	global upsample2d, upsample2dBackward
	upsample2d = Upsample2D.upsample2d
	upsample2dBackward = Upsample2D.upsample2dBackward

	global upsample3d, upsample3dBackward
	upsample3d = Upsample3D.upsample3d
	upsample3dBackward = Upsample3D.upsample3dBackward


def initCPU():
	from PuzzleLib.CPU.Kernels import Upsample2D

	global upsample2d
	upsample2d = Upsample2D.upsample2d


autoinit()
