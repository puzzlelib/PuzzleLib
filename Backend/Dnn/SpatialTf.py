from PuzzleLib import Config


spatialTf = None
spatialTfBackward = None


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
	from PuzzleLib.Cuda.Wrappers.CuDnn import context
	from PuzzleLib.Cuda.Utils import memoryPool

	def wrapSpatialTf(data, transform, outshape, getGrid):
		return context.spatialTf(data, transform, outshape, getGrid, allocator=memoryPool)

	def wrapSpatialTfBackward(grad, data, grid):
		return context.spatialTfBackward(grad, data, grid, allocator=memoryPool)

	global spatialTf, spatialTfBackward
	spatialTf = wrapSpatialTf
	spatialTfBackward = wrapSpatialTfBackward


def initOpenCL():
	pass


def initCPU():
	pass


autoinit()
