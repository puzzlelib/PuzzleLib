from PuzzleLib import Config


spatialTf = None
spatialTfBackward = None


def autoinit():
	if not Config.shouldInit():
		return

	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.hip:
		initHip()
	elif Config.isCPUBased(Config.backend):
		initCPU()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda.Backend import getBackend

	backend = getBackend(Config.deviceIdx, initmode=1)
	memoryPool, dnn = backend.memoryPool, backend.dnn

	def wrapSpatialTf(data, transform, outshape, getGrid):
		return dnn.spatialTf(data, transform, outshape, getGrid, allocator=memoryPool)

	def wrapSpatialTfBackward(grad, data, grid):
		return dnn.spatialTfBackward(grad, data, grid, allocator=memoryPool)

	global spatialTf, spatialTfBackward
	spatialTf = wrapSpatialTf
	spatialTfBackward = wrapSpatialTfBackward


def initHip():
	pass


def initCPU():
	pass


autoinit()
