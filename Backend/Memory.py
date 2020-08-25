from PuzzleLib import Config


depthConcat = None
depthSplit = None

moveaxis = None
swapaxes = None
transpose = None


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

	backend = getBackend(Config.deviceIdx, initmode=1, logger=Config.getLogger())
	memoryPool, dnn = backend.memoryPool, backend.dnn

	initGPU(memoryPool, dnn)


def initHip():
	from PuzzleLib.Hip.Backend import getBackend

	backend = getBackend(Config.deviceIdx, initmode=2, logger=Config.getLogger())
	memoryPool, memmod = backend.memoryPool, backend.memmod

	initGPU(memoryPool, memmod)


def initGPU(memoryPool, module):
	def wrapDepthConcat(data):
		return module.depthConcat(data, allocator=memoryPool)

	def wrapDepthSplit(grad, indata):
		return module.depthSplit(grad, indata, allocator=memoryPool)

	global depthConcat, depthSplit
	depthConcat = wrapDepthConcat
	depthSplit = wrapDepthSplit

	def wrapMoveaxis(data, src, dst):
		return module.moveaxis(data, src, dst, allocator=memoryPool)

	def wrapSwapaxes(data, axis1, axis2):
		return module.swapaxes(data, axis1, axis2, allocator=memoryPool)

	def wrapTranspose(data, axes):
		return module.transpose(data, tuple(axes), allocator=memoryPool)

	global moveaxis, swapaxes, transpose
	moveaxis = wrapMoveaxis
	swapaxes = wrapSwapaxes
	transpose = wrapTranspose


def initCPU():
	import numpy as np
	from PuzzleLib.CPU.CPUArray import CPUArray

	def wrapMoveAxis(a, src, dst):
		out = np.copy(np.moveaxis(a.get(copy=False), src, dst), order="C")
		return CPUArray(out.shape, out.dtype, data=out, acquire=True)

	def wrapSwapAxes(a, axis1, axis2):
		out = np.copy(np.swapaxes(a.get(copy=False), axis1, axis2), order="C")
		return CPUArray(out.shape, out.dtype, data=out, acquire=True)

	def wrapTranspose(a, axes):
		out = np.copy(np.transpose(a.get(copy=False), axes), order="C")
		return CPUArray(out.shape, out.dtype, data=out, acquire=True)

	global moveaxis, swapaxes, transpose
	moveaxis = wrapMoveAxis
	swapaxes = wrapSwapAxes
	transpose = wrapTranspose


autoinit()
