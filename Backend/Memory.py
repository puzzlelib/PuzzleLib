from PuzzleLib import Config


depthConcat = None
depthSplit = None

moveaxis = None
swapaxes = None
transpose = None


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
	from PuzzleLib.Cuda.Utils import memoryPool
	from PuzzleLib.Cuda.Wrappers.CuDnn import context

	def wrapDepthConcat(data):
		return context.depthConcat(data, allocator=memoryPool)

	def wrapDepthSplit(grad, indata):
		return context.depthSplit(grad, indata, allocator=memoryPool)

	global depthConcat, depthSplit
	depthConcat = wrapDepthConcat
	depthSplit = wrapDepthSplit

	def wrapMoveaxis(data, src, dst):
		return context.moveaxis(data, src, dst, allocator=memoryPool)

	def wrapSwapaxes(data, axis1, axis2):
		return context.swapaxes(data, axis1, axis2, allocator=memoryPool)

	def wrapTranspose(data, axes):
		return context.transpose(data, tuple(axes), allocator=memoryPool)

	global moveaxis, swapaxes, transpose
	moveaxis = wrapMoveaxis
	swapaxes = wrapSwapaxes
	transpose = wrapTranspose


def initOpenCL():
	from PuzzleLib.OpenCL.Kernels import Memory

	global depthConcat, depthSplit
	depthConcat = Memory.depthConcat
	depthSplit = Memory.depthSplit

	global moveaxis, swapaxes, transpose
	moveaxis = Memory.moveaxis
	swapaxes = Memory.swapaxes
	transpose = Memory.transpose


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
