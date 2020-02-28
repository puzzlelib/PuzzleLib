from PuzzleLib import Config


SharedArray = None
memoryPool = None

streamManager = None
globalRng = None

copy = None
concatenate = None
split = None
tile = None

fillUniform = None
fillNormal = None

setupDebugAllocator = None
dtypesSupported = None


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
	from PuzzleLib.Cuda import Utils as CudaUtils

	global SharedArray, memoryPool, streamManager, globalRng
	SharedArray = CudaUtils.SharedArray
	memoryPool = CudaUtils.memoryPool

	streamManager = CudaUtils.streamManager
	globalRng = CudaUtils.globalRng

	global copy, concatenate, split, tile
	copy = CudaUtils.copy
	concatenate = CudaUtils.concatenate
	split = CudaUtils.split
	tile = CudaUtils.tile

	global fillUniform, fillNormal
	fillUniform = CudaUtils.fillUniform
	fillNormal = CudaUtils.fillNormal

	global setupDebugAllocator, dtypesSupported
	setupDebugAllocator = CudaUtils.setupDebugAllocator
	dtypesSupported = CudaUtils.dtypesSupported


def initOpenCL():
	from PuzzleLib.OpenCL import Utils as OpenCLUtils

	global SharedArray, memoryPool, streamManager, globalRng
	SharedArray = OpenCLUtils.SharedArray
	memoryPool = OpenCLUtils.memoryPool

	streamManager = OpenCLUtils.streamManager
	globalRng = OpenCLUtils.globalRng

	global copy, concatenate, split, tile
	copy = OpenCLUtils.copy
	concatenate = OpenCLUtils.concatenate
	split = OpenCLUtils.split
	tile = OpenCLUtils.tile

	global fillUniform, fillNormal
	fillUniform = OpenCLUtils.fillUniform
	fillNormal = OpenCLUtils.fillNormal

	global setupDebugAllocator, dtypesSupported
	setupDebugAllocator = OpenCLUtils.setupDebugAllocator
	dtypesSupported = OpenCLUtils.dtypesSupported


def initCPU():
	import numpy as np

	from PuzzleLib.CPU.CPUArray import CPUArray
	from PuzzleLib.CPU import Utils

	class ProxyMemoryPool:
		def freeHeld(self):
			pass

	class ProxyRNG:
		@staticmethod
		def fillUniform(data, minval=0.0, maxval=1.0):
			data.set(np.random.uniform(minval, maxval, size=data.shape))

		@staticmethod
		def fillNormal(data, mean=0.0, sigma=1.0):
			data.set(np.random.normal(mean, sigma, size=data.shape))

		@staticmethod
		def fillInteger(data):
			data.set(np.random.randint(np.iinfo(data.dtype).min, np.iinfo(data.dtype).max, dtype=data.dtype))

	global SharedArray, memoryPool, globalRng
	SharedArray = Utils.SharedArray
	memoryPool = ProxyMemoryPool()
	globalRng = ProxyRNG()

	def wrapCopy(dest, source):
		if dest is None:
			dest = CPUArray.empty(source.shape, source.dtype)

		np.copyto(dest.data, source.data)
		return dest

	def wrapConcatenate(tup, axis, out=None):
		out = np.concatenate(tuple(ary.data for ary in tup), axis=axis, out=None if out is None else out.data)
		return CPUArray(out.shape, out.dtype, data=out, acquire=True)

	def wrapSplit(ary, sections, axis):
		outs = (out.copy() for out in np.split(ary.data, np.cumsum(sections)[:-1], axis))
		return [CPUArray(out.shape, out.dtype, data=out, acquire=True) for out in outs]

	def wrapTile(ary, times, axis):
		shape = (times, )
		if axis > 0:
			shape = (1, ) * axis + shape
		if axis < ary.ndim - 1:
			shape = shape + (1, ) * (ary.ndim - 1 - axis)

		out = np.tile(ary.data, shape)
		return CPUArray(out.shape, out.dtype, data=out, acquire=True)

	global copy, concatenate, split, tile
	copy = wrapCopy
	concatenate = wrapConcatenate
	split = wrapSplit
	tile = wrapTile

	def wrapFillUniform(data, minval, maxval, rng):
		rng.fillUniform(data, minval, maxval)

	def wrapFillNormal(data, mean, sigma, rng):
		rng.fillNormal(data, mean, sigma)

	global fillUniform, fillNormal
	fillUniform = wrapFillUniform
	fillNormal = wrapFillNormal

	global setupDebugAllocator, dtypesSupported
	setupDebugAllocator = lambda: None
	dtypesSupported = Utils.dtypesSupported


autoinit()
