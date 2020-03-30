from PuzzleLib import Config


backend = None

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
	from PuzzleLib.Cuda import Backend, Utils
	initGPU(Backend, Utils)


def initHip():
	from PuzzleLib.Cuda import Utils
	from PuzzleLib.Hip import Backend
	initGPU(Backend, Utils)


def initGPU(Backend, Utils):
	global backend
	backend = Backend.getBackend(Config.deviceIdx, initmode=0)

	global SharedArray, memoryPool, streamManager, globalRng
	SharedArray = backend.SharedArray
	memoryPool = backend.memoryPool

	streamManager = backend.streamManager
	globalRng = backend.globalRng

	def wrapCopy(dest, source):
		return backend.copy(dest, source, allocator=memoryPool)

	def wrapConcatenate(tup, axis, out=None):
		return backend.concatenate(tup, axis, out, allocator=memoryPool)

	def wrapSplit(ary, sections, axis):
		return backend.split(ary, sections, axis, allocator=memoryPool)

	def wrapTile(ary, times, axis):
		return backend.tile(ary, times, axis, allocator=memoryPool)

	global copy, concatenate, split, tile
	copy = wrapCopy
	concatenate = wrapConcatenate
	split = wrapSplit
	tile = wrapTile

	def wrapFillUniform(data, minval, maxval, rng):
		backend.fillUniform(data, minval, maxval, rng)

	def wrapFillNormal(data, mean, stddev, rng):
		backend.fillNormal(data, mean, stddev, rng)

	global fillUniform, fillNormal
	fillUniform = wrapFillUniform
	fillNormal = wrapFillNormal

	global setupDebugAllocator, dtypesSupported
	setupDebugAllocator = lambda: Utils.setupDebugAllocator(backend.GPUArray)
	dtypesSupported = backend.dtypesSupported


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
