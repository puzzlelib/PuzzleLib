import multiprocessing, sys, atexit
import functools, operator
from collections import OrderedDict

import numpy as np

from PuzzleLib import Config

from PuzzleLib.Cuda import Driver
from PuzzleLib.Cuda.Driver import CudaError, CuRand
from PuzzleLib.Cuda.GPUArray import GPUArray

from PuzzleLib.Cuda.Kernels.ElementWise import linearKer


device = None

memoryPool = None
globalRng = None

streamManager = None
eventManager = None


warpSize = 32
nthreads = 1024


def roundUpDiv(a, b):
	return (a + b - 1) // b


def roundUp(a, b):
	return roundUpDiv(a, b) * b


def prod(seq, start=1):
	return functools.reduce(operator.mul, seq, start)


def dtypesSupported():
	return [(np.float32, 1e-5), (np.float16, 1e-2)]


class SharedArray:
	alignment = 16


	def __init__(self, dtype=np.float32, allocator=None):
		self.ary = None
		self.blocks = OrderedDict()

		self.dtype = np.dtype(dtype)
		self.allocator = allocator


	def register(self, shape, dtype, name):
		assert name not in self.blocks
		assert dtype == self.dtype

		self.blocks[name] = (shape, prod(shape) * self.dtype.itemsize)


	def build(self):
		totalbytes = sum(self.align(nbytes) for _, nbytes in self.blocks.values())

		self.ary = GPUArray.empty(
			shape=(totalbytes // self.dtype.itemsize, ), dtype=self.dtype, allocator=self.allocator
		)

		blocks = OrderedDict()
		offset = 0

		for name, (shape, nbytes) in self.blocks.items():
			blocks[name] = GPUArray(shape=shape, dtype=self.dtype, gpudata=self.ary.gpudata[offset:offset + nbytes])
			offset += self.align(nbytes)

		self.blocks = blocks


	def __getitem__(self, item):
		return self.blocks[item]


	@classmethod
	def align(cls, nbytes):
		return (nbytes + cls.alignment - 1) // cls.alignment * cls.alignment


class CudaQueue:
	def __init__(self, objtype):
		self.objtype = objtype
		self.items = []


	def reserve(self, nitems):
		self.items.extend(self.objtype() for _ in range(nitems))


	def borrow(self, nitems):
		if len(self.items) < nitems:
			self.reserve(nitems - len(self.items))

		newEnd = len(self.items) - nitems

		borrowed = self.items[newEnd:]
		self.items = self.items[:newEnd]

		return borrowed


	def give(self, items):
		self.items.extend(items)


	def clear(self):
		self.items.clear()


def finishUp():
	if memoryPool is not None:
		memoryPool.stopHolding()


def autoinit():
	atexit.register(finishUp)

	ndevices = Driver.Device.count()
	if ndevices == 0:
		raise CudaError("No CUDA enabled device found")

	if Config.deviceIdx >= ndevices:
		raise CudaError("Invalid CUDA config device index")

	global device
	device = Driver.Device(Config.deviceIdx).set()
	print("[%s] Using device #%s (%s)" % (Config.libname, Config.deviceIdx, device.name()), flush=True)

	if Config.systemLog:
		print(
			"[%s] Created Cuda context (Using driver version: %s)" % (Config.libname, Driver.getDriverVersion()),
			flush=True
		)

	global memoryPool
	memoryPool = Driver.MemoryPool()

	rngtype, seed = CuRand.RAND_RNG_PSEUDO_XORWOW, int(np.random.randint(sys.maxsize, dtype=np.intp))

	global globalRng
	globalRng = CuRand.RandomNumberGenerator(type=rngtype, seed=seed)

	if Config.systemLog:
		print("[%s] Created CuRand global rng (type=%s, seed=%s)" % (Config.libname, rngtype, hex(seed)), flush=True)

	global streamManager, eventManager
	streamManager, eventManager = CudaQueue(objtype=Driver.Stream), CudaQueue(objtype=Driver.Event)


if device is None and (multiprocessing.current_process().name == "MainProcess" or Config.allowMultiContext):
	autoinit()


def setupDebugAllocator():
	empty = GPUArray.empty

	def emptyDebug(shape, dtype, allocator=None):
		ary = empty(shape, dtype, allocator)
		dtype = np.dtype(dtype).type

		if issubclass(dtype, np.floating):
			value = dtype(np.nan)
		elif issubclass(dtype, np.integer):
			value = np.iinfo(dtype).max
		else:
			raise NotImplementedError(dtype)

		ary.fill(value)
		return ary

	GPUArray.empty = emptyDebug


def getDeviceComputeCap(index=Config.deviceIdx):
	return Driver.Device(index).computeCapability()


def fillUniform(data, minval=0.0, maxval=1.0, rng=globalRng):
	assert data.dtype == np.float32
	rng.fillUniform(data)

	dtype = data.dtype
	linearKer(dtype)(data, data, dtype.type(maxval - minval), dtype.type(minval))


def fillNormal(data, mean=0.0, stddev=1.0, rng=globalRng):
	rng.fillNormal(data, mean=mean, stddev=stddev)


def copy(dest, source, allocator=None):
	if dest is None:
		return source.copy(allocator=allocator)
	else:
		dest.set(source)
		return dest


def dstack(tup):
	return concatenate(tup, axis=2)


def hstack(tup):
	return concatenate(tup, axis=1)


def vstack(tup):
	return concatenate(tup, axis=0)


def dsplit(ary, sections):
	return split(ary, sections, axis=2)


def hsplit(ary, sections):
	return split(ary, sections, axis=1)


def vsplit(ary, sections):
	return split(ary, sections, axis=0)


def concatenate(tup, axis, out=None, allocator=memoryPool):
	ary = tup[0]

	dtype, reducedShape = ary.dtype, ary.shape
	reducedShape = reducedShape[:axis] + reducedShape[axis + 1:]

	assert all(a.dtype == dtype and a.shape[:axis] + a.shape[axis + 1:] == reducedShape for a in tup[1:])

	concatDim = sum(a.dimAt(axis) for a in tup)
	shape = reducedShape[:axis] + (concatDim, ) + reducedShape[axis:]

	if out is None:
		out = GPUArray.empty(shape, dtype=dtype, allocator=allocator)
	else:
		assert out.shape == shape and out.dtype == dtype

	dstPitch = out.strideAt(axis - 1) if axis > 0 else out.nbytes
	height = prod(shape[:axis])

	stride = 0

	for a in tup:
		srcPitch = width = a.strideAt(axis - 1) if axis > 0 else a.nbytes

		Driver.memcpy2D(width, height, a.gpudata, srcPitch, out.gpudata, dstPitch, dstX=stride)
		stride += width

	return out


def split(ary, sections, axis, allocator=memoryPool):
	shape = ary.shape
	assert sum(sections) == shape[axis]

	outs = [
		GPUArray.empty(shape[:axis] + (sec, ) + shape[axis + 1:], dtype=ary.dtype, allocator=allocator)
		for sec in sections
	]

	srcPitch = ary.strideAt(axis - 1) if axis > 0 else ary.nbytes
	height = prod(shape[:axis])

	stride = 0

	for out in outs:
		dstPitch = width = out.strideAt(axis - 1) if axis > 0 else out.nbytes

		Driver.memcpy2D(width, height, ary.gpudata, srcPitch, out.gpudata, dstPitch, srcX=stride)
		stride += width

	return outs


def tile(ary, repeats, axis, allocator=memoryPool):
	return concatenate([ary] * repeats, axis=axis, allocator=allocator)


def unittest():
	for dtype, _ in dtypesSupported():
		shareMemTest(dtype)
		memCopyTest(dtype)

	randomTest()


def shareMemTest(dtype):
	shMem = SharedArray(dtype=dtype)

	shMem.register((10, 10, 10), dtype, "a")
	shMem.register((50, 1, 5), dtype, "b")
	shMem.build()

	a, b = shMem["a"], shMem["b"]
	assert a.shape == (10, 10, 10) and a.dtype == dtype
	assert b.shape == (50, 1, 5) and b.dtype == dtype


def memCopyTest(dtype):
	hostSrc = np.random.randn(4, 4, 4, 4).astype(dtype)

	src = GPUArray.toGpu(hostSrc)
	assert np.allclose(hostSrc, src.copy().get())

	hostA = np.random.randn(7, 4, 4, 4).astype(dtype)
	a = GPUArray.toGpu(hostA)

	out = concatenate((src, a), axis=0)
	assert np.allclose(np.concatenate((hostSrc, hostA), axis=0), out.get())

	hostA = np.random.randn(4, 2, 4, 4).astype(dtype)
	hostB = np.random.randn(4, 1, 4, 4).astype(dtype)

	a, b = GPUArray.toGpu(hostA), GPUArray.toGpu(hostB)

	out = concatenate((src, a, b), axis=1)
	assert np.allclose(np.concatenate((hostSrc, hostA, hostB), axis=1), out.get())

	hostA = np.random.randn(4, 4, 5, 4).astype(dtype)

	out = concatenate((GPUArray.toGpu(hostA), src), axis=2)
	assert np.allclose(np.concatenate((hostA, hostSrc), axis=2), out.get())

	hostA = np.random.randn(4, 4, 4, 5).astype(dtype)

	out = concatenate((GPUArray.toGpu(hostA), src), axis=3)
	assert np.allclose(np.concatenate((hostA, hostSrc), axis=3), out.get())

	outs = split(src, (2, 2), axis=0)
	assert all(np.allclose(hostSrc[2 * i:2 * (i + 1)], out.get()) for i, out in enumerate(outs))

	outs = split(src, (2, 2), axis=1)
	assert all(np.allclose(hostSrc[:, 2 * i:2 * (i + 1), :, :], out.get()) for i, out in enumerate(outs))

	outs = split(src, (2, 2), axis=2)
	assert all(np.allclose(hostSrc[:, :, 2 * i:2 * (i + 1), :], out.get()) for i, out in enumerate(outs))

	outs = split(src, (2, 2), axis=3)
	assert all(np.allclose(hostSrc[:, :, :, 2 * i:2 * (i + 1)], out.get()) for i, out in enumerate(outs))

	assert np.allclose(np.tile(hostB, (1, 3, 1, 1)), tile(b, 3, axis=1).get())


def randomTest():
	data = GPUArray.empty((100, ), dtype=np.float32)

	fillUniform(data, minval=-1.0, maxval=1.0)
	fillNormal(data, mean=1.0, stddev=0.1)


if __name__ == "__main__":
	unittest()
