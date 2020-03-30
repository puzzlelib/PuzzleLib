import functools, operator
from collections import OrderedDict

import numpy as np


def roundUpDiv(a, b):
	return (a + b - 1) // b


def roundUp(a, b):
	return roundUpDiv(a, b) * b


def prod(seq, start=1):
	return functools.reduce(operator.mul, seq, start)


class SharedArray:
	GPUArray = None
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

		self.ary = self.GPUArray.empty(
			shape=(totalbytes // self.dtype.itemsize, ), dtype=self.dtype, allocator=self.allocator
		)

		blocks = OrderedDict()
		offset = 0

		for name, (shape, nbytes) in self.blocks.items():
			blocks[name] = self.GPUArray(
				shape=shape, dtype=self.dtype, gpudata=self.ary.gpudata[offset:offset + nbytes]
			)
			offset += self.align(nbytes)

		self.blocks = blocks


	def __getitem__(self, item):
		return self.blocks[item]


	@classmethod
	def align(cls, nbytes):
		return (nbytes + cls.alignment - 1) // cls.alignment * cls.alignment


class QueueManager:
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


def setupDebugAllocator(GPUArray):
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


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx, initmode=2)

		for dtype, _ in bnd.dtypesSupported():
			shareMemTest(bnd, dtype)
			memCopyTest(bnd, dtype)

		randomTest(bnd)


def shareMemTest(bnd, dtype):
	shMem = bnd.SharedArray(dtype=dtype)

	shMem.register((10, 10, 10), dtype, "a")
	shMem.register((50, 1, 5), dtype, "b")
	shMem.build()

	a, b = shMem["a"], shMem["b"]
	assert a.shape == (10, 10, 10) and a.dtype == dtype
	assert b.shape == (50, 1, 5) and b.dtype == dtype


def memCopyTest(bnd, dtype):
	hostSrc = np.random.randn(4, 4, 4, 4).astype(dtype)

	src = bnd.GPUArray.toGpu(hostSrc)
	assert np.allclose(hostSrc, src.copy().get())

	hostA = np.random.randn(7, 4, 4, 4).astype(dtype)
	a = bnd.GPUArray.toGpu(hostA)

	out = bnd.concatenate((src, a), axis=0)
	assert np.allclose(np.concatenate((hostSrc, hostA), axis=0), out.get())

	hostA = np.random.randn(4, 2, 4, 4).astype(dtype)
	hostB = np.random.randn(4, 1, 4, 4).astype(dtype)

	a, b = bnd.GPUArray.toGpu(hostA), bnd.GPUArray.toGpu(hostB)

	out = bnd.concatenate((src, a, b), axis=1)
	assert np.allclose(np.concatenate((hostSrc, hostA, hostB), axis=1), out.get())

	hostA = np.random.randn(4, 4, 5, 4).astype(dtype)

	out = bnd.concatenate((bnd.GPUArray.toGpu(hostA), src), axis=2)
	assert np.allclose(np.concatenate((hostA, hostSrc), axis=2), out.get())

	hostA = np.random.randn(4, 4, 4, 5).astype(dtype)

	out = bnd.concatenate((bnd.GPUArray.toGpu(hostA), src), axis=3)
	assert np.allclose(np.concatenate((hostA, hostSrc), axis=3), out.get())

	outs = bnd.split(src, (2, 2), axis=0)
	assert all(np.allclose(hostSrc[2 * i:2 * (i + 1)], out.get()) for i, out in enumerate(outs))

	outs = bnd.split(src, (2, 2), axis=1)
	assert all(np.allclose(hostSrc[:, 2 * i:2 * (i + 1), :, :], out.get()) for i, out in enumerate(outs))

	outs = bnd.split(src, (2, 2), axis=2)
	assert all(np.allclose(hostSrc[:, :, 2 * i:2 * (i + 1), :], out.get()) for i, out in enumerate(outs))

	outs = bnd.split(src, (2, 2), axis=3)
	assert all(np.allclose(hostSrc[:, :, :, 2 * i:2 * (i + 1)], out.get()) for i, out in enumerate(outs))

	assert np.allclose(np.tile(hostB, (1, 3, 1, 1)), bnd.tile(b, 3, axis=1).get())


def randomTest(bnd):
	data = bnd.GPUArray.empty((100, ), dtype=np.float32)

	bnd.fillUniform(data, minval=-1.0, maxval=1.0)
	bnd.fillNormal(data, mean=1.0, stddev=0.1)


if __name__ == "__main__":
	unittest()
