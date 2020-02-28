import multiprocessing
from collections import deque

import numpy as np

from PuzzleLib import Config
from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Kernels import Templates, Random
from PuzzleLib.OpenCL.Kernels.ElementWise import linearKer, addKer


context = None
queue = None
platform = None
device = None

memoryPool = None
globalRng = None

streamManager = None
globalFills = None


def dtypesSupported():
	return [(np.float32, 1e-5)]


class SharedArray:
	def __init__(self, dtype=np.float32):
		self.regs = []

		self.mem = None
		self.dtype = dtype

		self.blocks = {}
		self.offset = 0

		self.ary = None


	def register(self, shape, dtype, name):
		self.regs.append((shape, dtype, name))


	def build(self):
		nbytes = 0
		for reg in self.regs:
			shape, dtype, _ = reg
			assert dtype == self.dtype

			itemsize = dtype(0).itemsize
			align = device.mem_base_addr_align

			nbytes += (np.prod(shape) * itemsize + align - 1) // align * align

		self.mem = Driver.Buffer(context, nbytes)

		for shape, dtype, name in self.regs:
			self.blocks[name] = Driver.empty(queue, shape=shape, dtype=dtype, data=self.allocate(shape, dtype))

		self.regs.clear()
		self.ary = Driver.empty(queue, shape=(nbytes // self.dtype(0).itemsize, ), dtype=self.dtype, data=self.mem)


	def __getitem__(self, item):
		return self.blocks[item]


	def allocate(self, shape, dtype):
		nbytes = np.prod(shape) * dtype(0).itemsize

		align = device.mem_base_addr_align
		nbytes = (nbytes + align - 1) // align * align

		assert self.offset + nbytes <= self.mem.size

		alloc = self.mem[self.offset:self.offset + nbytes]
		self.offset += nbytes

		return alloc


class StreamManager:
	def __init__(self):
		self.streams = deque([])


	def reserve(self, numOfStreams):
		self.streams.extend([Driver.CommandQueue(context, profiling=True)] * numOfStreams)


	def borrow(self, numOfStreams):
		if len(self.streams) < numOfStreams:
			self.reserve(numOfStreams - len(self.streams))

		borrowed = []
		for _ in range(numOfStreams):
			borrowed.append(self.streams.pop())

		return borrowed


	def give(self, streams):
		self.streams.extend(streams)


class GPUArrayFills:
	def __init__(self):
		self.pool = {}


	def __call__(self, shape, dtype, constant=1.0):
		tup = (shape, dtype, constant)
		fills = self.pool.get(tup, None)

		if fills is None:
			if constant == 1.0:
				fills = Driver.to_device(queue, np.ones(shape, dtype=dtype), allocator=memoryPool)
			elif constant == 0.0:
				fills = Driver.zeros(queue, shape, dtype=dtype, allocator=memoryPool)
			else:
				if isinstance(constant, tuple) or isinstance(constant, list) or isinstance(constant, range):
					fills = []
					for i in range(shape[0]):
						fills.extend([constant[i]] * shape[1])
					fills = np.array(fills, dtype=dtype)
				else:
					fills = np.full(shape, constant, dtype=dtype)

				fills = Driver.to_device(queue, fills, allocator=memoryPool)

			self.pool[tup] = fills

		return fills


	def clear(self):
		self.pool.clear()


def clearCaches():
	global memoryPool, globalFills

	if memoryPool is not None:
		memoryPool.stopHolding()

	if globalFills is not None:
		globalFills.clear()


def autoinit():
	global context
	if context is not None:
		return

	def finishUp():
		clearCaches()

	import atexit
	atexit.register(finishUp)

	platforms = Driver.get_platforms()
	if len(platforms) == 0:
		raise RuntimeError("No OpenCL enabled platform found")

	amd = "AMD Accelerated Parallel Processing"

	global platform
	for vendor in [amd]:
		platform = next((pl for pl in platforms if pl.name == vendor), None)
		if platform is not None:
			break

	if platform is None:
		platform = platforms[0]

	devices = platform.get_devices(type=Driver.device_type.GPU)
	if len(devices) == 0:
		raise RuntimeError("No OpenCL enabled devices found")

	if Config.deviceIdx >= len(devices):
		raise RuntimeError("Invalid OpenCL config device index")

	global device
	device = devices[Config.deviceIdx]

	context = Driver.Context([device])

	global queue
	queue = Driver.CommandQueue(context, profiling=True)

	addinfo = " %sCU" % device.max_compute_units if platform.name == amd else ""
	print("[%s]: Using device #%s (%s%s)" % (Config.libname, Config.deviceIdx, device.name, addinfo))

	if Config.systemLog:
		clVersion = device.version
		print("[%s]: Created OpenCL context (Using driver version: %s)" % (Config.libname, clVersion))

	global memoryPool
	memoryPool = Driver.MemoryPool(context)

	global globalRng
	globalRng = Random.PhiloxGenerator(context, queue)

	if Config.systemLog:
		print("[%s]: Created global rng (%s)" % (Config.libname, globalRng.__class__.__name__))

	global streamManager
	streamManager = StreamManager()

	global globalFills
	globalFills = GPUArrayFills()


if context is None and (multiprocessing.current_process().name == "MainProcess" or Config.allowMultiContext):
	autoinit()


def decoratedEltWiseCall(func):
	def handleEltWiseCall(self, *args, **kwargs):
		self.context = context
		self.queue = queue

		if "stream" in kwargs:
			kwargs["queue"] = kwargs["stream"]
			del kwargs["stream"]

		return func(self, *args, **kwargs)

	return handleEltWiseCall


def decoratedSet(func):
	def gpuOrCpuSet(*args, **kwargs):
		self, ary = args[:2]

		if isinstance(ary, Driver.Array):
			copy(self, ary)
		else:
			func(self, ary, *args[2:], **kwargs)

	return gpuOrCpuSet


Templates.ElementwiseKernel.__call__ = decoratedEltWiseCall(Templates.ElementwiseKernel.__call__)
Driver.Array.set = decoratedSet(Driver.Array.set)
Driver.CommandQueue.synchronize = Driver.CommandQueue.finish


def iadd(self, other):
	assert self.dtype == other.dtype and self.dtype == np.float32
	assert self.shape == other.shape

	addKer(np.float32)(self, self, 1.0, other, 1.0)
	return self


def add(self, other):
	assert self.dtype == other.dtype and self.dtype == np.float32
	assert self.shape == other.shape

	result = Driver.empty_like(self)
	addKer(np.float32)(result, self, 1.0, other, 1.0)
	return result


def isub(self, other):
	assert self.dtype == other.dtype and self.dtype == np.float32
	assert self.shape == other.shape

	addKer(np.float32)(self, self, 1.0, other, -1.0)
	return self


def sub(self, other):
	assert self.dtype == other.dtype and self.dtype == np.float32
	assert self.shape == other.shape

	result = Driver.empty_like(self)
	addKer(np.float32)(result, self, 1.0, other, -1.0)
	return result


driverFill = Driver.Array.fill


def decoratedFill(self, value):
	driverFill(self, value)
	return self


Driver.Array.__iadd__ = iadd
Driver.Array.__add__ = add
Driver.Array.__isub__ = isub
Driver.Array.__sub__ = sub
Driver.Array.fill = decoratedFill


def setupDebugAllocator():
	def fillEmpty(cq, shape, dtype, data=None, offset=0, allocator=None):
		buffer = Driver.Array(cq, shape, dtype, data, offset, allocator)

		import inspect, math
		if inspect.isclass(dtype):
			tp = dtype
		elif hasattr(dtype, "type"):
			tp = dtype.type
		else:
			raise RuntimeError()

		if issubclass(tp, np.floating):
			buffer.fill(math.nan)
		else:
			Driver.enqueue_fill_buffer_8b(cq, buffer.base_data, 0xcc, buffer.nbytes, buffer.offset)

		return buffer

	Driver.empty = fillEmpty


def fillUniform(data, minval=0.0, maxval=1.0, rng=globalRng):
	rng.fillUniform(data)
	linearKer(data.dtype)(data, data, maxval - minval, minval)


def fillNormal(data, mean=0.0, sigma=1.0, rng=globalRng):
	rng.fillNormal(data)
	linearKer(data.dtype)(data, data, sigma, mean)


def checkOffsets(*tensors, allowOffset=False):
	outTensors = []

	for tensor in tensors:
		if not allowOffset:
			assert tensor.offset == 0
		elif tensor.offset > 0:
			tensor = copy(None, tensor)

		outTensors.append(tensor)

	return outTensors


def copy(dest, source):
	if dest is None:
		dest = Driver.empty(queue, source.shape, dtype=source.dtype, allocator=memoryPool)

	else:
		assert source.shape == dest.shape
		assert source.dtype == dest.dtype

	Driver.enqueue_barrier(queue)
	Driver.enqueue_copy_1d(queue, dest.base_data, source.base_data, byte_count=min(source.nbytes, dest.nbytes),
						   dest_offset=dest.offset, src_offset=source.offset)

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


def concatenate(tup, axis, out=None):
	assert all(a.dtype == tup[0].dtype and
			   a.shape[:axis] + a.shape[axis+1:] == tup[0].shape[:axis] + tup[0].shape[axis+1:] for a in tup)

	concatDim = 0
	for a in tup:
		concatDim += a.shape[axis]

	shape = tup[0].shape[:axis] + (concatDim, ) + tup[0].shape[axis+1:]

	if out is None:
		out = Driver.empty(queue, shape, dtype=tup[0].dtype, allocator=memoryPool)
	else:
		assert out.shape == shape and out.dtype == tup[0].dtype

	out = out.reshape(int(np.prod(out.shape[:axis])), int(np.prod(out.shape[axis:])))

	stride = 0
	for i, a in enumerate(tup):
		a = a.reshape(int(np.prod(a.shape[:axis])), int(np.prod(a.shape[axis:])))

		Driver.enqueue_copy_3d(queue, out.base_data, a.base_data, dest_origin=(out.offset + stride, 0, 0),
							   src_origin=(a.offset, 0, 0), region=(a.strides[0], out.shape[0], 1),
							   dest_pitches=(out.strides[0], 0), src_pitches=(a.strides[0], 0))

		stride += a.strides[0]

	out = out.reshape(shape)

	return out


def split(ary, sections, axis):
	assert np.sum(sections) == ary.shape[axis]

	outs = []
	for sec in sections:
		shape = ary.shape[:axis] + (sec, ) + ary.shape[axis+1:]
		outs.append(Driver.empty(queue, shape, dtype=ary.dtype, allocator=memoryPool))

	ary = ary.reshape(int(np.prod(ary.shape[:axis])), int(np.prod(ary.shape[axis:])))

	stride = 0
	for i, out in enumerate(outs):
		out = out.reshape(int(np.prod(out.shape[:axis])), int(np.prod(out.shape[axis:])))

		Driver.enqueue_copy_3d(queue, out.base_data, ary.base_data, dest_origin=(out.offset, 0, 0),
							   src_origin=(ary.offset + stride, 0, 0), region=(out.strides[0], ary.shape[0], 1),
							   dest_pitches=(out.strides[0], 0), src_pitches=(ary.strides[0], 0))

		stride += out.strides[0]

	return outs


def tile(ary, repeats, axis):
	return concatenate([ary] * repeats, axis=axis)


def unittest():
	shareMemTest()
	memCopyTest()
	randomTest()


def shareMemTest():
	shMem = SharedArray()

	shMem.register((10, 10, 10), np.float32, "a")
	shMem.register((50, 1, 5), np.float32, "b")
	shMem.build()

	assert shMem["a"].shape == (10, 10, 10) and shMem["a"].dtype == np.float32
	assert shMem["b"].shape == (50, 1, 5) and shMem["b"].dtype == np.float32


def memCopyTest():
	shape = (4, 4, 4, 4)
	source = Driver.to_device(queue, np.random.randn(*shape).astype(np.float32))
	dest = Driver.empty(queue, shape, dtype=np.float32)

	copy(dest, source)
	assert np.allclose(dest.get(), source.get())

	a = Driver.to_device(queue, np.random.randn(7, 4, 4, 4).astype(np.float32))
	out = concatenate((source, a), axis=0)
	assert np.allclose(out.get(), np.concatenate((source.get(), a.get()), axis=0))

	a = Driver.to_device(queue, np.random.randn(4, 2, 4, 4).astype(np.float32))
	b = Driver.to_device(queue, np.random.randn(4, 1, 4, 4).astype(np.float32))
	out = concatenate((source, a, b), axis=1)
	assert np.allclose(out.get(), np.concatenate((source.get(), a.get(), b.get()), axis=1))

	a = Driver.to_device(queue, np.random.randn(4, 4, 5, 4).astype(np.float32))
	out = concatenate((a, source), axis=2)
	assert np.allclose(out.get(), np.concatenate((a.get(), source.get()), axis=2))

	a = Driver.to_device(queue, np.random.randn(4, 4, 4, 5).astype(np.float32))
	out = concatenate((a, source), axis=3)
	assert np.allclose(out.get(), np.concatenate((a.get(), source.get()), axis=3))

	outs = split(source, (2, 2), axis=0)
	assert all(np.allclose(out.get(), source.get()[2 * i:2 * (i + 1)]) for i, out in enumerate(outs))

	outs = split(source, (2, 2), axis=1)
	assert all(np.allclose(out.get(), source.get()[:, 2 * i:2 * (i + 1), :, :]) for i, out in enumerate(outs))

	outs = split(source, (2, 2), axis=2)
	assert all(np.allclose(out.get(), source.get()[:, :, 2 * i:2 * (i + 1), :]) for i, out in enumerate(outs))

	outs = split(source, (2, 2), axis=3)
	assert all(np.allclose(out.get(), source.get()[:, :, :, 2 * i:2 * (i + 1)]) for i, out in enumerate(outs))

	assert np.allclose(tile(b, 3, axis=1).get(), np.tile(b.get(), (1, 3, 1, 1)))


def randomTest():
	data = Driver.empty(queue, (100, ), dtype=np.float32)
	fillUniform(data, -1.0, 1.0)


if __name__ == "__main__":
	unittest()
