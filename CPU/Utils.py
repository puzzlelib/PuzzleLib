import time, platform
import numpy as np

from PuzzleLib import Config
from PuzzleLib.CPU.CPUArray import CPUArray


def getDeviceName():
	return platform.processor()


def autoinit():
	Config.getLogger().info("Using device #%s (%s)", Config.deviceIdx, getDeviceName())


if Config.shouldInit():
	autoinit()


def memoize(fn):
	cache = {}

	def memoizer(*args):
		obj = cache.get(args, None)
		if obj is not None:
			return obj

		obj = fn(*args)
		cache[args] = obj

		return obj

	return memoizer


def dtypesSupported():
	return [(np.float32, 1e-5)]


class SharedArray:
	def __init__(self, dtype=np.float32):
		self.regs = []

		self.mem = None
		self.dtype = dtype

		self.blocks = {}
		self.ary = None


	def register(self, shape, dtype, name):
		self.regs.append((shape, dtype, name))


	def build(self):
		nbytes = 0
		for reg in self.regs:
			shape, dtype, _ = reg
			assert dtype == self.dtype
			nbytes += int(np.prod(shape) * dtype(0).itemsize)

		self.mem = CPUArray.empty((nbytes, ), np.uint8)
		offset = 0

		for shape, dtype, name in self.regs:
			regbytes = int(np.prod(shape) * dtype(0).itemsize)
			assert offset + regbytes <= self.mem.size

			self.blocks[name] = self.mem[offset:offset + regbytes].view(dtype).reshape(shape)
			offset += regbytes

		self.regs.clear()
		self.ary = self.mem.view(dtype=self.dtype)


	def __getitem__(self, item):
		return self.blocks[item]


def timeKernel(func, args, kwargs=None, looplength=1000, log=True, logname=None, normalize=False, hotpass=True):
	if kwargs is None:
		kwargs = {}

	if hotpass:
		func(*args, **kwargs)

	hostStart = time.time()

	for _ in range(looplength):
		func(*args, **kwargs)

	hostEnd = time.time()
	hostsecs = hostEnd - hostStart

	if logname is None:
		if hasattr(func, "__name__"):
			logname = "%s.%s" % (func.__module__, func.__name__)
		else:
			logname = "%s.%s" % (func.__module__ , func.__class__.__name__)

	if normalize:
		hostsecs /= looplength

	if log:
		print("%s host time: %s secs" % (logname, hostsecs))

	return hostsecs


def unittest():
	shareMemTest()


def shareMemTest():
	shMem = SharedArray()

	shMem.register((10, 10, 10), np.float32, "a")
	shMem.register((50, 1, 5), np.float32, "b")
	shMem.build()

	assert shMem["a"].shape == (10, 10, 10) and shMem["a"].dtype == np.float32
	assert shMem["b"].shape == (50, 1, 5) and shMem["b"].dtype == np.float32


if __name__ == "__main__":
	unittest()
