import platform, multiprocessing
import numpy as np

from PuzzleLib import Config
from PuzzleLib.CPU.CPUArray import CPUArray


def autoinit():
	print("[%s]: Using device #%s (%s)" % (Config.libname, Config.deviceIdx, platform.processor()))


if multiprocessing.current_process().name == "MainProcess" or Config.allowMultiContext:
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
