import ctypes

import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.gpuarray import memoryPool as memPool
from PuzzleLib.Converter.TensorRT import Driver


class CalibratorError(Exception):
	pass


class DataCalibrator(Driver.ICalibrator):
	def __init__(self, data, batchsize=100, cachename=None):
		super().__init__("" if cachename is None else cachename)

		if data is None:
			if cachename is None:
				raise CalibratorError("Invalid calibration cache file")

			self.nbatches = 0

		else:
			if data.shape[0] % batchsize != 0:
				raise CalibratorError("TensorRT calibration engine requires data size to be divisible by batch size")

			if data.dtype != np.float32:
				raise CalibratorError("Invalid data type")

			self.nbatches = data.shape[0] // batchsize

		self.data = data
		self.idx = 0

		self.batchsize = batchsize
		self.batch = None


	def getDataShape(self):
		return self.data.shape[1:]


	def getBatchSize(self):
		return self.batchsize


	def getBatch(self, bindings, names):
		assert len(bindings) == 1 and len(names) == 1

		if self.idx >= self.nbatches:
			return False

		self.batch = gpuarray.to_gpu(
			self.data[self.idx * self.batchsize:(self.idx + 1) * self.batchsize], allocator=memPool
		)

		ptr = ctypes.cast(bindings[0], ctypes.POINTER(ctypes.c_void_p))
		ptr.contents.value = self.batch.ptr

		print("Sending batch #%s out of %s" % (self.idx + 1, self.nbatches))
		self.idx += 1

		return True
