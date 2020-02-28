import ctypes

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool

from PuzzleLib.Converter.TensorRT import Driver


class CalibratorError(Exception):
	pass


class DataCalibrator(Driver.ICalibrator):
	def __init__(self, data, batchsize=100, cachename=None, dataname=None):
		super().__init__()

		self.data = data

		self.idx = 0
		self.offset = 0

		self.nbatches = (data.shape[0] + batchsize - 1) // batchsize
		self.batchsize = batchsize

		if cachename is not None:
			raise NotImplementedError()

		self.cachename = cachename
		self.dataname = dataname

		self.cache = None


	def getDataShape(self):
		return self.data.shape[1:]


	def getBatchSize(self):
		batchsize = min(self.data.shape[0] - self.offset, self.batchsize)
		return batchsize


	def getBatch(self, bindings, names):
		assert len(bindings) == 1

		if self.dataname is not None:
			assert names[0] == self.dataname

		batchsize = self.getBatchSize()
		if batchsize == 0:
			return False

		batch = self.data[self.offset:self.offset + batchsize]
		self.offset += batchsize

		batch = gpuarray.to_gpu(batch, allocator=memPool)

		ptr = ctypes.cast(int(bindings[0]), ctypes.POINTER(ctypes.c_void_p))
		ptr.contents.value = batch.ptr

		print("Sending batch #%s (%s out of %s)" % (self.idx, self.idx + 1, self.nbatches))
		self.idx += 1

		return True
