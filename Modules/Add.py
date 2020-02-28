import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.Utils import dtypesSupported, memoryPool as memPool

from PuzzleLib.Modules.Module import ModuleError, Module


class Add(Module):
	def __init__(self, name=None):
		super().__init__(name)
		self.movesGrad = True


	def updateData(self, data):
		firstdata = data[0]

		self.data = gpuarray.empty(firstdata.shape, dtype=firstdata.dtype, allocator=memPool)
		self.data.fill(0)

		for dat in data:
			Blas.toVectorAddVector(self.data.ravel(), dat.ravel())


	def updateGrad(self, grad):
		self.grad = [grad] * len(self.inData)


	def checkDataShape(self, shapes):
		for shape in shapes:
			if shape != shapes[0]:
				raise ModuleError("Shape %s is not equal to initial shape %s" % (shape, shapes[0]))


	def dataShapeFrom(self, shape):
		return shape[0]


	def gradShapeFrom(self, shape):
		return [shape] * len(self.inData)


	def calcMode(self, T):
		if Config.backend == Config.Backend.cuda:
			if T not in {np.float16, np.float32}:
				raise ModuleError("Unsupported dtype %s" % T)

		elif T != np.float32:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, _ in dtypesSupported():
		addTest(dtype)


def addTest(dtype):
	hostData1 = np.random.randn(2, 5, 5).astype(dtype)
	hostData2 = np.random.randn(*hostData1.shape).astype(dtype)

	data1, data2 = gpuarray.to_gpu(hostData1), gpuarray.to_gpu(hostData2)

	add = Add()
	add.calcMode(dtype)

	add([data1, data2])
	assert np.allclose(hostData1 + hostData2, add.data.get())


if __name__ == "__main__":
	unittest()
