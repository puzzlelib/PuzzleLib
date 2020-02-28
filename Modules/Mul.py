import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import copy, memoryPool as memPool
from PuzzleLib.Backend.Kernels.ElementWise import mulKer

from PuzzleLib.Modules.Module import ModuleError, Module


class Mul(Module):
	def updateData(self, data):
		self.data = gpuarray.empty(data[0].shape, dtype=np.float32, allocator=memPool)
		self.data.fill(1.0)

		for dat in data:
			mulKer(dat.dtype)(self.data, dat, self.data)


	def updateGrad(self, grad):
		self.grad = []
		for i in range(len(self.inData)):
			ingrad = copy(None, grad)

			for k in range(len(self.inData)):
				if k != i:
					mulKer(ingrad.dtype)(ingrad, self.inData[k], ingrad)

			self.grad.append(ingrad)


	def checkDataShape(self, shapes):
		for shape in shapes:
			if shape != shapes[0]:
				raise ModuleError("Shape %s is not equal to initial shape %s" % (shape, shapes[0]))


	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return [shape] * len(self.inData)


def unittest():
	hostData1 = np.random.randn(2, 5, 5).astype(np.float32)
	hostData2 = np.random.randn(*hostData1.shape).astype(np.float32)

	data1, data2 = gpuarray.to_gpu(hostData1), gpuarray.to_gpu(hostData2)

	mul = Mul()
	mul([data1, data2])

	assert np.allclose(mul.data.get(), hostData1 * hostData2)

	hostGrad = np.random.randn(*mul.data.shape).astype(np.float32)

	grad = gpuarray.to_gpu(hostGrad)
	mul.backward(grad)

	assert np.allclose(mul.grad[0].get(), hostGrad * hostData2)
	assert np.allclose(mul.grad[1].get(), hostGrad * hostData1)


if __name__ == "__main__":
	unittest()
