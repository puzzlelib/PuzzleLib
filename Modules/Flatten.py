import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Modules.Module import Module


class Flatten(Module):
	def __init__(self, name=None):
		super().__init__(name)

		self.movesData = True
		self.movesGrad = True

		self.inshape = None


	def updateData(self, data):
		self.inshape = data.shape
		self.data = data.reshape(data.shape[0], int(np.prod(data.shape[1:])))


	def updateGrad(self, grad):
		self.grad = grad.reshape(self.inshape)


	def dataShapeFrom(self, shape):
		return shape[0], int(np.prod(shape[1:]))


	def gradShapeFrom(self, shape):
		return (shape[0], ) + self.inshape[1:]


	def calcMode(self, T):
		self.calctype = T


def unittest():
	data = gpuarray.to_gpu(np.random.randn(10, 10, 10, 10).astype(np.float32))

	flatten = Flatten()
	flatten(data)

	shape = (10, 1000)
	assert flatten.data.shape == shape

	grad = gpuarray.to_gpu(np.random.randn(*flatten.data.shape).astype(np.float32))
	flatten.backward(grad)

	assert flatten.grad.shape == data.shape


if __name__ == "__main__":
	unittest()
