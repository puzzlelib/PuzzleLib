import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Modules.Module import Module


class Identity(Module):
	def __init__(self, name=None):
		super().__init__(name)

		self.movesData = True
		self.movesGrad = True


	def updateData(self, data):
		self.data = data


	def updateGrad(self, grad):
		self.grad = grad


	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return shape


	def calcMode(self, T):
		self.calctype = T


def unittest():
	data = gpuarray.to_gpu(np.random.normal(0.0, 0.01, (10, 3, 40, 40)).astype(np.float32))

	identity = Identity()
	identity(data)

	assert np.allclose(data.get(), identity.data.get())


if __name__ == "__main__":
	unittest()
