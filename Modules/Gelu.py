import math

import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.gpuarray import memoryPool as memPool
from PuzzleLib.Backend.Kernels.ElementWise import geluKer, geluDerKer

from PuzzleLib.Modules.Module import ModuleError, Module


class Gelu(Module):
	def __init__(self, inplace=False, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.inplace = inplace

		if inplace and Config.showWarnings:
			Config.getLogger().info("Warning: %s is using inplace flag", self)


	def updateData(self, data):
		self.data = data if self.inplace else gpuarray.empty(data.shape, dtype=data.dtype, allocator=memPool)
		geluKer(data.dtype)(self.data, data)


	def updateGrad(self, grad):
		self.grad = grad if self.inplace else gpuarray.empty(grad.shape, dtype=grad.dtype, allocator=memPool)
		geluDerKer(grad.dtype)(self.grad, grad, self.inData)


	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return shape


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in gpuarray.dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, atol in gpuarray.dtypesSupported():
		geluTest(dtype, atol)


def geluTest(dtype, atol):
	gelu = Gelu()
	gelu.calcMode(dtype)

	hostData = np.random.randn(11, 51).astype(dtype)

	data = gpuarray.to_gpu(hostData)
	gelu(data)

	erf = np.vectorize(math.erf)
	hostOutData = 0.5 * hostData * (1.0 + erf(hostData / math.sqrt(2)))

	assert np.allclose(hostOutData, gelu.data.get(), atol=atol)

	hostGrad = np.random.randn(*gelu.data.shape).astype(dtype)

	grad = gpuarray.to_gpu(hostGrad)
	gelu.backward(grad)

	hostInGrad = hostGrad * (0.5 * (1.0 + erf(hostData / math.sqrt(2))) +
				 hostData / math.sqrt(math.pi) * np.exp(-0.5 * hostData**2))
	assert np.allclose(hostInGrad, gelu.grad.get(), atol=atol)


if __name__ == "__main__":
	unittest()
