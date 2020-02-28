import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray, Memory
from PuzzleLib.Backend.Utils import dtypesSupported

from PuzzleLib.Modules.Module import ModuleError, Module


class SwapAxes(Module):
	def __init__(self, axis1, axis2, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.axis1, self.axis2 = (axis2, axis1) if axis1 > axis2 else (axis1, axis2)


	def updateData(self, data):
		self.data = Memory.swapaxes(data, self.axis1, self.axis2)


	def updateGrad(self, grad):
		self.grad = Memory.swapaxes(grad, self.axis1, self.axis2)


	def checkDataShape(self, shape):
		if len(shape) - 1 < self.axis2:
			raise ModuleError("Data dimension needs to be at least %d, (data has %d)" % (self.axis2 + 1, len(shape)))


	def checkGradShape(self, shape):
		if len(shape) - 1 < self.axis2:
			raise ModuleError("Grad dimension needs to be at least %d, (grad has %d)" % (self.axis2 + 1, len(shape)))


	def dataShapeFrom(self, shape):
		return shape[:self.axis1] + (shape[self.axis2], ) + shape[self.axis1 + 1:self.axis2] + \
			   (shape[self.axis1], ) + shape[self.axis2 + 1:]


	def gradShapeFrom(self, shape):
		return shape[:self.axis1] + (shape[self.axis2], ) + shape[self.axis1 + 1:self.axis2] + \
			   (shape[self.axis1], ) + shape[self.axis2 + 1:]


	def calcMode(self, T):
		if Config.backend == Config.Backend.cuda:
			if T not in {np.float16, np.float32}:
				raise ModuleError("Unsupported dtype %s" % T)

		elif T != np.float32:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, _ in dtypesSupported():
		swapAxesTest(dtype)


def swapAxesTest(dtype):
	shape = (10, 3, 5, 4, 2)

	for axis1 in range(len(shape)):
		for axis2 in range(axis1 + 1, len(shape)):
			hostData = np.random.randn(*shape).astype(dtype)
			data = gpuarray.to_gpu(hostData)

			swapaxes = SwapAxes(axis1, axis2)
			swapaxes.calcMode(dtype)

			swapaxes(data)

			hostOutData = np.swapaxes(hostData, axis1=axis1, axis2=axis2)
			assert np.allclose(hostOutData, swapaxes.data.get())

			hostGrad = np.random.randn(*swapaxes.data.shape).astype(dtype)
			grad = gpuarray.to_gpu(hostGrad)

			swapaxes.backward(grad)

			hostInGrad = np.swapaxes(hostGrad, axis1=axis2, axis2=axis1)

			assert swapaxes.grad.shape == data.shape
			assert np.allclose(hostInGrad, swapaxes.grad.get())


if __name__ == "__main__":
	unittest()
