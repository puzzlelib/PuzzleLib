import numpy as np

from PuzzleLib.Backend import gpuarray, Memory
from PuzzleLib.Modules.Module import ModuleError, Module


class Transpose(Module):
	def __init__(self, axes=None, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.axes = axes

		if axes is None:
			self.invaxes = None
		else:
			self.invaxes = [0] * len(axes)
			for i, axis in enumerate(axes):
				self.invaxes[axis] = i


	def updateData(self, data):
		self.data = Memory.transpose(data, self.axes)


	def updateGrad(self, grad):
		self.grad = Memory.transpose(grad, self.invaxes)


	def checkDataShape(self, shape):
		if self.axes is not None and len(shape) != len(self.axes):
			raise ModuleError("Data dimension needs to be %d, (data has %d)" % (len(self.axes), len(shape)))


	def checkGradShape(self, shape):
		if self.axes is not None and len(shape) != len(self.axes):
			raise ModuleError("Grad dimension needs to be %d, (grad has %d)" % (len(self.axes), len(shape)))


	def dataShapeFrom(self, shape):
		return tuple(shape[axis] for axis in self.axes)


	def gradShapeFrom(self, shape):
		return tuple(shape[axis] for axis in self.invaxes)


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in gpuarray.dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, _ in gpuarray.dtypesSupported():
		transposeTest(dtype)


def transposeTest(dtype):
	shape = (10, 3, 5, 4, 2)
	axes = (2, 4, 1, 3, 0)

	hostData = np.random.randn(*shape).astype(dtype)
	data = gpuarray.to_gpu(hostData)

	transpose = Transpose(axes)
	transpose.calcMode(dtype)

	transpose(data)

	hostOutData = np.transpose(hostData, axes=axes)
	assert np.allclose(hostOutData, transpose.data.get())

	hostGrad = np.random.randn(*transpose.data.shape).astype(dtype)
	grad = gpuarray.to_gpu(hostGrad)

	transpose.backward(grad)

	hostInGrad = np.transpose(hostGrad, axes=transpose.invaxes)
	assert np.allclose(hostInGrad, transpose.grad.get())


if __name__ == "__main__":
	unittest()
