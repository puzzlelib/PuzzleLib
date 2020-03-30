import numpy as np

from PuzzleLib.Backend import gpuarray, Blas, Utils
from PuzzleLib.Backend.Utils import dtypesSupported

from PuzzleLib.Modules.Module import ModuleError, Module


class Tile(Module):
	def __init__(self, axis, times, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.axis = axis
		self.times = times


	def updateData(self, data):
		self.data = Utils.tile(data, self.times, axis=self.axis)


	def updateGrad(self, grad):
		sections = [grad.shape[self.axis] // self.times] * self.times
		ingrad = Utils.split(grad, sections, axis=self.axis)

		for i in range(1, len(ingrad)):
			Blas.toVectorAddVector(ingrad[0].ravel(), ingrad[i].ravel())

		self.grad = ingrad[0]


	def checkDataShape(self, shape):
		if len(shape) < self.axis + 1:
			raise ModuleError("Not enough dimensions in data shape (%s given, %s required)" % (len(shape), self.axis+1))


	def dataShapeFrom(self, shape):
		return shape[:self.axis] + (shape[self.axis] * self.times, ) + shape[self.axis + 1:]


	def checkGradShape(self, shape):
		if len(shape) < self.axis + 1:
			raise ModuleError("Not enough dimensions in grad shape (%s given, %s required)" % (len(shape), self.axis+1))

		if shape[self.axis] % self.times != 0:
			raise ModuleError("Dimension %s in grad shape must be divisible by %s" % (shape[self.axis], self.times))


	def gradShapeFrom(self, shape):
		return shape[:self.axis] + (shape[self.axis] // self.times, ) + shape[self.axis + 1:]


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, _ in dtypesSupported():
		alongBatchAxisTest(dtype)
		alongDataAxisTest(dtype)


def alongBatchAxisTest(dtype):
	hostData = np.random.randn(3, 4, 5).astype(dtype)
	data = gpuarray.to_gpu(hostData)

	axis, times = 0, 3

	tile = Tile(axis=axis, times=times)
	tile.calcMode(dtype)

	tile(data)

	hostOutData = np.concatenate([data.get()] * times, axis=axis)
	assert np.allclose(hostOutData, tile.data.get())

	hostGrad = np.random.randn(*hostOutData.shape).astype(dtype)
	grad = gpuarray.to_gpu(hostGrad)

	tile.backward(grad)

	hostInGrad = np.sum(hostGrad.reshape((-1, 3, 4, 5)), axis=axis)
	assert np.allclose(hostInGrad, tile.grad.get())


def alongDataAxisTest(dtype):
	hostData = np.random.randn(3, 4, 5).astype(dtype)
	data = gpuarray.to_gpu(hostData)

	axis, times = 1, 4

	tile = Tile(axis=axis, times=times)
	tile.calcMode(dtype)

	tile(data)

	hostOutData = np.concatenate([data.get()] * times, axis=axis)
	assert np.allclose(hostOutData, tile.data.get())

	hostGrad = np.random.randn(*hostOutData.shape).astype(dtype)
	grad = gpuarray.to_gpu(hostGrad)

	tile.backward(grad)

	hostInGrad = np.sum(hostGrad.reshape((3, -1, 4, 5)), axis=axis)
	assert np.allclose(hostInGrad, tile.grad.get())


if __name__ == "__main__":
	unittest()
