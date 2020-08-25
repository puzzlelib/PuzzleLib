import numpy as np

from PuzzleLib.Backend import gpuarray, Memory
from PuzzleLib.Modules.Module import ModuleError, Module


class MoveAxis(Module):
	def __init__(self, src, dst, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		if src == dst:
			raise ModuleError("Trivial axis move is treated as error")

		self.src, self.dst = src, dst


	def updateData(self, data):
		self.data = Memory.moveaxis(data, self.src, self.dst)


	def updateGrad(self, grad):
		self.grad = Memory.moveaxis(grad, self.dst, self.src)


	def checkDataShape(self, shape):
		ln = max(self.src, self.dst)

		if len(shape) - 1 < ln:
			raise ModuleError("Data dimension needs to be at least %d, (data has %d)" % (ln + 1, len(shape)))


	def checkGradShape(self, shape):
		ln = max(self.src, self.dst)

		if len(shape) - 1 < ln:
			raise ModuleError("Grad dimension needs to be at least %d, (grad has %d)" % (ln + 1, len(shape)))


	def dataShapeFrom(self, shape):
		if self.src < self.dst:
			return shape[:self.src] + shape[self.src + 1:self.dst + 1] + (shape[self.src], ) + shape[self.dst + 1:]
		else:
			return shape[:self.dst] + (shape[self.src], ) + shape[self.dst:self.src] + shape[self.src + 1:]


	def gradShapeFrom(self, shape):
		if self.src < self.dst:
			return shape[:self.src] + (shape[self.dst], ) + shape[self.src:self.dst] + shape[self.dst + 1:]
		else:
			return shape[:self.dst] + shape[self.dst + 1:self.src + 1] + (shape[self.dst], ) + shape[self.src + 1:]


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in gpuarray.dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, _ in gpuarray.dtypesSupported():
		moveAxisTest(dtype)


def moveAxisTest(dtype):
	shape = (10, 3, 5, 4, 2)

	for src in range(len(shape)):
		for dst in range(len(shape)):
			if src == dst:
				continue

			hostData = np.random.randn(*shape).astype(dtype)
			data = gpuarray.to_gpu(hostData)

			moveaxis = MoveAxis(src, dst)
			moveaxis.calcMode(dtype)

			moveaxis(data)

			hostOutData = np.moveaxis(hostData, source=src, destination=dst)
			assert np.allclose(hostOutData, moveaxis.data.get())

			hostGrad = np.random.randn(*moveaxis.data.shape).astype(dtype)
			grad = gpuarray.to_gpu(hostGrad)

			moveaxis.backward(grad)

			hostInGrad = np.moveaxis(hostGrad, source=dst, destination=src)

			assert moveaxis.grad.shape == data.shape
			assert np.allclose(hostInGrad, moveaxis.grad.get())


if __name__ == "__main__":
	unittest()
