import numpy as np

from PuzzleLib.Config import libname
from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError, Module


class Reshape(Module):
	def __init__(self, shape, showWarnings=True, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.showWarnings = showWarnings

		self.movesData = True
		self.movesGrad = True

		self.shape = shape
		self.inshape = None

		self.copyIdx = tuple(idx for idx, value in enumerate(shape) if value == 0)


	def updateData(self, data):
		self.inshape = data.shape
		modShape = self.copyAxis(self.shape, self.inshape)
		self.data = data.reshape(modShape)

		if self.showWarnings:
			if self.data.shape[0] != self.inshape[0]:
				print("[%s] Warning: %s changed data batch axis size (was given %s, reshaped to %s)" %
					  (libname, self, data.shape, self.data.shape))


	def updateGrad(self, grad):
		self.grad = grad.reshape(self.inshape)

		if self.showWarnings:
			if self.grad.shape[0] != self.inshape[0]:
				print("[%s] Warning: %s changed grad batch axis size (was given %s, reshaped to %s)" %
					  (libname, self, grad.shape, self.grad.shape))


	def copyAxis(self, shape, mask):
		return tuple(mask[idx] if idx in self.copyIdx else value for idx, value in enumerate(shape))


	def checkDataShape(self, shape):
		modShape = self.copyAxis(self.shape, shape)
		try:
			idx = modShape.index(-1)

		except ValueError:
			if int(np.prod(shape)) != int(np.prod(modShape)):
				raise ModuleError("Data shape %s is inconsistent with reshape %s" % (shape, modShape))

			return

		if int(np.prod(shape)) % int(np.prod(modShape[:idx] + modShape[idx + 1:])) != 0:
			raise ModuleError("Data shape %s is inconsistent with reshape %s" % (shape, modShape))


	def checkGradShape(self, shape):
		if int(np.prod(shape)) != int(np.prod(self.inshape)):
			raise ModuleError("Grad shape %s is inconsistent with reshape %s" % (shape, self.inshape))


	def dataShapeFrom(self, shape):
		modShape = self.copyAxis(self.shape, shape)

		try:
			idx = self.shape.index(-1)
			dim = int(np.prod(shape)) // int(np.prod(modShape[:idx]) * np.prod(modShape[idx + 1:]))

			return modShape[:idx] + (dim, ) + modShape[idx + 1:]

		except ValueError:
			return modShape


	def gradShapeFrom(self, shape):
		return self.inshape


	def calcMode(self, T):
		self.calctype = T


def unittest():
	shapes = [
		[(10, 10, 10, 10), (10, -1, 100), (10, 10, 100)],
		[(1, 4, 7, 7), (0, 2, -1, 0), (1, 2, 14, 7)]
	]

	for inshape, shape, targetShape in shapes:
		data = gpuarray.to_gpu(np.random.randn(*inshape).astype(np.float32))

		reshape = Reshape(shape)

		reshape(data)
		assert reshape.data.shape == targetShape

		grad = gpuarray.to_gpu(np.random.randn(*reshape.data.shape).astype(np.float32))

		reshape.backward(grad)
		assert reshape.grad.shape == data.shape

		assert reshape.dataShapeFrom(data.shape) == targetShape and targetShape == reshape.data.shape
		assert reshape.gradShapeFrom(grad.shape) == data.shape


if __name__ == "__main__":
	unittest()
