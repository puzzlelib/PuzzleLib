import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool

from PuzzleLib.Modules.Module import ModuleError, Module


class Slice(Module):
	def __init__(self, slc=None, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.slc = slc
		self.inshape = None


	def __getitem__(self, slc):
		if not isinstance(slc, tuple):
			slc = (slc, )

		self.slc = slc
		return self


	def updateData(self, data):
		self.inshape = data.shape
		self.data = data[self.slc].copy()


	def updateGrad(self, grad):
		self.grad = gpuarray.zeros(self.inshape, dtype=np.float32, allocator=memPool)
		self.grad[self.slc] = grad


	def dataShapeFrom(self, shape):
		if self.slc is None:
			raise ModuleError("Slice parameter is not initialized")

		outshape = [None] * len(shape)
		for i, dim in enumerate(shape):
			slc = self.slc[i]
			start, stop, step = slc.indices(dim)

			outshape[i] = (stop - start) // step

		return tuple(outshape)


	def checkDataShape(self, shape):
		if self.slc is None:
			raise ModuleError("Slice parameter is not initialized")

		if len(shape) < len(self.slc):
			raise ModuleError("Expected at least %d data dimensions, %d were given" % (len(self.slc), len(shape)))


	def gradShapeFrom(self, shape):
		return self.inshape


	def checkGradShape(self, shape):
		if shape != self.data.shape:
			raise ModuleError("Grad shape %s is inconsistent with output data shape %s" % (shape, self.data.shape))


def unittest():
	data = gpuarray.to_gpu(np.random.randn(3, 4, 5, 6).astype(np.float32))

	slc = Slice()[:, :, 1:-1, 1:-1]
	slc(data)

	assert slc.dataShapeFrom(data.shape) == slc.data.shape
	assert np.allclose(slc.data.get(), data.get()[slc.slc])

	grad = gpuarray.to_gpu(np.random.randn(*slc.data.shape).astype(np.float32))
	slc.backward(grad)

	assert slc.gradShapeFrom(grad.shape) == data.shape
	assert np.allclose(slc.grad.get()[slc.slc], grad.get())


if __name__ == "__main__":
	unittest()
