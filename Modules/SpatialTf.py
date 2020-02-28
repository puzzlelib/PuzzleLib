import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Dnn.SpatialTf import spatialTf, spatialTfBackward

from PuzzleLib.Modules.Module import ModuleError, Module


class SpatialTf(Module):
	def __init__(self, shape=None, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.shape = shape
		self.grid = None


	def updateData(self, data):
		data, transform = data

		if self.train:
			self.data, self.grid = spatialTf(data, transform, outshape=self.shape, getGrid=True)
		else:
			self.data = spatialTf(data, transform, outshape=self.shape, getGrid=False)


	def updateGrad(self, grad):
		data, _ = self.inData
		self.grad = spatialTfBackward(grad, data, self.grid)


	def checkDataShape(self, shapes):
		dshape, tshape = shapes

		if len(tshape) != 3 or tshape[1:] != (2, 3):
			raise ModuleError("Bad transform shape (%s was given)" % tshape)

		if len(dshape) != 4:
			raise ModuleError("Data must be 4d tensor")

		if tshape[0] != dshape[0]:
			raise ModuleError("Inconsistency in transform and data batch size (%d in transform vs %d in data)" %
							  (tshape[0], dshape[0]))


	def checkGradShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Grad must be 4d tensor")

		if self.shape is not None:
			if self.shape != shape[1:]:
				raise ModuleError("Bad grad shape (was given %s, expected %s)" % (shape[1:], self.shape))
		else:
			if self.inData[0].shape != shape:
				raise ModuleError("Bad grad shape (was given %s, expected %s)" % (shape, self.inData[0].shape))


	def dataShapeFrom(self, shapes):
		dshape, tshape = shapes
		return (dshape[0], ) + self.shape if self.shape is not None else dshape


	def gradShapeFrom(self, shape):
		return (shape[0], ) + self.inData[0].shape[1:], (shape[0], 2, 3)


	def reset(self):
		super().reset()
		self.grid = None


def unittest():
	batchsize, maps, inh, inw = 1, 1, 4, 4
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, inh, inw).astype(np.float32))

	transform = gpuarray.to_gpu(
		np.tile(np.array([[1.0, 0.0, 0.001], [0, 1.0, 0.001]], dtype=np.float32), reps=(batchsize, 1, 1))
	)

	spatialtf = SpatialTf()
	spatialtf([data, transform])

	grad = gpuarray.to_gpu(np.random.randn(*spatialtf.data.shape).astype(np.float32))
	spatialtf.backward(grad)


if __name__ == "__main__":
	unittest()
