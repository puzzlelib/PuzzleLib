import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Modules.Module import ModuleError, Module


class Split(Module):
	def __init__(self, axis, sections, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.sections = sections
		self.axis = axis


	def updateData(self, data):
		self.data = gpuarray.split(data, self.sections, self.axis)


	def updateGrad(self, grad):
		self.grad = gpuarray.concatenate(grad, self.axis)


	def dataShapeFrom(self, shape):
		shapes = []
		for sec in self.sections:
			shapes.append(shape[:self.axis] + (sec, ) + shape[self.axis + 1:])

		return shapes


	def gradShapeFrom(self, shapes):
		concatDim = 0
		for shape in shapes:
			concatDim += shape[self.axis]

		return shapes[0][:self.axis] + (concatDim, ) + shapes[0][self.axis + 1:]


	def checkDataShape(self, shape):
		if len(shape) < self.axis:
			raise ModuleError("Not enough dims in data (%d were given, need at least %d)" % (len(shape), self.axis))

		concatDim = 0
		for sec in self.sections:
			concatDim += sec

		if concatDim != shape[self.axis]:
			raise ModuleError(
				"Data shape %s is inconsistent with given sections %s"
				"(expected size %d on axis %d, %d was given)" %
				(shape, self.sections, concatDim, self.axis, shape[self.axis])
			)


	def checkGradShape(self, shapes):
		for i, shape in enumerate(shapes):
			if shape != self.data[i].shape:
				raise ModuleError(
					"Expected grad shape %s on %d place (%s was given)" % (self.data[i].shape, i + 1, shape)
				)


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in gpuarray.dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, _ in gpuarray.dtypesSupported():
		splitTest(dtype)


def splitTest(dtype):
	batchsize, groups, size = 5, 3, 4

	hostData = np.random.randn(batchsize, groups, size).astype(dtype)
	data = gpuarray.to_gpu(hostData)

	split = Split(axis=2, sections=(3, 1))
	split.calcMode(dtype)

	split(data)

	hostOutData = np.split(hostData, [split.sections[0]], axis=split.axis)
	assert all(np.allclose(hostOutData[i], split.data[i].get()) for i in range(len(hostOutData)))

	hostGrad = [np.random.randn(*split.data[i].shape).astype(dtype) for i in range(len(split.data))]
	grad = [gpuarray.to_gpu(gr) for gr in hostGrad]

	split.backward(grad)

	hostInGrad = np.concatenate(hostGrad, axis=split.axis)
	assert np.allclose(hostInGrad, split.grad.get())


if __name__ == "__main__":
	unittest()
