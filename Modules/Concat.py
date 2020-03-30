import numpy as np

from PuzzleLib.Backend import gpuarray, Utils
from PuzzleLib.Modules.Module import ModuleError, Module


class Concat(Module):
	def __init__(self, axis, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.axis = axis
		self.sections = None


	def updateData(self, data):
		self.sections = [d.shape[self.axis] for d in data]
		self.data = Utils.concatenate(data, axis=self.axis)


	def updateGrad(self, grad):
		self.grad = Utils.split(grad, self.sections, axis=self.axis)


	def checkDataShape(self, shapes):
		for i, shape in enumerate(shapes[1:]):
			if not shape[:self.axis] + shape[self.axis + 1:] == shapes[0][:self.axis] + shapes[0][self.axis + 1:]:
				raise ModuleError(
					"Shape %d is inconsistent with initial shape (checking %s, init is %s)" % (i, shape, shapes[0])
				)


	def dataShapeFrom(self, shapes):
		concatDim = 0
		for shape in shapes:
			concatDim += shape[self.axis]

		shape = shapes[0][:self.axis] + (concatDim, ) + shapes[0][self.axis + 1:]
		return shape


	def checkGradShape(self, shape):
		concatDim = 0
		for sec in self.sections:
			concatDim += sec

		gradShape = self.data.shape[:self.axis] + (concatDim, ) + self.data.shape[self.axis+1:]
		if gradShape != shape:
			raise ModuleError("Expected grad shape %s (given %s)" % (gradShape, shape))


	def gradShapeFrom(self, shape):
		shapes = []
		for sec in self.sections:
			shapes.append(shape[:self.axis] + (sec, ) + shape[self.axis + 1:])

		return shapes


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in Utils.dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	alongBatchAxis()
	alongDataAxis()


def alongBatchAxis():
	data = []
	for _ in range(3):
		data.append(gpuarray.to_gpu(np.random.randn(np.random.randint(low=5, high=10), 10, 5, 3).astype(np.float32)))

	concat = Concat(axis=0)
	concat(data)

	hostOutData = np.concatenate([d.get() for d in data], axis=0)
	assert np.allclose(hostOutData, concat.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*hostOutData.shape).astype(np.float32))
	concat.backward(grad)

	stride = 0
	hostInGrad = []
	for i in range(len(data)):
		hostInGrad.append(grad.get()[stride:stride + data[i].shape[0], :])
		stride += data[i].shape[0]

	assert all([np.allclose(hostInGrad[i], concat.grad[i].get()) for i in range(len(data))])


def alongDataAxis():
	data = []
	for _ in range(3):
		data.append(gpuarray.to_gpu(np.random.randn(10, np.random.randint(low=4, high=8), 4, 5).astype(np.float32)))

	concat = Concat(axis=1)
	concat(data)

	hostOutData = np.concatenate([d.get() for d in data], axis=1)
	assert np.allclose(hostOutData, concat.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*hostOutData.shape).astype(np.float32))
	concat.backward(grad)

	stride = 0
	hostInGrad = []
	for i in range(len(data)):
		hostInGrad.append(grad.get()[:, stride:stride + data[i].shape[1]])
		stride += data[i].shape[1]

	assert all([np.allclose(hostInGrad[i], concat.grad[i].get()) for i in range(len(data))])


if __name__ == "__main__":
	unittest()
