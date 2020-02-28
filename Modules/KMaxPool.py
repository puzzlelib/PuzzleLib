import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool

from PuzzleLib.Modules.Module import ModuleError, Module


class KMaxPool(Module):
	def __init__(self, topk, axis, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.topk = topk
		self.axis = axis
		self.indices = None


	@staticmethod
	def sliceAlongAxis(ary, axis, start, end):
		ary = np.rollaxis(ary, axis)[start:end]
		return np.rollaxis(ary, 0, axis + 1)


	@staticmethod
	def getIndicesSlice(ary, axis, indices):
		tup = ()

		for i in range(ary.ndim):
			if i == axis:
				tup += (indices, )
			else:
				shape = tuple([None] * i + [slice(None)] + [None] * (ary.ndim - 1 - i))
				r = np.arange(ary.shape[i])[shape]

				tup += (r, )

		return tup


	def sliceWithIndicesAlongAxis(self, ary, axis, indices):
		return ary[self.getIndicesSlice(ary, axis, indices)]


	def fillSliceWithIndicesAlongAxis(self, ary, axis, indices, data):
		ary[self.getIndicesSlice(ary, axis, indices)] = data


	def updateData(self, data):
		data = data.get()

		indices = np.argpartition(data, -self.topk, axis=self.axis)
		indices = self.sliceAlongAxis(indices, self.axis, -self.topk, None)

		topkData = self.sliceWithIndicesAlongAxis(data, self.axis, indices)

		topkIndices = np.argsort(topkData, axis=self.axis)
		topkData = self.sliceWithIndicesAlongAxis(topkData, self.axis, topkIndices)

		self.indices = self.sliceWithIndicesAlongAxis(indices, self.axis, topkIndices)
		self.data = gpuarray.to_gpu(topkData, allocator=memPool)


	def updateGrad(self, grad):
		grad = grad.get()

		ingrad = np.zeros(self.inData.shape, dtype=np.float32)
		self.fillSliceWithIndicesAlongAxis(ingrad, self.axis, self.indices, grad)

		self.grad = gpuarray.to_gpu(ingrad, allocator=memPool)


	def checkDataShape(self, shape):
		if self.axis >= len(shape):
			raise ModuleError("Data dimension needs to be at least %d, (data has %d)" % (self.axis + 1, len(shape)))

		if shape[self.axis] < self.topk:
			raise ModuleError("Data topk axis is too small (got %d, expected at least %d)" %
							  (shape[self.axis], self.topk))


	def checkGradShape(self, shape):
		if self.axis >= len(shape):
			raise ModuleError("Grad dimension needs to be at least %d, (grad has %d)" % (self.axis + 1, len(shape)))

		if shape[self.axis] != self.topk:
			raise ModuleError("Grad topk axis is wrong (got %d, expected exactly %d)" % (shape[self.axis], self.topk))


	def dataShapeFrom(self, shape):
		return shape[:self.axis] + (self.topk, ) + shape[self.axis + 1:]


	def gradShapeFrom(self, shape):
		return shape[:self.axis] + self.inData.shape[self.axis] + shape[self.axis + 1:]


def unittest():
	topk = 5
	axis = 2

	data = gpuarray.to_gpu(np.random.randn(32, 10, 16).astype(np.float32))

	kmaxpool = KMaxPool(topk=topk, axis=axis)
	kmaxpool(data)

	hostData = data.get()

	hostOutData = np.partition(hostData, -topk, axis=axis)[:, :, -topk:]
	hostIndices = np.argpartition(hostData, -topk, axis=axis)[:, :, -topk:]

	hostInIndices = np.argsort(hostOutData, axis=axis)

	tup = (np.arange(hostOutData.shape[0])[:,None,None], np.arange(hostOutData.shape[1])[None,:,None], hostInIndices)
	hostIndices = hostIndices[tup]
	hostOutData = hostOutData[tup]

	assert np.allclose(kmaxpool.data.get(), hostOutData)

	grad = gpuarray.to_gpu(np.random.randn(*data.shape[:axis], topk).astype(np.float32))
	kmaxpool.backward(grad)

	hostGrad = grad.get()
	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	tup = (np.arange(hostInGrad.shape[0])[:, None, None], np.arange(hostInGrad.shape[1])[None, :, None], hostIndices)
	hostInGrad[tup] = hostGrad

	assert np.allclose(hostInGrad, kmaxpool.grad.get())


if __name__ == "__main__":
	unittest()
