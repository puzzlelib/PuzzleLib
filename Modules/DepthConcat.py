import numpy as np

from PuzzleLib.Backend import gpuarray, Memory
from PuzzleLib.Modules.Module import ModuleError, Module


class DepthConcat(Module):
	def __init__(self, name=None):
		super().__init__(name)
		self.movesData = True


	def updateData(self, data):
		self.data = Memory.depthConcat(data)


	def updateGrad(self, grad):
		self.grad = Memory.depthSplit(grad, self.inData)


	def checkDataShape(self, shapes):
		if not isinstance(shapes, list):
			raise ModuleError("Data must be list of tensors")

		for shape in shapes:
			if len(shape) != 4:
				raise ModuleError("Data must consist of 4d tensors")

			if shape[0] != shapes[0][0]:
				raise ModuleError("Inconsistency in batch size")


	def dataShapeFrom(self, shapes):
		depth, h, w = 0, 0, 0
		for shape in shapes:
			depth += shape[1]
			h, w = max(h, shape[2]), max(w, shape[3])

		return shapes[0][0], depth, h, w


	def checkGradShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Grad must be 4d tensor")

		depth, h, w = 0, 0, 0
		for data in self.inData:
			sh = data.shape

			depth += sh[1]
			h, w = max(h, sh[2]), max(w, sh[3])

		gradshape = (self.inData[0].shape[0], depth, h, w)
		if shape != gradshape:
			raise ModuleError("Bad grad shape (%s given, %s expected)" % (shape, gradshape))


	def gradShapeFrom(self, shape):
		shapes = [data.shape for data in self.inData]
		return shapes


def unittest():
	data1 = gpuarray.to_gpu(np.random.randn(3, 4, 2, 2).astype(np.float32))
	data2 = gpuarray.to_gpu(np.random.randn(3, 2, 6, 6).astype(np.float32))
	data3 = gpuarray.to_gpu(np.random.randn(3, 5, 4, 4).astype(np.float32))
	data4 = gpuarray.to_gpu(np.random.randn(3, 3, 5, 5).astype(np.float32))
	alldata = [data1, data2, data3, data4]

	concat = DepthConcat()
	concat(alldata)

	depth, h, w = 0, 0, 0
	for data in alldata:
		depth += data.shape[1]
		h, w = max(h, data.shape[2]), max(w, data.shape[3])

	hostOutData = np.zeros(shape=(data1.shape[0], depth, h, w), dtype=np.float32)

	hostOutData[:, :4, 2:4, 2:4] = data1.get()
	hostOutData[:, 4:6, :, :] = data2.get()
	hostOutData[:, 6:11, 1:5, 1:5] = data3.get()
	hostOutData[:, 11:, :5, :5] = data4.get()

	assert np.allclose(hostOutData, concat.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*hostOutData.shape).astype(np.float32))
	concat.backward(grad)

	hostInGrads = [np.empty(data.shape, dtype=np.float32) for data in alldata]

	hostInGrads[0] = grad.get()[:, :4, 2:4, 2:4]
	hostInGrads[1] = grad.get()[:, 4:6, :, :]
	hostInGrads[2] = grad.get()[:, 6:11, 1:5, 1:5]
	hostInGrads[3] = grad.get()[:, 11:, :5, :5]

	assert all(np.allclose(hostInGrad, concat.grad[i].get()) for i, hostInGrad in enumerate(hostInGrads))


if __name__ == "__main__":
	unittest()
