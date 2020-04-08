import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import dtypesSupported
from PuzzleLib.Backend.Dnn.Basic import softmaxNd, softmaxNdBackward

from PuzzleLib.Modules.Module import ModuleError, Module


class SoftMax(Module):
	def __init__(self, name=None):
		super().__init__(name)
		self.gradUsesOutData = True


	def updateData(self, data):
		shape = data.shape
		ndim = max(0, 4 - len(shape))

		data = data.reshape(shape + tuple(1 for _ in range(ndim)))
		self.data = softmaxNd(data).reshape(shape)


	def updateGrad(self, grad):
		shape = grad.shape
		ndim = max(0, 4 - len(shape))

		grad = grad.reshape(shape + tuple(1 for _ in range(ndim)))
		data = self.data.reshape(shape + tuple(1 for _ in range(ndim)))

		self.grad = softmaxNdBackward(data, grad).reshape(shape)


	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return shape


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	batchsize, maps = 2, 3

	hostData = np.random.randn(batchsize, maps, 1).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	softmax = SoftMax()
	softmax(data)

	def softMaxForward(w):
		e = np.exp(w - np.amax(w))
		p = e / np.sum(e)
		return p

	hostData = hostData.reshape(batchsize, maps).astype(np.float32)

	hostOutData = np.vstack([softMaxForward(hostData[i]) for i in range(batchsize)])
	assert np.allclose(hostOutData, softmax.data.get().reshape(batchsize, maps).astype(np.float32))

	hostGrad = np.random.randn(batchsize, maps, 1, 1).astype(np.float32)
	grad = gpuarray.to_gpu(hostGrad)

	softmax.backward(grad)
	hostGrad = hostGrad.reshape(batchsize, maps).astype(np.float32)

	def softMaxBackward(outdata, gr):
		ingrad = np.zeros(outdata.shape, dtype=np.float32)
		for i in range(ingrad.shape[0]):
			ingrad[i] += outdata[i] * gr[i]

			for j in range(outdata.shape[0]):
				ingrad[i] -= outdata[i] * outdata[j] * gr[j]
		return ingrad

	hostInGrad = np.vstack([softMaxBackward(hostOutData[i], hostGrad[i]) for i in range(batchsize)])
	assert np.allclose(hostInGrad, softmax.grad.get().reshape(batchsize, maps).astype(np.float32))


if __name__ == "__main__":
	unittest()
