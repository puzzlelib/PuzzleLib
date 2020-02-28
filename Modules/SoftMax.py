import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Dnn.Basic import softmaxNd, softmaxNdBackward

from PuzzleLib.Modules.Module import ModuleError, Module


class SoftMax(Module):
	def __init__(self, name=None):
		super().__init__(name)
		self.gradUsesOutData = True


	def updateData(self, data):
		if data.ndim == 2:
			indata = data.reshape(*data.shape, 1, 1)
		else:
			indata = data

		self.data = softmaxNd(indata)

		if data.ndim == 2:
			self.data = self.data.reshape(*self.data.shape[:2])


	def updateGrad(self, grad):
		if grad.ndim == 2:
			ingrad = grad.reshape(*grad.shape, 1, 1)
		else:
			ingrad = grad

		if self.data.ndim == 2:
			indata = self.data.reshape(*self.data.shape, 1, 1)
		else:
			indata = self.data

		self.grad = softmaxNdBackward(indata, ingrad)

		if grad.ndim == 2:
			self.grad = self.grad.reshape(*self.grad.shape[:2])


	def checkDataShape(self, shape):
		if len(shape) != 4 and len(shape) != 2:
			raise ModuleError("Data must be 4d or 2d tensor")


	def dataShapeFrom(self, shape):
		return shape


	def checkGradShape(self, shape):
		if len(shape) != 4 and len(shape) != 2:
			raise ModuleError("Grad must be 4d or 2d tensor")


	def gradShapeFrom(self, shape):
		return shape


	def calcMode(self, T):
		if Config.backend == Config.Backend.cuda:
			if T not in {np.float16, np.float32}:
				raise ModuleError("Unsupported dtype %s" % T)

		elif T != np.float32:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	batchsize, maps = 2, 3
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, 1, 1).astype(np.float32))

	softmax = SoftMax()
	softmax(data)

	def softMaxForward(w):
		e = np.exp(w - np.amax(w))
		p = e / np.sum(e)
		return p

	hostData = data.get().reshape(batchsize, maps).astype(np.float32)
	hostOutData = np.vstack([softMaxForward(hostData[i]) for i in range(batchsize)])
	assert np.allclose(hostOutData, softmax.data.get().reshape(batchsize, maps).astype(np.float32))

	grad = gpuarray.to_gpu(np.random.randn(batchsize, maps, 1, 1).astype(np.float32))
	softmax.backward(grad)

	hostGrad = grad.get().reshape(batchsize, maps).astype(np.float32)
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
