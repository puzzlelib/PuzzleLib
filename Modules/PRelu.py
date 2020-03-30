import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.Kernels.PRelu import prelu, preluBackwardData, preluBackwardParams

from PuzzleLib.Variable import Variable
from PuzzleLib.Modules.Module import ModuleError, Module


class PRelu(Module):
	def __init__(self, maps, inplace=False, sharedMaps=False, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.sharedMaps = sharedMaps

		self.inplace = inplace
		if inplace and Config.showWarnings:
			print("[%s] Warning: %s is using inplace flag" % (Config.libname, self))

		shape = (1, ) if sharedMaps else (maps, )
		slopes = gpuarray.to_gpu(np.full(shape, 0.25, dtype=np.float32))

		self.slopes = None
		self.setVar("slopes", Variable(slopes))


	def updateData(self, data):
		self.data = prelu(data, self.slopes, self.inplace, self.sharedMaps)


	def updateGrad(self, grad):
		if self.inplace:
			raise ModuleError("%s: using inplace flag while calculating gradient is prohibited" % self)

		self.grad = preluBackwardData(grad, self.slopes, self.inData, self.sharedMaps)


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		if self.inplace:
			raise ModuleError("%s: using inplace flag while calculating gradient is prohibited" % self)

		slopegrad = preluBackwardParams(self.inData, grad, self.sharedMaps)
		Blas.addVectorToVector(
			slopegrad, self.vars["slopes"].grad, out=self.vars["slopes"].grad, alpha=scale, beta=momentum
		)


	def dataShapeFrom(self, shape):
		return shape


	def checkDataShape(self, shape):
		if len(shape) < 2:
			raise ModuleError("Data tensor dimension must be at least 2")

		if not self.sharedMaps and shape[1] != self.slopes.shape[0]:
			raise ModuleError("Data tensor has %s maps (expected %s)" % (shape[1], self.slopes.shape[0]))


	def gradShapeFrom(self, shape):
		return shape


	def checkGradShape(self, shape):
		if shape != self.inData.shape:
			raise ModuleError("Grad tensor has shape %s (expected %s)" % (shape, self.inData.shape))


def unittest():
	preluTest()


def preluTest():
	batchsize, maps, h, w = 5, 4, 6, 6

	hostData = np.random.randn(batchsize, maps, h, w).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	preluMod = PRelu(maps)
	preluMod(data)

	hostSlopes = preluMod.slopes.get()
	hostOutData = np.empty(preluMod.data.shape, dtype=np.float32)

	for c in range(maps):
		hostOutData[:, c] = ((hostData[:, c] > 0.0) + (hostData[:, c] <= 0.0) * hostSlopes[c]) * hostData[:, c]

	assert np.allclose(hostOutData, preluMod.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*preluMod.data.shape).astype(np.float32))
	preluMod.backward(grad)

	hostGrad = grad.get()
	hostInGrad = np.empty(preluMod.grad.shape, dtype=np.float32)

	for c in range(maps):
		hostInGrad[:, c] = hostGrad[:, c] * ((hostData[:, c] > 0.0) + (hostData[:, c] <= 0.0) * hostSlopes[c])

	assert np.allclose(hostInGrad, preluMod.grad.get())

	hostSlopeGrad = np.empty(preluMod.vars["slopes"].grad.shape, dtype=np.float32)

	for c in range(maps):
		hostSlopeGrad[c] = np.sum(hostGrad[:, c] * hostData[:, c] * (hostData[:, c] <= 0.0))

	assert np.allclose(hostSlopeGrad, preluMod.vars["slopes"].grad.get())


if __name__ == "__main__":
	unittest()
