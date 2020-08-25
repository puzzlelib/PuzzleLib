import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.Dnn import BatchNormMode, batchNormNd, batchNormNdBackward

from PuzzleLib.Variable import Variable
from PuzzleLib.Modules.Module import ModuleError, Module


class BatchNorm(Module):
	def __init__(self, size, epsilon=1e-5, initFactor=1.0, minFactor=0.1, sscale=0.01, affine=True, name=None,
				 empty=False, inplace=False):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.inplace = inplace

		if inplace and Config.showWarnings:
			Config.getLogger().info("Warning: %s is using inplace flag", self)

		self.size = size
		self.epsilon = epsilon
		self.initFactor = initFactor
		self.minFactor = minFactor
		self.numOfProps = 0

		self.affine = affine

		self.scale, self.bias, self.mean, self.var = None, None, None, None
		self.savemean, self.saveinvvar, self.scalegrad, self.biasgrad = None, None, None, None

		if empty:
			return

		scale = np.random.normal(1.0, sscale if affine else 0.0, (1, size, 1, 1)).astype(np.float32)
		var = np.ones((1, size, 1, 1), dtype=np.float32)

		self.setVar("scale", Variable(gpuarray.to_gpu(scale)))
		self.setVar("bias", Variable(gpuarray.zeros((1, size, 1, 1), dtype=np.float32)))

		self.setAttr("mean", gpuarray.zeros((1, size, 1, 1), dtype=np.float32))
		self.setAttr("var", gpuarray.to_gpu(var))


	def updateData(self, data):
		indata = data.reshape(data.shape[0], self.size, 1, 1)

		if self.train:
			if self.inplace:
				raise ModuleError("%s: using inplace flag in train mode is prohibited" % self)

			self.numOfProps += 1
			factor = max(self.initFactor / self.numOfProps, self.minFactor)

			self.data, self.savemean, self.saveinvvar = batchNormNd(
				indata, self.scale, self.bias, self.mean, self.var, self.epsilon, factor, False,
				BatchNormMode.perActivation
			)

		else:
			self.data = batchNormNd(
				indata, self.scale, self.bias, self.mean, self.var, self.epsilon, 0, True,
				BatchNormMode.perActivation, out=indata if self.inplace else None
			)

		self.data = self.data.reshape(*data.shape)


	def updateGrad(self, grad):
		data = self.inData.reshape(self.inData.shape[0], self.size, 1, 1)
		outgrad = grad.reshape(grad.shape[0], self.size, 1, 1)

		tup = batchNormNdBackward(
			data, outgrad, self.scale, self.savemean, self.saveinvvar, self.epsilon, mode=BatchNormMode.perActivation
		)

		if self.affine:
			self.grad, self.scalegrad, self.biasgrad = tup
		else:
			self.grad, _, _ = tup

		self.grad = self.grad.reshape(*grad.shape)


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		if self.affine:
			Blas.addVectorToVector(
				self.scalegrad.ravel(), self.vars["scale"].grad.ravel(),
				out=self.vars["scale"].grad.ravel(), alpha=scale, beta=momentum
			)
			Blas.addVectorToVector(
				self.biasgrad.ravel(), self.vars["bias"].grad.ravel(),
				out=self.vars["bias"].grad.ravel(), alpha=scale, beta=momentum
			)


	def dataShapeFrom(self, shape):
		return shape


	def checkDataShape(self, shape):
		if len(shape) != 2:
			raise ModuleError("Data must be 2d matrix")

		if int(np.prod(shape[1])) != self.size:
			raise ModuleError("Expected %d data dimensions, %d were given" % (self.size, shape[1]))


	def gradShapeFrom(self, shape):
		return shape


	def checkGradShape(self, shape):
		if len(shape) != 2:
			raise ModuleError("Grad must be 2d matrix")

		if int(np.prod(shape[1])) != self.size:
			raise ModuleError("Expected %d grad dimensions, %d were given" % (self.size, shape[1]))


	def reset(self):
		super().reset()
		self.savemean, self.saveinvvar = None, None

		if self.affine:
			self.scalegrad, self.biasgrad = None, None


	def calcMode(self, T):
		if Config.backend == Config.Backend.cuda:
			if T not in {np.float16, np.float32}:
				raise ModuleError("Unsupported dtype %s" % T)

		elif T != np.float32:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	batchsize, insize = 16, 10
	data = gpuarray.to_gpu(np.random.randn(batchsize, insize).astype(np.float32))

	bn = BatchNorm(insize)
	bn(data)

	hostData = data.get()
	hostMean = np.mean(hostData, axis=0, keepdims=False)
	hostInvVar = 1.0 / np.sqrt(np.sum((hostData - hostMean[np.newaxis, :])**2, axis=0) / batchsize + 1e-5)

	hostScale = bn.scale.get().squeeze()
	hostBias = bn.bias.get().squeeze()

	hostNormData = (data.get() - hostMean) * hostInvVar
	hostOutData = hostNormData * hostScale + hostBias

	assert np.allclose(hostMean, bn.savemean.get().squeeze())
	assert np.allclose(hostInvVar, bn.saveinvvar.get().squeeze())
	assert np.allclose(hostOutData, bn.data.get().squeeze())

	grad = gpuarray.to_gpu(np.random.randn(batchsize, insize).astype(np.float32))
	bn.backward(grad)

	hostGrad = grad.get()

	hostBiasGrad = np.sum(hostGrad, axis=0)
	hostScaleGrad = np.sum(hostGrad * hostNormData, axis=0)
	hostMeanGrad = np.sum(hostGrad, axis=0) * hostScale * -hostInvVar
	hostVarGrad = np.sum(hostGrad * (hostData - hostMean[np.newaxis, :]), axis=0) * \
				  hostScale[np.newaxis, :] * -0.5 * hostInvVar[np.newaxis, :]**3

	hostInGrad = grad.get() * hostScale[np.newaxis, :] * hostInvVar[np.newaxis, :] + \
				 hostVarGrad * 2 / batchsize * (data.get() - hostMean) + hostMeanGrad / batchsize

	assert np.allclose(hostBiasGrad, bn.vars["bias"].grad.get().squeeze())
	assert np.allclose(hostScaleGrad, bn.vars["scale"].grad.get().squeeze())
	assert np.allclose(hostInGrad, bn.grad.get().squeeze())


if __name__ == "__main__":
	unittest()
