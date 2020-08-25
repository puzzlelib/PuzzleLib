import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.Dnn import batchNormNd, batchNormNdBackward

from PuzzleLib.Variable import Variable
from PuzzleLib.Modules.Module import ModuleError, Module


class BatchNormND(Module):
	def __init__(self, nd, maps, epsilon=1e-5, initFactor=1.0, minFactor=0.1, sscale=0.01, affine=True, name=None,
				 empty=False, inplace=False):
		super().__init__(name)
		self.inplace = inplace

		if inplace and Config.showWarnings:
			Config.getLogger().info("Warning: %s is using inplace flag", self)

		self.maps = maps
		self.epsilon = epsilon
		self.initFactor = initFactor
		self.minFactor = minFactor
		self.numOfProps = 0

		self.affine = affine

		self.scale, self.bias, self.mean, self.var = None, None, None, None
		self.savemean, self.saveinvvar, self.scalegrad, self.biasgrad = None, None, None, None

		if empty:
			return

		shape = (1, maps) + self.repeat(1, nd)

		scale = np.random.normal(1.0, sscale if affine else 0.0, shape).astype(self.calctype)
		var = np.ones(shape, dtype=self.calctype)

		self.setVar("scale", Variable(gpuarray.to_gpu(scale)))
		self.setVar("bias", Variable(gpuarray.zeros(shape, dtype=self.calctype)))

		self.setAttr("mean", gpuarray.zeros(shape, dtype=self.calctype))
		self.setAttr("var", gpuarray.to_gpu(var))


	def updateData(self, data):
		if self.train:
			if self.inplace:
				raise ModuleError("%s: using inplace flag in train mode is prohibited" % self)

			self.numOfProps += 1
			factor = max(self.initFactor / self.numOfProps, self.minFactor)

			self.data, self.savemean, self.saveinvvar = batchNormNd(
				data, self.scale, self.bias, self.mean, self.var, self.epsilon, factor, False
			)
		else:
			self.data = batchNormNd(
				data, self.scale, self.bias, self.mean, self.var, self.epsilon, 0, True,
				out=data if self.inplace else None
			)


	def updateGrad(self, grad):
		tup = batchNormNdBackward(self.inData, grad, self.scale, self.savemean, self.saveinvvar, self.epsilon)

		if self.affine:
			self.grad, self.scalegrad, self.biasgrad = tup
		else:
			self.grad, _, _ = tup


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		if self.affine:
			Blas.addVectorToVector(
				self.scalegrad.ravel(), self.vars["scale"].grad.ravel(), out=self.vars["scale"].grad.ravel(),
				alpha=scale, beta=momentum
			)
			Blas.addVectorToVector(
				self.biasgrad.ravel(), self.vars["bias"].grad.ravel(), out=self.vars["bias"].grad.ravel(),
				alpha=scale, beta=momentum
			)


	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return shape


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
