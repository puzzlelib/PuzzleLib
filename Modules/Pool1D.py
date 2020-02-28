import numpy as np

from PuzzleLib import Config
from PuzzleLib.Modules.Module import ModuleError, Module


class Pool1D(Module):
	def __init__(self, size=2, stride=2, pad=0, name=None):
		super().__init__(name)
		self.gradUsesOutData = True

		self.size = (1, size)
		self.stride = (1, stride)
		self.pad = (0, pad)

		self.workspace = None


	def dataShapeFrom(self, shape):
		batchsize, maps, insize = shape

		_, size = self.size
		_, pad = self.pad
		_, stride = self.stride

		outsize = (insize + 2 * pad - size) // stride + 1

		return batchsize, maps, outsize


	def checkDataShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Data must be 3d tensor")

		_, _, insize = shape
		if insize + 2 * self.pad[1] < self.size[1]:
			raise ModuleError("Data maps size is too small (got %d, expected at least %d)" %
							  (insize + 2 * self.pad[1], self.size[1]))


	def gradShapeFrom(self, shape):
		batchsize, maps, outsize = shape

		_, size = self.size
		_, pad = self.pad
		_, stride = self.stride

		insize = (outsize - 1) * stride - 2 * pad + size

		return batchsize, maps, insize


	def checkGradShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Grad must be 3d tensor")


	def updateData(self, data):
		raise NotImplementedError()


	def updateGrad(self, grad):
		raise NotImplementedError()


	def reset(self):
		super().reset()
		self.workspace = None


	def calcMode(self, T):
		if Config.backend == Config.Backend.cuda:
			if T not in {np.float16, np.float32}:
				raise ModuleError("Unsupported dtype %s" % T)

		elif T != np.float32:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T
