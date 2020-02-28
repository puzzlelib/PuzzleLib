import numpy as np

from PuzzleLib import Config
from PuzzleLib.Modules.Module import ModuleError, Module


class Pool3D(Module):
	def __init__(self, size=2, stride=2, pad=0, name=None):
		super().__init__(name)
		self.gradUsesOutData = True

		self.size = self.repeat(size, 3)
		self.stride = self.repeat(stride, 3)
		self.pad = self.repeat(pad, 3)

		self.workspace = None


	def dataShapeFrom(self, shape):
		batchsize, maps, ind, inh, inw = shape

		dsize, hsize, wsize = self.size
		dpad, hpad, wpad = self.pad
		dstride, hstride, wstride = self.stride

		outd = (ind + 2 * dpad - dsize) // dstride + 1
		outh = (inh + 2 * hpad - hsize) // hstride + 1
		outw = (inw + 2 * wpad - wsize) // wstride + 1

		return batchsize, maps, outd, outh, outw


	def checkDataShape(self, shape):
		if len(shape) != 5:
			raise ModuleError("Data must be 5d tensor")

		_, _, ind, inh, inw = shape
		if ind + 2 * self.pad[0] < self.size[0]:
			raise ModuleError("Data cube time is too small (got %d, expected at least %d)" %
							  (ind + 2 * self.pad[0], self.size[0]))

		if inh + 2 * self.pad[1] < self.size[1]:
			raise ModuleError("Data cube height is too small (got %d, expected at least %d)" %
							  (inh + 2 * self.pad[1], self.size[1]))

		if inw + 2 * self.pad[2] < self.size[2]:
			raise ModuleError("Data cube width is too small (got %d, expected at least %d)" %
							  (inw + 2 * self.pad[2], self.size[2]))


	def gradShapeFrom(self, shape):
		batchsize, maps, outd, outh, outw = shape

		dsize, hsize, wsize = self.size
		dpad, hpad, wpad = self.pad
		dstride, hstride, wstride = self.stride

		ind = (outd - 1) * dstride - 2 * dpad + dsize
		inh = (outh - 1) * hstride - 2 * hpad + hsize
		inw = (outw - 1) * wstride - 2 * wpad + wsize

		return batchsize, maps, ind, inh, inw


	def checkGradShape(self, shape):
		if len(shape) != 5:
			raise ModuleError("Grad must be 5d tensor")


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
