from PuzzleLib.Backend import gpuarray
from PuzzleLib.Modules.Module import ModuleError, Module


class Pool2D(Module):
	def __init__(self, size=2, stride=2, pad=0, name=None):
		super().__init__(name)
		self.gradUsesOutData = True

		self.size = self.repeat(size, 2)
		self.stride = self.repeat(stride, 2)
		self.pad = self.repeat(pad, 2)

		self.workspace = None


	def dataShapeFrom(self, shape):
		batchsize, maps, inh, inw = shape

		hsize, wsize = self.size
		hpad, wpad = self.pad
		hstride, wstride = self.stride

		outh = (inh + 2 * hpad - hsize) // hstride + 1
		outw = (inw + 2 * wpad - wsize) // wstride + 1

		return batchsize, maps, outh, outw


	def checkDataShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Data must be 4d tensor")

		_, _, inh, inw = shape
		if inh + 2 * self.pad[0] < self.size[0]:
			raise ModuleError("Data maps height is too small (got %d, expected at least %d)" %
							  (inh + 2 * self.pad[0], self.size[0]))

		if inw + 2 * self.pad[1] < self.size[1]:
			raise ModuleError("Data maps width is too small (got %d, expected at least %d)" %
							  (inw + 2 * self.pad[1], self.size[1]))


	def gradShapeFrom(self, shape):
		batchsize, maps, outh, outw = shape

		hsize, wsize = self.size
		hpad, wpad = self.pad
		hstride, wstride = self.stride

		inh = (outh - 1) * hstride - 2 * hpad + hsize
		inw = (outw - 1) * wstride - 2 * wpad + wsize

		return batchsize, maps, inh, inw


	def checkGradShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Grad must be 4d tensor")


	def updateData(self, data):
		raise NotImplementedError()


	def updateGrad(self, grad):
		raise NotImplementedError()


	def reset(self):
		super().reset()
		self.workspace = None


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in gpuarray.dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T
