import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Modules.Module import ModuleError, Module


class Glue(Module):
	def __init__(self, modules=None, fwdGlue=None, bwdGlue=None, fwdShapeGlue=None, bwdShapeGlue=None, name=None):
		super().__init__(name)

		if modules is not None and not isinstance(modules, dict):
			raise ModuleError("Modules object must be non-empty dictionary")

		self.modules = modules

		self.fwdGlue = fwdGlue
		self.bwdGlue = bwdGlue

		self.fwdShapeGlue = fwdShapeGlue
		self.bwdShapeGlue = bwdShapeGlue


	def updateData(self, data):
		self.data = self.fwdGlue(data, self.modules)


	def updateGrad(self, grad):
		self.grad = self.bwdGlue(grad, self.modules)


	def dataShapeFrom(self, shape):
		if self.fwdShapeGlue is not None:
			return self.fwdShapeGlue(shape)
		else:
			raise ModuleError("Forward shape glue hook is not installed")


	def gradShapeFrom(self, shape):
		if self.bwdShapeGlue is not None:
			return self.bwdShapeGlue(shape)
		else:
			raise ModuleError("Backward shape glue hook is not installed")


def unittest():
	data1 = gpuarray.to_gpu(np.random.randn(10, 2, 3, 3).astype(np.float32))
	data2 = gpuarray.to_gpu(np.random.randn(10, 2, 3, 3).astype(np.float32))
	data3 = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))

	def fwdGlue(data, modules):
		dat1, dat2, dat3 = data
		split = modules["split"]
		out1, out2 = split(data3)

		return [dat1 + dat2, out1, out2]

	def bwdGlue(grad, modules):
		gr1, gr2, gr3 = grad
		split = modules["split"]
		split.backward([gr2, gr3])

		return [gr1, gr1, split.grad]

	from PuzzleLib.Modules.Split import Split
	glue = Glue(fwdGlue=fwdGlue, bwdGlue=bwdGlue, modules={"split": Split(axis=1, sections=(5, 5))})
	glue([data1, data2, data3])

	grad1 = gpuarray.to_gpu(np.random.randn(*glue.data[0].shape).astype(np.float32))
	grad2 = gpuarray.to_gpu(np.random.randn(*glue.data[1].shape).astype(np.float32))
	grad3 = gpuarray.to_gpu(np.random.randn(*glue.data[2].shape).astype(np.float32))

	glue.backward([grad1, grad2, grad3])


if __name__ == "__main__":
	unittest()
