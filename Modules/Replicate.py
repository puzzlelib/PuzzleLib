import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.Utils import dtypesSupported, memoryPool as memPool

from PuzzleLib.Modules.Module import ModuleError, Module


class Replicate(Module):
	def __init__(self, times, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.movesData = True
		self.times = times


	def updateData(self, data):
		self.data = [data] * self.times


	def updateGrad(self, grad):
		firstgrad = grad[0]

		self.grad = gpuarray.empty(firstgrad.shape, dtype=firstgrad.dtype, allocator=memPool)
		self.grad.fill(0)

		for gr in grad:
			Blas.toVectorAddVector(self.grad.ravel(), gr.ravel())


	def dataShapeFrom(self, shape):
		return [shape] * self.times


	def gradShapeFrom(self, shape):
		return shape[0]


	def calcMode(self, T):
		if Config.backend == Config.Backend.cuda:
			if T not in {np.float16, np.float32}:
				raise ModuleError("Unsupported dtype %s" % T)

		elif T != np.float32:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, _ in dtypesSupported():
		replicateTest(dtype)


def replicateTest(dtype):
	hostData = np.random.randn(10, 10, 3, 3).astype(dtype)
	data = gpuarray.to_gpu(hostData)

	times = 3

	repl = Replicate(times)
	repl.calcMode(dtype)

	repl(data)

	assert len(repl.data) == times

	hostGrad = [np.random.randn(10, 10, 3, 3).astype(dtype) for _ in range(times)]
	grad = [gpuarray.to_gpu(gr) for gr in hostGrad]

	repl.backward(grad)

	hostInGrad = np.zeros(grad[0].shape, dtype=dtype)
	for i in range(times):
		hostInGrad += hostGrad[i]

	assert np.allclose(hostInGrad, repl.grad.get())


if __name__ == "__main__":
	unittest()
