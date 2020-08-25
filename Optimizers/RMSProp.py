import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Kernels.ElementWise import rmspropKer

from PuzzleLib.Optimizers.Optimizer import Optimizer, trainSimpleTest, trainHardTest


class RMSProp(Optimizer):
	def __init__(self, learnRate=1e-3, factor=0.9, epsilon=1e-5, nodeinfo=None):
		super().__init__(nodeinfo)

		self.factor = None
		self.epsilon = None

		self.setAttr("learnRate", learnRate)
		self.setAttr("factor", factor)
		self.setAttr("epsilon", epsilon)


	def setupState(self, var):
		return {"ms": gpuarray.zeros(var.data.shape, dtype=var.data.dtype)}


	def updateVar(self, var, state, stream=None):
		rmspropKer(var.data.dtype)(
			var.data, var.grad, state["ms"], self.learnRate * var.learnRate, self.factor, self.epsilon, stream=stream
		)


def unittest():
	for dtype, atol in gpuarray.dtypesSupported():
		calcTest(dtype, atol)
		trainSimpleTest(RMSProp, dtype, learnRate=1e-2)

		if Config.backend == Config.Backend.cuda:
			trainHardTest(RMSProp, dtype, learnRate=1e-2)


def calcTest(dtype, atol):
	lr, factor, epsilon = 0.01, 0.9, 1e-5
	shape = (11, 13)

	hostW, hostDw = np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)
	hostMs = (1.0 + np.random.randn(*shape)**2).astype(dtype)

	w, dw, ms = gpuarray.to_gpu(hostW), gpuarray.to_gpu(hostDw), gpuarray.to_gpu(hostMs)
	rmspropKer(w.dtype)(w, dw, ms, lr, factor, epsilon)

	hostW, hostDw, hostMs = hostW.astype(np.float32), hostDw.astype(np.float32), hostMs.astype(np.float32)

	hostMs = factor * hostMs + (1 - factor) * hostDw**2
	hostW += lr * hostDw / (np.sqrt(hostMs) + epsilon)

	hostW, hostDw, hostMs = hostW.astype(dtype), hostDw.astype(dtype), hostMs.astype(dtype)

	assert np.allclose(hostMs, ms.get(), atol=atol)
	assert np.allclose(hostW, w.get(), atol=atol)


if __name__ == "__main__":
	unittest()
