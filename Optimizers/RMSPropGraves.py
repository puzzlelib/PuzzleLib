import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Kernels.ElementWise import rmspropGravesKer

from PuzzleLib.Optimizers.Optimizer import Optimizer, trainSimpleTest, trainHardTest


class RMSPropGraves(Optimizer):
	def __init__(self, learnRate=1e-4, alpha=0.95, momRate=0.9, epsilon=1e-4, nodeinfo=None):
		super().__init__(nodeinfo)

		self.alpha = None
		self.momRate = None
		self.epsilon = None

		self.setAttr("learnRate", learnRate)
		self.setAttr("alpha", alpha)
		self.setAttr("momRate", momRate)
		self.setAttr("epsilon", epsilon)


	def setupState(self, var):
		return {
			"mg": gpuarray.zeros(var.data.shape, dtype=var.data.dtype),
			"ms": gpuarray.zeros(var.data.shape, dtype=var.data.dtype),
			"delta": gpuarray.zeros(var.data.shape, dtype=var.data.dtype)
		}


	def updateVar(self, var, state, stream=None):
		rmspropGravesKer(var.data.dtype)(
			var.data, var.grad, state["mg"], state["ms"], state["delta"], self.learnRate * var.learnRate, self.alpha,
			self.momRate * var.momRate, self.epsilon, stream=stream
		)


def unittest():
	for dtype, atol in gpuarray.dtypesSupported():
		calcTest(dtype, atol)
		trainSimpleTest(RMSPropGraves, dtype, learnRate=1e-2)

		if Config.backend == Config.Backend.cuda:
			trainHardTest(RMSPropGraves, dtype, learnRate=1e-2)


def calcTest(dtype, atol):
	lr, alpha, mr, epsilon = 0.01, 0.95, 0.9, 10.0
	shape = (11, 13)

	hostW, hostDw = np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)
	hostMs, hostMg = (5.0 + np.random.randn(*shape)**2).astype(dtype), np.random.randn(*shape).astype(dtype)
	hostDelta = np.random.randn(*shape).astype(dtype)

	w, dw = gpuarray.to_gpu(hostW), gpuarray.to_gpu(hostDw)
	ms, mg, delta = gpuarray.to_gpu(hostMs), gpuarray.to_gpu(hostMg), gpuarray.to_gpu(hostDelta)

	rmspropGravesKer(w.dtype)(w, dw, mg, ms, delta, lr, alpha, mr, epsilon)

	hostW, hostDw = hostW.astype(np.float32), hostDw.astype(np.float32)
	hostMs, hostMg, hostDelta = hostMs.astype(np.float32), hostMg.astype(np.float32), hostDelta.astype(np.float32)

	hostMg = alpha * hostMg + (1 - alpha) * hostDw
	hostMs = alpha * hostMs + (1 - alpha) * hostDw**2
	hostDelta = mr * hostDelta + lr * hostDw / np.sqrt(hostMs - hostMg**2 + epsilon)
	hostW += hostDelta

	hostW, hostDw = hostW.astype(dtype), hostDw.astype(dtype)
	hostMs, hostMg, hostDelta = hostMs.astype(dtype), hostMg.astype(dtype), hostDelta.astype(dtype)

	assert np.allclose(hostMg, mg.get(), atol=atol)
	assert np.allclose(hostMs, ms.get(), atol=atol)
	assert np.allclose(hostDelta, delta.get(), atol=atol)
	assert np.allclose(hostW, w.get(), atol=atol)


if __name__ == "__main__":
	unittest()
