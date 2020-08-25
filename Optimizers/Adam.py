import math

import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Kernels.ElementWise import adamKer

from PuzzleLib.Optimizers.Optimizer import Optimizer, trainSimpleTest, trainHardTest


class Adam(Optimizer):
	def __init__(self, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, nodeinfo=None):
		super().__init__(nodeinfo)

		self.alpha = None
		self.beta1 = None
		self.beta2 = None
		self.epsilon = None

		self.setAttr("alpha", alpha)
		self.setAttr("beta1", beta1)
		self.setAttr("beta2", beta2)
		self.setAttr("epsilon", epsilon)


	def setupState(self, var):
		return {
			"mg": gpuarray.zeros(var.data.shape, dtype=np.float32),
			"ms": gpuarray.zeros(var.data.shape, dtype=np.float32)
		}


	def updateVar(self, var, state, stream=None):
		fix1, fix2 = 1.0 - self.beta1**self.t, 1.0 - self.beta2**self.t
		self.learnRate = self.alpha * math.sqrt(fix2) / fix1

		fix1, fix2 = 1.0 - self.beta1, 1.0 - self.beta2
		adamKer(var.data.dtype)(
			var.data, var.grad, state["mg"], state["ms"], self.learnRate * var.learnRate, fix1, fix2, self.epsilon,
			stream=stream
		)


def unittest():
	for dtype, atol in gpuarray.dtypesSupported():
		calcTest(dtype, atol)
		trainSimpleTest(Adam, dtype, alpha=1e-2)

		if Config.backend == Config.Backend.cuda:
			trainHardTest(Adam, dtype, alpha=1e-2)


def calcTest(dtype, atol):
	alpha, beta1, beta2, epsilon = 0.01, 0.9, 0.999, 1e-8
	shape = (11, 13)

	hostW, hostDw = np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)
	hostMs, hostMg = (1.0 + np.random.randn(*shape)**2).astype(np.float32), np.random.randn(*shape).astype(np.float32)

	w, dw = gpuarray.to_gpu(hostW), gpuarray.to_gpu(hostDw)
	ms, mg = gpuarray.to_gpu(hostMs), gpuarray.to_gpu(hostMg)

	fix1, fix2 = 1.0 - beta1, 1.0 - beta2
	lr = alpha * math.sqrt(fix2) / fix1

	fix1, fix2 = 1.0 - beta1, 1.0 - beta2
	adamKer(w.dtype)(w, dw, mg, ms, lr, fix1, fix2, epsilon)

	hostW, hostDw = hostW.astype(np.float32), hostDw.astype(np.float32)

	hostMg = (1 - fix1) * hostMg + fix1 * hostDw
	hostMs = (1 - fix2) * hostMs + fix2 * hostDw**2
	hostW += lr * hostMg / (np.sqrt(hostMs) + epsilon)

	hostW, hostDw = hostW.astype(dtype), hostDw.astype(dtype)

	assert np.allclose(hostMg, mg.get(), atol=atol)
	assert np.allclose(hostMs, ms.get(), atol=atol)
	assert np.allclose(hostW, w.get(), atol=atol)


if __name__ == "__main__":
	unittest()
