import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import dtypesSupported
from PuzzleLib.Backend.Kernels.ElementWise import smorms3Ker

from PuzzleLib.Optimizers.Optimizer import Optimizer, trainSimpleTest, trainHardTest


class SMORMS3(Optimizer):
	def __init__(self, learnRate=1e-3, epsilon=1e-16, nodeinfo=None):
		super().__init__(nodeinfo)

		self.epsilon = None

		self.setAttr("learnRate", learnRate)
		self.setAttr("epsilon", epsilon)


	def setupState(self, var):
		return {
			"mem": gpuarray.to_gpu(np.ones(var.data.shape, dtype=np.float32)),
			"mg": gpuarray.zeros(var.data.shape, dtype=np.float32),
			"ms": gpuarray.zeros(var.data.shape, dtype=np.float32)
		}


	def updateVar(self, var, state, stream=None):
		smorms3Ker(var.data.dtype)(
			var.data, var.grad, state["mem"], state["mg"], state["ms"], self.learnRate * var.learnRate, self.epsilon,
			stream=stream
		)


def unittest():
	for dtype, atol in dtypesSupported():
		calcTest(dtype, atol)
		trainSimpleTest(SMORMS3, dtype, learnRate=1e-2)

		if Config.backend == Config.Backend.cuda:
			trainHardTest(SMORMS3, dtype, learnRate=1e-2)


def calcTest(dtype, atol):
	lr, epsilon = 1e-3, 1e-16
	shape = (11, 13)

	hostW, hostDw = np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)
	hostMem = (1.0 + np.random.randn(*shape)**2).astype(np.float32)
	hostMg, hostMs = np.random.randn(*shape).astype(np.float32), np.random.randn(*shape).astype(np.float32)**2

	w, dw = gpuarray.to_gpu(hostW), gpuarray.to_gpu(hostDw)
	mem, mg, ms = gpuarray.to_gpu(hostMem), gpuarray.to_gpu(hostMg), gpuarray.to_gpu(hostMs)

	smorms3Ker(w.dtype)(w, dw, mem, mg, ms, lr, epsilon)

	hostW, hostDw = hostW.astype(np.float32), hostDw.astype(np.float32)

	r = 1.0 / (1.0 + hostMem)
	hostMg = (1.0 - r) * hostMg + r * hostDw
	hostMs = (1.0 - r) * hostMs + r * hostDw**2
	x = hostMg**2 / (hostMs + epsilon)

	hostMem = 1.0 + hostMem * (1.0 - x)
	hostW += hostDw * np.minimum(lr, x) / (np.sqrt(hostMs) + epsilon)

	hostW, hostDw = hostW.astype(dtype), hostDw.astype(dtype)

	assert np.allclose(hostMem, mem.get(), atol=atol)
	assert np.allclose(hostMg, mg.get(), atol=atol)
	assert np.allclose(hostMs, ms.get(), atol=atol)
	assert np.allclose(hostW, w.get(), atol=atol)


if __name__ == "__main__":
	unittest()
