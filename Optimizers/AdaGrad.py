import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Kernels.ElementWise import adagradKer

from PuzzleLib.Optimizers.Optimizer import Optimizer, trainSimpleTest, trainHardTest


class AdaGrad(Optimizer):
	def __init__(self, learnRate=1e-3, epsilon=1e-8, nodeinfo=None):
		super().__init__(nodeinfo)

		self.epsilon = None

		self.setAttr("learnRate", learnRate)
		self.setAttr("epsilon", epsilon)


	def setupState(self, var):
		return {"h": gpuarray.zeros(var.data.shape, dtype=var.data.dtype)}


	def updateVar(self, var, state, stream=None):
		adagradKer(var.data.dtype)(
			var.data, var.grad, state["h"], self.learnRate * var.learnRate, self.epsilon, stream=stream
		)


def unittest():
	for dtype, atol in gpuarray.dtypesSupported():
		calcTest(dtype, atol)
		trainSimpleTest(AdaGrad, dtype, learnRate=1e-2)

		if Config.backend == Config.Backend.cuda:
			trainHardTest(AdaGrad, dtype, learnRate=1e-2)


def calcTest(dtype, atol):
	lr, epsilon = 0.01, 1e-8
	shape = (11, 13)

	hostW, hostDw = np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)
	hostH = (1.0 + np.random.randn(*shape)**2).astype(dtype)

	w, dw, h = gpuarray.to_gpu(hostW), gpuarray.to_gpu(hostDw), gpuarray.to_gpu(hostH)
	adagradKer(w.dtype)(w, dw, h, lr, epsilon)

	hostW, hostDw, hostH = hostW.astype(np.float32), hostDw.astype(np.float32), hostH.astype(np.float32)

	hostH += hostDw**2
	hostW += lr * hostDw / (np.sqrt(hostH) + epsilon)

	hostW, hostDw, hostH = hostW.astype(dtype), hostDw.astype(dtype), hostH.astype(dtype)

	assert np.allclose(hostH, h.get(), atol=atol)
	assert np.allclose(hostW, w.get(), atol=atol)


if __name__ == "__main__":
	unittest()
