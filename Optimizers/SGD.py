import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import dtypesSupported
from PuzzleLib.Backend.Kernels.ElementWise import toVectorAddVectorKer

from PuzzleLib.Optimizers.Optimizer import Optimizer, trainSimpleTest, trainHardTest


class SGD(Optimizer):
	def __init__(self, learnRate=1e-3, nodeinfo=None):
		super().__init__(nodeinfo)
		self.setAttr("learnRate", learnRate)


	def updateVar(self, var, state, stream=None):
		toVectorAddVectorKer(var.data.dtype)(var.data, var.grad, self.learnRate * var.learnRate, stream=stream)


def unittest():
	for dtype, atol in dtypesSupported():
		calcTest(dtype, atol)
		trainSimpleTest(SGD, dtype, learnRate=1e-1)

		if Config.backend == Config.Backend.cuda:
			trainHardTest(SGD, dtype, learnRate=1e-1)


def calcTest(dtype, atol):
	lr = 0.01
	shape = (11, 13)

	hostW, hostDw = np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)

	w, dw = gpuarray.to_gpu(hostW), gpuarray.to_gpu(hostDw)
	toVectorAddVectorKer(w.dtype)(w, dw, lr)

	hostW, hostDw = hostW.astype(np.float32), hostDw.astype(np.float32)

	hostW += lr * hostDw
	hostW, hostDw = hostW.astype(dtype), hostDw.astype(dtype)

	assert np.allclose(hostW, w.get(), atol=atol)


if __name__ == "__main__":
	unittest()
