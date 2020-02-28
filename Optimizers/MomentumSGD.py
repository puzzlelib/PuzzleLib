import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import dtypesSupported
from PuzzleLib.Backend.Kernels.ElementWise import classicMomSGDKer

from PuzzleLib.Optimizers.Optimizer import trainSimpleTest, trainHardTest
from PuzzleLib.Optimizers.SGD import SGD


class MomentumSGD(SGD):
	def __init__(self, learnRate=1e-3, momRate=0.9, nodeinfo=None):
		super().__init__(learnRate, nodeinfo)

		self.momRate = None
		self.setAttr("momRate", momRate)


	def setupState(self, var):
		return {"mom": gpuarray.zeros(var.data.shape, dtype=var.data.dtype)}


	def updateVar(self, var, state, stream=None):
		classicMomSGDKer(var.data.dtype)(
			var.data, var.grad, state["mom"], self.learnRate * var.learnRate, self.momRate * var.momRate, stream=stream
		)


def unittest():
	for dtype, atol in dtypesSupported():
		calcTest(dtype, atol)
		trainSimpleTest(MomentumSGD, dtype, learnRate=1e-1, momRate=0.9)

		if Config.backend == Config.Backend.cuda:
			trainHardTest(MomentumSGD, dtype, learnRate=1e-1, momRate=0.9)


def calcTest(dtype, atol):
	lr, mr = 0.01, 0.9
	shape = (11, 13)

	hostW, hostDw = np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)
	hostMom = np.random.randn(*shape).astype(dtype)

	w, dw, mom = gpuarray.to_gpu(hostW), gpuarray.to_gpu(hostDw), gpuarray.to_gpu(hostMom)
	classicMomSGDKer(w.dtype)(w, dw, mom, lr, mr)

	hostW, hostDw, hostMom = hostW.astype(np.float32), hostDw.astype(np.float32), hostMom.astype(np.float32)

	hostMom = mr * hostMom + lr * hostDw
	hostW += hostMom

	hostW, hostDw, hostMom = hostW.astype(dtype), hostDw.astype(dtype), hostMom.astype(dtype)

	assert np.allclose(hostMom, mom.get(), atol=atol)
	assert np.allclose(hostW, w.get(), atol=atol)


if __name__ == "__main__":
	unittest()
