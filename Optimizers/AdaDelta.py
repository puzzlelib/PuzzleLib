import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import dtypesSupported
from PuzzleLib.Backend.Kernels.ElementWise import adadeltaKer

from PuzzleLib.Optimizers.Optimizer import Optimizer, trainSimpleTest, trainHardTest


class AdaDelta(Optimizer):
	def __init__(self, rho=0.95, epsilon=1e-6, nodeinfo=None):
		super().__init__(nodeinfo)

		self.rho = None
		self.epsilon = None

		self.setAttr("rho", rho)
		self.setAttr("epsilon", epsilon)


	def setupState(self, var):
		return {
			"msg": gpuarray.zeros(var.data.shape, dtype=var.data.dtype),
			"msdx": gpuarray.zeros(var.data.shape, dtype=var.data.dtype)
		}


	def updateVar(self, var, state, stream=None):
		adadeltaKer(var.data.dtype)(
			var.data, var.grad, state["msg"], state["msdx"], self.rho, self.epsilon, stream=stream
		)


def unittest():
	for dtype, atol in dtypesSupported():
		calcTest(dtype, atol)
		trainSimpleTest(AdaDelta, dtype)

		if Config.backend == Config.Backend.cuda:
			trainHardTest(AdaDelta, dtype)


def calcTest(dtype, atol):
	rho, epsilon = 0.95, 1e-6
	shape = (11, 13)

	hostW, hostDw = np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)
	hostMsg = (1.0 + np.random.randn(*shape)**2).astype(dtype)
	hostMsdx = (1.0 + np.random.randn(*shape)**2).astype(dtype)

	w, dw = gpuarray.to_gpu(hostW), gpuarray.to_gpu(hostDw)
	msg, msdx = gpuarray.to_gpu(hostMsg), gpuarray.to_gpu(hostMsdx)

	adadeltaKer(w.dtype)(w, dw, msg, msdx, rho, epsilon)

	hostW, hostDw = hostW.astype(np.float32), hostDw.astype(np.float32)
	hostMsg, hostMsdx = hostMsg.astype(np.float32), hostMsdx.astype(np.float32)

	hostMsg += (1.0 - rho) * (hostDw * hostDw - hostMsg)
	hostDx = np.sqrt((hostMsdx + epsilon) / (hostMsg + epsilon)) * hostDw
	hostMsdx += (1.0 - rho) * (hostDx**2 - hostMsdx)
	hostW += hostDx

	hostW, hostDw = hostW.astype(dtype), hostDw.astype(dtype)
	hostMsg, hostMsdx = hostMsg.astype(dtype), hostMsdx.astype(dtype)

	assert np.allclose(hostMsg, msg.get(), atol=atol)
	assert np.allclose(hostMsdx, msdx.get(), atol=atol)
	assert np.allclose(hostW, w.get(), atol=atol)


if __name__ == "__main__":
	unittest()
