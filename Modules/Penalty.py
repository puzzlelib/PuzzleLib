from enum import Enum

import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool
from PuzzleLib.Backend import Blas
from PuzzleLib.Backend.Kernels.ElementWise import l1penaltyKer

from PuzzleLib.Modules.Module import Module


class PenaltyMode(str, Enum):
	l1 = "l1"
	l2 = "l2"


class Penalty(Module):
	def __init__(self, mode="l1", weight=1e-2, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.gradUsesOutData = True
		self.movesData = True

		self.mode = PenaltyMode(mode)
		self.weight = weight


	def updateData(self, data):
		self.data = data


	def updateGrad(self, grad):
		if self.mode == PenaltyMode.l1:
			self.grad = gpuarray.empty(grad.shape, dtype=grad.dtype, allocator=memPool)
			l1penaltyKer(self.grad, grad, self.data, self.weight / grad.shape[0])

		elif self.mode == PenaltyMode.l2:
			self.grad = Blas.addVectorToVector(grad.ravel(), self.data.ravel(), alpha=1.0,
											   beta=-self.weight / grad.shape[0])
			self.grad = self.grad.reshape(grad.shape)

		else:
			raise NotImplementedError(self.mode)


	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return shape


def unittest():
	data = gpuarray.to_gpu(np.random.randn(10, 50).astype(np.float32))

	penalty = Penalty()
	penalty(data)

	grad = gpuarray.to_gpu(np.random.randn(10, 50).astype(np.float32))
	penalty.backward(grad)

	hostGrad = grad.get() - penalty.weight * np.sign(data.get()) / data.shape[0]
	assert np.allclose(hostGrad, penalty.grad.get())


if __name__ == "__main__":
	unittest()
