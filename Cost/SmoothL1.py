import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool
from PuzzleLib.Backend.Kernels.Costs import smoothL1Ker

from PuzzleLib.Cost.Cost import Cost


class SmoothL1(Cost):
	def calcGrad(self, pred, target):
		grad = gpuarray.empty(pred.shape, dtype=np.float32, allocator=memPool)

		fullnorm = 1.0 / np.prod(target.shape)
		norm = 1.0 / np.prod(target.shape[1:])

		self.devErr.fill(0.0)

		smoothL1Ker(pred, target, self.devErr, grad, norm, fullnorm)
		return grad


	def calcError(self, pred, target):
		self.accumErr += self.devErr


	def calcVal(self, pred, target):
		diff = gpuarray.empty(pred.shape, dtype=np.float32, allocator=memPool)

		fullnorm = 1.0 / np.prod(target.shape)

		devErr = gpuarray.zeros((), dtype=np.float32, allocator=memPool)
		smoothL1Ker(pred, target, devErr, diff, fullnorm, fullnorm)

		return devErr.get()


	def checkDataShape(self, pred, target):
		assert pred.shape[1:] == target.shape[1:]


	def checkValDataShape(self, pred, target):
		assert pred.shape[1:] == target.shape[1:]


def unittest():
	errorTest()
	valTest()


def errorTest():
	pred = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))
	target = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))

	smoothL1 = SmoothL1()
	smoothL1(pred, target)

	hostPred, hostTarget = pred.get(), target.get()
	hostGrad = ((np.abs(hostPred - hostTarget) >= 1.0) * np.sign(hostPred - hostTarget) +
			   (np.abs(hostPred - hostTarget) < 1.0) * (hostPred - hostTarget)) / np.prod(pred.shape)

	assert np.allclose(hostGrad, smoothL1.grad.get())

	hostError = np.mean((np.abs(hostPred - hostTarget) >= 1.0) * (np.abs(hostPred - hostTarget) - 0.5) +
						(np.abs(hostPred - hostTarget) < 1.0) * (hostPred - hostTarget)**2 / 2.0)

	assert np.isclose(smoothL1.error, hostError)


def valTest():
	pred = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))
	target = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))

	smoothL1 = SmoothL1()
	error = smoothL1.validate(pred, target)

	hostPred, hostTarget = pred.get(), target.get()

	hostError = np.mean((np.abs(hostPred - hostTarget) >= 1.0) * (np.abs(hostPred - hostTarget) - 0.5) +
						(np.abs(hostPred - hostTarget) < 1.0) * (hostPred - hostTarget)**2 / 2.0)

	assert np.isclose(error, hostError)


if __name__ == "__main__":
	unittest()
