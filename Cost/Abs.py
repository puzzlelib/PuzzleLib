import numpy as np

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.gpuarray import memoryPool as memPool
from PuzzleLib.Backend.Kernels.ElementWise import l1gradKer

from PuzzleLib.Cost.Cost import Cost


class Abs(Cost):
	def calcGrad(self, pred, target):
		grad = gpuarray.empty(pred.shape, dtype=np.float32, allocator=memPool)
		norm = 1.0 / np.prod(target.shape)

		l1gradKer(grad, pred, target, norm)

		return grad


	def calcError(self, pred, target):
		diff = Blas.addVectorToVector(pred.ravel(), target.ravel(), alpha=1.0, beta=-1.0)

		self.devErr.fill(Blas.vectorL1Norm(diff) / np.prod(pred.shape[1:]))
		self.accumErr += self.devErr


	def calcVal(self, pred, target):
		diff = Blas.addVectorToVector(pred.ravel(), target.ravel(), alpha=1.0, beta=-1.0)
		error = Blas.vectorL1Norm(diff) / np.prod(target.shape)

		return error


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

	abscost = Abs()
	abscost(pred, target)

	assert np.isclose(abscost.error, np.linalg.norm((target.get() - pred.get()).ravel(), ord=1) / np.prod(target.shape))


def valTest():
	pred = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))
	target = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))

	abscost = Abs()
	error = abscost.validate(pred, target)

	assert np.isclose(error, np.linalg.norm((target.get() - pred.get()).ravel(), ord=1) / np.prod(target.shape))


if __name__ == "__main__":
	unittest()
