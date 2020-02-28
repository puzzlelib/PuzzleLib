import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend import Blas

from PuzzleLib.Cost.Cost import Cost


class MSE(Cost):
	def calcGrad(self, pred, target):
		c = 1.0 / np.prod(target.shape)
		grad = Blas.addVectorToVector(target.ravel(), pred.ravel(), alpha=c, beta=-c)
		grad = grad.reshape(pred.shape)

		return grad


	def calcError(self, pred, target):
		self.devErr.fill(Blas.dot(self.grad.ravel(), self.grad.ravel()) * np.prod(self.grad.shape)
						 * self.grad.shape[0] / 2.0)
		self.accumErr += self.devErr


	def calcVal(self, pred, target):
		diff = Blas.addVectorToVector(target.ravel(), pred.ravel(), alpha=1.0, beta=-1.0)
		error = Blas.dot(diff, diff) / (2.0 * np.prod(target.shape))

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

	mse = MSE()
	mse(pred, target)

	assert np.isclose(mse.error, np.linalg.norm(target.get() - pred.get())**2 / (2.0 * np.prod(target.shape)))


def valTest():
	pred = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))
	target = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))

	mse = MSE()
	error = mse.validate(pred, target)

	assert np.isclose(error, np.linalg.norm(target.get() - pred.get())**2 / (2.0 * np.prod(target.shape)))


if __name__ == "__main__":
	unittest()
