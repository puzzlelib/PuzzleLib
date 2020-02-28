import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool
from PuzzleLib.Backend.Kernels.Costs import getAccuracyKernel, bceKer

from PuzzleLib.Cost.Cost import CostError, Cost


class BCE(Cost):
	def calcGrad(self, scores, labels):
		if Config.verifyData:
			self.verifyLabels(labels)

		grad = gpuarray.empty(scores.shape, dtype=np.float32, allocator=memPool)
		self.devErr.fill(0.0)

		bceKer(scores, labels, self.devErr, grad, scores.shape[0], np.prod(scores.shape[1:]))
		return grad


	def calcError(self, scores, labels):
		self.accumErr += self.devErr


	def calcVal(self, scores, labels):
		if Config.verifyData:
			self.verifyLabels(labels)

		calcBCEAccuracy = getAccuracyKernel("calcBCEAccuracy")
		error = calcBCEAccuracy(scores, labels, allocator=memPool).get() / np.prod(scores.shape)
		return error


	def checkDataShape(self, scores, labels):
		self.checkShapeCompatibility(scores, labels)


	def checkValDataShape(self, scores, labels):
		self.checkShapeCompatibility(scores, labels)


	@staticmethod
	def checkShapeCompatibility(scores, labels):
		assert labels.dtype == np.int32

		if scores.ndim == 2 and scores.shape[1] == 1:
			assert labels.ndim == 1
		else:
			assert np.prod(scores.shape[1:]) == np.prod(labels.shape[1:])


	@staticmethod
	def verifyLabels(labels):
		mn, mx = gpuarray.minimum(labels).get(), gpuarray.maximum(labels).get()
		if mn < 0:
			raise CostError("BCE labels verification failed, found index %s (< 0)" % mn)

		if mx > 1:
			raise CostError("BCE labels verification failed, found index %s (> 1)" % mx)


def unittest():
	errorTest()
	valTest()
	verifyLabelsTest()


def errorTest():
	scores = gpuarray.to_gpu(np.random.randn(20, 1, 4, 4).astype(np.float32))
	labels = gpuarray.to_gpu(np.random.randint(low=0, high=2, size=(20, 4, 4)).astype(np.int32))

	bce = BCE()
	error, grad = bce(scores, labels)

	hostSigm = 1.0 / (1.0 + np.exp(-scores.get()))

	hostGrad = (labels.get().flatten() - hostSigm.flatten()).reshape(*scores.shape) / np.prod(scores.shape)
	assert np.allclose(hostGrad, grad.get())

	hostError = -np.sum(labels.get().flatten() * np.log(hostSigm).flatten() +
						(1.0 - labels.get().flatten()) * np.log(1.0 - hostSigm).flatten()) / np.prod(hostSigm.shape)
	assert np.isclose(hostError, error)


def valTest():
	scores = gpuarray.to_gpu(np.array([[-1.0]] * 500 + [[1.0]] * 500, dtype=np.float32))
	labels = gpuarray.to_gpu(np.array([0] * 490 + [1] * 510, dtype=np.int32))

	bce = BCE()
	error = bce.validate(scores, labels)

	print("Validation error: %s" % error)
	assert np.isclose(error, np.sum(np.equal((scores.get()<=0.0), labels.get()[:,np.newaxis]), axis=0)/scores.shape[0])


def verifyLabelsTest():
	scores = gpuarray.to_gpu(np.random.randn(20, 1).astype(np.float32))
	labels = gpuarray.to_gpu(np.random.randint(low=0, high=3, size=(20, )).astype(np.int32))

	bce = BCE()

	Config.verifyData = True

	try:
		bce(scores, labels)

	except CostError as e:
		print("Caught labels verification error: %s" % e)


if __name__ == "__main__":
	unittest()
