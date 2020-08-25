import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.gpuarray import memoryPool as memPool
from PuzzleLib.Backend.Kernels.Costs import getAccuracyKernel, svmKernel
from PuzzleLib.Backend.Kernels.MatVec import argmax, argmaxBatch

from PuzzleLib.Cost.Cost import CostError, Cost


class SVM(Cost):
	def __init__(self, mode="l1"):
		super().__init__()

		self.mode = mode
		self.mostProb = None


	def calcGrad(self, scores, labels):
		if Config.verifyData:
			self.verifyLabels(scores, labels)

		self.devErr, grad = svmKernel(scores, labels, mode=self.mode, error=self.devErr)
		return grad


	def calcError(self, scores, labels):
		self.accumErr += self.devErr


	def calcVal(self, scores, labels):
		if Config.verifyData:
			self.verifyLabels(scores, labels)

		if scores.ndim == 2:
			shape = scores.shape
			self.mostProb = argmax(scores, axis=1)

		else:
			shape = scores.shape[:1] + scores.shape[2:]
			scores = scores.reshape(*scores.shape[:2], np.prod(scores.shape[2:]))

			self.mostProb = argmaxBatch(scores, axis=1).reshape(shape)

		calcAccuracy = getAccuracyKernel("calcAccuracy")
		error = calcAccuracy(self.mostProb, labels, allocator=memPool).get() / shape[0]

		return error


	def reset(self):
		super().reset()
		self.mostProb = None


	def checkDataShape(self, scores, labels):
		assert scores.ndim > 1 and labels.ndim == scores.ndim - 1
		assert labels.dtype == np.int32

		if scores.ndim > 2:
			assert scores.shape[2:] == labels.shape[1:]


	def checkValDataShape(self, scores, labels):
		assert scores.ndim > 1 and labels.ndim == scores.ndim - 1
		assert labels.dtype == np.int32

		if scores.ndim > 2:
			assert scores.shape[2:] == labels.shape[1:]


	@staticmethod
	def verifyLabels(scores, labels):
		mn, mx = gpuarray.minimum(labels).get(), gpuarray.maximum(labels).get()
		if mn < 0:
			raise CostError("SVM labels verification failed, found index %s (< 0)" % mn)

		if mx >= scores.shape[1]:
			raise CostError("SVM labels verification failed, found index %s (> %s)" % (mx, scores.shape[1] - 1))


def unittest():
	l1SVMTest()
	l2SVMTest()
	verifyLabelsTest()


def l1SVMTest():
	batchsize, size = 20, 4

	scores = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))
	labels = gpuarray.to_gpu(np.random.randint(low=0, high=size, size=(batchsize, ), dtype=np.int32))

	svm = SVM(mode="l1")
	error, grad = svm(scores, labels)

	hostScores, hostLabels = scores.get(), labels.get()

	hostGrad = np.empty(grad.shape, dtype=np.float32)
	hostError = 0.0

	for b in range(batchsize):
		for n in range(size):
			cls = 2 * (hostLabels[b] == n) - 1
			val = hostScores[b, n] * cls

			hostGrad[b, n] = cls / batchsize / size if val < 1 else 0.0
			hostError += max(0.0, 1.0 - val) / batchsize / size

	assert np.allclose(hostGrad, grad.get())
	assert np.isclose(hostError, error)

	error = svm.validate(scores, labels)
	print("Validation error: %s" % error)
	assert np.allclose(np.argmax(scores.get(), axis=1), svm.mostProb.get())


def l2SVMTest():
	batchsize, size = 20, 4

	scores = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))
	labels = gpuarray.to_gpu(np.random.randint(low=0, high=size, size=(batchsize, ), dtype=np.int32))

	svm = SVM(mode="l2")
	error, grad = svm(scores, labels)

	hostScores, hostLabels = scores.get(), labels.get()

	hostGrad = np.empty(grad.shape, dtype=np.float32)
	hostError = 0.0

	for b in range(batchsize):
		for n in range(size):
			cls = 2 * (hostLabels[b] == n) - 1
			err = max(0.0, 1.0 - hostScores[b, n] * cls)

			hostGrad[b, n] = 2.0 * cls * err / batchsize / size
			hostError += err**2 / batchsize / size

	assert np.allclose(hostGrad, grad.get())
	assert np.isclose(hostError, error)

	error = svm.validate(scores, labels)

	print("Validation error: %s" % error)
	assert np.allclose(np.argmax(scores.get(), axis=1), svm.mostProb.get())


def verifyLabelsTest():
	batchsize, size = 20, 4

	scores = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))
	labels = gpuarray.to_gpu(np.random.randint(low=0, high=size + 1, size=(batchsize, ), dtype=np.int32))

	svm = SVM(mode="l1")

	Config.verifyData = True

	try:
		svm(scores, labels)

	except CostError as e:
		print("Caught labels verification error: %s" % e)


if __name__ == "__main__":
	unittest()
