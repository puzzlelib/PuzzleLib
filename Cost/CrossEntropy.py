import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool
from PuzzleLib.Backend.Kernels.Costs import getAccuracyKernel, crossEntropyKernel
from PuzzleLib.Backend.Kernels.MatVec import argmax
from PuzzleLib.Backend.Kernels.MatVecBatch import argmaxBatch

from PuzzleLib.Cost.Cost import CostError, Cost


class CrossEntropy(Cost):
	def __init__(self, maxlabels=None, weights=None):
		super().__init__()

		self.maxlabels = maxlabels
		self.mostProb = None

		if isinstance(weights, np.ndarray):
			weights = gpuarray.to_gpu(weights)

		self.weights = weights


	def calcGrad(self, scores, labels):
		if Config.verifyData:
			self.verifyLabels(scores, labels)

		self.devErr, grad = crossEntropyKernel(scores, labels, weights=self.weights, error=self.devErr)
		return grad


	def calcError(self, scores, labels):
		self.accumErr += self.devErr


	def calcVal(self, scores, labels):
		if Config.verifyData:
			self.verifyLabels(scores, labels)

		if scores.ndim == 2:
			self.mostProb = argmax(scores, axis=1)

		else:
			scores = scores.reshape(*scores.shape[:2], int(np.prod(scores.shape[2:])))
			self.mostProb = argmaxBatch(scores, axis=1).reshape(labels.shape)

		calcAccuracy = getAccuracyKernel("calcAccuracy")
		error = calcAccuracy(self.mostProb, labels, allocator=memPool).get() / np.prod(labels.shape)

		return error


	def reset(self):
		super().reset()
		self.mostProb = None


	def checkDataShape(self, scores, labels):
		assert scores.ndim > 1 and labels.ndim == scores.ndim - 1
		assert labels.dtype == np.int32

		if scores.ndim > 2:
			assert scores.shape[2:] == labels.shape[1:]

		if self.maxlabels:
			assert scores.shape[1] == self.maxlabels

		if self.weights is not None:
			assert self.weights.shape[0] == scores.shape[1]


	def checkValDataShape(self, scores, labels):
		assert scores.ndim > 1 and labels.ndim == scores.ndim - 1
		assert labels.dtype == np.int32

		if scores.ndim > 2:
			assert scores.shape[2:] == labels.shape[1:]

		if self.maxlabels:
			assert scores.shape[1] == self.maxlabels


	@staticmethod
	def verifyLabels(scores, labels):
		mn, mx = gpuarray.minimum(labels).get(), gpuarray.maximum(labels).get()
		if mn < 0:
			raise CostError("Cross entropy labels verification failed, found index %s (< 0)" % mn)

		if mx >= scores.shape[1]:
			raise CostError("Cross entropy labels verification failed, found index %s (> %s)" %
							(mx, scores.shape[1] - 1))


def unittest():
	errorTest()
	valTest()

	wceErrorTest()
	wceValTest()

	verifyLabelsTest()


def errorTest():
	scores = gpuarray.to_gpu(np.random.randn(20, 10, 3).astype(np.float32))
	labels = gpuarray.to_gpu(np.random.randint(low=0, high=10, size=(20, 3)).astype(np.int32))

	entr = CrossEntropy()
	error, grad = entr(scores, labels)

	def softmax(w):
		e = np.exp(w - np.amax(w))
		dist = e / np.sum(e)
		return dist

	def crossEntropy(smax, target):
		smax = np.moveaxis(smax, 1, -1).reshape(-1, smax.shape[1])
		target = target.flatten()
		err = np.sum(np.log(np.array([smax[i, target[i]] for i in range(smax.shape[0])])))

		return -err / target.size

	def crossEntropyGrad(target, smax):
		return np.array([(target == i) - smax[i] for i in range(smax.shape[0])])

	hostSoftmax = np.apply_along_axis(softmax, 1, scores.get())

	hostGrad = np.vstack([crossEntropyGrad(labels.get()[i], hostSoftmax[i]) / scores.shape[0]
						  for i in range(scores.shape[0])]).reshape(*hostSoftmax.shape)

	assert np.allclose(hostGrad, grad.get())

	hostError = crossEntropy(hostSoftmax, labels.get())
	assert np.isclose(hostError, error)


def valTest():
	scores = gpuarray.to_gpu(np.array([[0.1, 0.0, 0.0, -1.0]] * 150 + [[-0.2, 1.0, 0.0, 0.5]] * 150 +
									  [[0.0, -1.0, 2.0, 1.5]] * 300 + [[0.0, 0.0, -6.0, 1.0]] * 400, dtype=np.float32))
	scores = scores.reshape(*scores.shape, 1)

	labels = gpuarray.to_gpu(np.array([0] * 100 + [1] * 200 + [2] * 300 + [3] * 400, dtype=np.int32))
	labels = labels.reshape(*labels.shape, 1)

	entr = CrossEntropy()
	error = entr.validate(scores, labels)
	print("Validation error: %s" % error)
	assert np.allclose(np.argmax(scores.get(), axis=1), entr.mostProb.get())


def wceErrorTest():
	scores = gpuarray.to_gpu(np.random.randn(20, 10, 3).astype(np.float32))
	labels = gpuarray.to_gpu(np.random.randint(low=0, high=10, size=(20, 3)).astype(np.int32))

	weights = np.random.random_sample(10).astype(np.float32)
	weights /= np.sum(weights)

	entr = CrossEntropy(weights=weights)
	error, grad = entr(scores, labels)

	def softmax(w):
		e = np.exp(w - np.amax(w))
		dist = e / np.sum(e)
		return dist

	def weightedCrossEntropy(smax, w, target):
		smax = np.moveaxis(smax, 1, -1).reshape(-1, smax.shape[1])
		target = target.flatten()
		err = np.sum(np.array([w[target[i]] * np.log(smax[i, target[i]]) for i in range(smax.shape[0])]))
		return -err / smax.shape[0]

	def weightedCrossEntropyGrad(target, w, smax):
		return np.array([w[i] * ((target == i) - smax[i]) for i in range(smax.shape[0])])

	hostSoftmax = np.apply_along_axis(softmax, 1, scores.get())

	hostGrad = np.vstack([weightedCrossEntropyGrad(labels.get()[i], weights, hostSoftmax[i]) / scores.shape[0]
						  for i in range(scores.shape[0])]).reshape(*hostSoftmax.shape)

	assert np.allclose(hostGrad, grad.get())

	hostError = np.array(weightedCrossEntropy(hostSoftmax, weights, labels.get()))
	assert np.isclose(hostError, error)


def wceValTest():
	scores = gpuarray.to_gpu(np.array([[0.1, 0.0, 0.0, -1.0]] * 150 + [[-0.2, 1.0, 0.0, 0.5]] * 150 +
									  [[0.0, -1.0, 2.0, 1.5]] * 300 + [[0.0, 0.0, -6.0, 1.0]] * 400, dtype=np.float32))
	scores = scores.reshape(*scores.shape, 1)

	labels = gpuarray.to_gpu(np.array([0] * 100 + [1] * 200 + [2] * 300 + [3] * 400, dtype=np.int32))
	labels = labels.reshape(*labels.shape, 1)

	weights = np.random.random_sample(10).astype(np.float32)
	weights /= np.sum(weights)

	entr = CrossEntropy(weights=weights)
	error = entr.validate(scores, labels)

	print("Validation error: %s" % error)
	assert np.allclose(np.argmax(scores.get(), axis=1), entr.mostProb.get())


def verifyLabelsTest():
	scores = gpuarray.to_gpu(np.random.randn(20, 10).astype(np.float32))
	labels = gpuarray.to_gpu(np.random.randint(low=0, high=11, size=(20, )).astype(np.int32))

	entr = CrossEntropy()

	Config.verifyData = True

	try:
		entr(scores, labels)

	except CostError as e:
		print("Caught labels verification error: %s" % e)


if __name__ == "__main__":
	unittest()
