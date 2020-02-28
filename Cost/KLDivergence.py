import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool
from PuzzleLib.Backend.Kernels.Costs import getAccuracyKernel
from PuzzleLib.Backend.Dnn.Basic import softmaxNd

from PuzzleLib.Cost.Cost import Cost


class KLDivergence(Cost):
	def __init__(self, maxlabels=None, normTarget=False):
		super().__init__()

		self.maxlabels = maxlabels
		self.normTarget = normTarget


	def calcGrad(self, pred, target):
		shape = pred.shape
		softmax = softmaxNd(pred.reshape(shape[0], int(np.prod(shape[1:])), 1, 1))

		if self.normTarget:
			shape = target.shape
			target = softmaxNd(target.reshape(shape[0], int(np.prod(shape[1:])), 1, 1))

		grad = gpuarray.empty(pred.shape, dtype=np.float32, allocator=memPool)

		self.devErr = None

		gradnorm = 1.0 / softmax.shape[0]

		klDivergence = getAccuracyKernel("klDivergence")
		self.devErr = klDivergence(softmax, target, grad, gradnorm, allocator=memPool)

		return grad


	def calcError(self, pred, target):
		self.accumErr += self.devErr


	def calcVal(self, pred, target):
		shape = pred.shape
		softmax = softmaxNd(pred.reshape(shape[0], int(np.prod(shape[1:])), 1, 1))

		if self.normTarget:
			shape = target.shape
			target = softmaxNd(target.reshape(shape[0], int(np.prod(shape[1:])), 1, 1))

		grad = gpuarray.empty(pred.shape, dtype=np.float32, allocator=memPool)

		gradnorm = 1.0 / softmax.shape[0]

		klDivergence = getAccuracyKernel("klDivergence")
		error = klDivergence(softmax, target, grad, gradnorm, allocator=memPool)

		return error.get() / shape[0]


	def checkDataShape(self, pred, target):
		assert pred.shape[1:] == target.shape[1:]

		if self.maxlabels:
			assert pred.shape[1] == self.maxlabels


	def checkValDataShape(self, pred, target):
		assert pred.shape[1:] == target.shape[1:]

		if self.maxlabels:
			assert pred.shape[1] == self.maxlabels


def unittest():
	def softmax(w):
		e = np.exp(w - np.amax(w))
		dist = e / np.sum(e)
		return dist

	def klDivergence(smax, target):
		error = np.sum(target * (np.log(target) - np.log(smax)))
		return error / smax.shape[0]

	def klDivergenceGrad(target, smax):
		return target - smax

	errorTest(softmax, klDivergence, klDivergenceGrad)
	valTest(softmax, klDivergence)


def errorTest(softmax, klDivergence, klDivergenceGrad):
	pred = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))
	target = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))

	div = KLDivergence(normTarget=True)
	error, grad = div(pred, target)

	hostSoftmax = np.vstack([softmax(pred.get()[i]) for i in range(pred.shape[0])])
	hostTarget = np.vstack([softmax(target.get()[i]) for i in range(target.shape[0])])

	hostGrad = np.vstack([klDivergenceGrad(hostTarget[i], hostSoftmax[i]) / pred.shape[0]
						  for i in range(pred.shape[0])])
	assert np.allclose(hostGrad, grad.get())

	hostError = np.array(klDivergence(hostSoftmax, hostTarget))
	assert np.isclose(hostError, error)


def valTest(softmax, klDivergence):
	pred = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))
	target = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))

	div = KLDivergence(normTarget=True)
	error = div.validate(pred, target)

	hostSoftmax = np.vstack([softmax(pred.get()[i]) for i in range(pred.shape[0])])
	hostTarget = np.vstack([softmax(target.get()[i]) for i in range(target.shape[0])])

	hostError = np.array(klDivergence(hostSoftmax, hostTarget))
	assert np.isclose(hostError, error)


if __name__ == "__main__":
	unittest()
