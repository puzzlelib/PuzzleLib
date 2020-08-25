import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.gpuarray import memoryPool as memPool
from PuzzleLib.Backend.Kernels.Costs import hingeKer

from PuzzleLib.Cost.Cost import CostError, Cost


class Hinge(Cost):
	def calcGrad(self, scores, labels):
		if Config.verifyData:
			self.verifyLabels(labels)

		grad = gpuarray.empty(scores.shape, dtype=np.float32, allocator=memPool)
		self.devErr.fill(0.0)

		hingeKer(scores, labels, self.devErr, grad, scores.shape[0], scores.shape[1])
		return grad


	def calcError(self, scores, labels):
		self.accumErr += self.devErr


	def calcVal(self, scores, labels):
		if Config.verifyData:
			self.verifyLabels(labels)

		diff = gpuarray.empty(scores.shape, dtype=np.float32, allocator=memPool)
		devErr = gpuarray.zeros((), dtype=np.float32, allocator=memPool)

		hingeKer(scores, labels, devErr, diff, scores.shape[0], scores.shape[1])
		return devErr.get() / scores.shape[0]


	def checkDataShape(self, scores, labels):
		assert scores.ndim == 2 and scores.shape == labels.shape
		assert labels.dtype == np.int32


	def checkValDataShape(self, scores, labels):
		assert scores.ndim == 2 and scores.shape == labels.shape
		assert labels.dtype == np.int32


	@staticmethod
	def verifyLabels(labels):
		mn, mx = gpuarray.minimum(labels).get(), gpuarray.maximum(labels).get()
		if mn < -1:
			raise CostError("Hinge labels verification failed, found index %s (< -1)" % mn)

		if mx > 1:
			raise CostError("Hinge labels verification failed, found index %s (> 1)" % mx)


def unittest():
	errorValTest()
	verifyLabelsTest()


def errorValTest():
	batchsize, size = 20, 4

	scores = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))
	labels = gpuarray.to_gpu((np.random.randint(low=0, high=2, size=(batchsize, size)) * 2 - 1).astype(np.int32))

	hinge = Hinge()
	error, grad = hinge(scores, labels)

	hostScores, hostLabels = scores.get(), labels.get()

	hostGrad = np.empty(grad.shape, dtype=np.float32)
	hostError = 0.0

	for b in range(batchsize):
		for n in range(size):
			val = hostLabels[b, n] * hostScores[b, n]

			hostGrad[b, n] = hostLabels[b, n] / batchsize / size if val < 1.0 else 0.0
			hostError += max(0.0, 1.0 - val) / batchsize / size

	assert np.allclose(hostGrad, grad.get())
	assert np.isclose(hostError, error)

	error = hinge.validate(scores, labels)
	assert np.isclose(hostError, error)


def verifyLabelsTest():
	batchsize, size = 20, 4

	scores = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))
	labels = gpuarray.to_gpu(np.random.randint(low=-2, high=3, size=(batchsize, size)).astype(np.int32))

	hinge = Hinge()

	Config.verifyData = True

	try:
		hinge(scores, labels)

	except CostError as e:
		print("Caught labels verification error: %s" % e)


if __name__ == "__main__":
	unittest()
