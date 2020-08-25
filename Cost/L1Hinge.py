import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.gpuarray import memoryPool as memPool
from PuzzleLib.Backend.Kernels.ElementWise import absKer
from PuzzleLib.Backend.Kernels.Costs import l1HingeKer, getAccuracyKernel

from PuzzleLib.Cost.Cost import CostError, Cost


class L1Hinge(Cost):
	def calcGrad(self, pair, labels):
		if Config.verifyData:
			self.verifyLabels(labels)

		g1 = gpuarray.empty(pair[0].shape, dtype=np.float32, allocator=memPool)
		g2 = gpuarray.empty(pair[1].shape, dtype=np.float32, allocator=memPool)

		self.devErr.fill(0.0)

		l1HingeKer(pair[0], pair[1], labels, self.devErr, g1, g2, pair[0].shape[0], pair[0].shape[1])
		return [g1, g2]


	def calcError(self, pair, labels):
		self.accumErr += self.devErr


	def calcVal(self, pair, labels):
		if Config.verifyData:
			self.verifyLabels(labels)

		diff = Blas.addVectorToVector(pair[0].ravel(), pair[1].ravel(), alpha=1.0, beta=-1.0).reshape(pair[0].shape)
		absKer(diff, diff)

		dist = Blas.sumOnMatrix(diff, cols=False, alpha=1.0 / pair[0].shape[1])

		l1HingeAccuracy = getAccuracyKernel("l1HingeAccuracy")
		error = l1HingeAccuracy(dist, labels, allocator=memPool).get() / pair[0].shape[0]

		return error


	def checkDataShape(self, pair, labels):
		assert len(pair) == 2 and pair[0].shape == pair[1].shape and pair[0].dtype == pair[1].dtype
		assert pair[0].dtype == np.float32
		assert pair[0].ndim == 2

		assert labels.dtype == np.int32


	def checkValDataShape(self, pair, labels):
		assert len(pair) == 2 and pair[0].shape == pair[1].shape and pair[0].dtype == pair[1].dtype
		assert pair[0].dtype == np.float32
		assert pair[0].ndim == 2

		assert labels.dtype == np.int32


	def getBatchsize(self, pair):
		return pair[0].shape[0]


	@staticmethod
	def verifyLabels(labels):
		mn, mx = gpuarray.minimum(labels).get(), gpuarray.maximum(labels).get()
		if mn < 0:
			raise CostError("L1 Hinge labels verification failed, found index %s (< 0)" % mn)

		if mx > 1:
			raise CostError("L1 Hinge labels verification failed, found index %s (> 1)" % mx)


def unittest():
	errorTest()
	valTest()
	verifyLabelsTest()


def errorTest():
	batchsize, size = 20, 4

	x1 = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))
	x2 = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))

	labels = gpuarray.to_gpu(np.random.randint(low=0, high=2, size=(batchsize, )).astype(np.int32))

	l1Hinge = L1Hinge()
	error, (g1, g2) = l1Hinge([x1, x2], labels)

	hostX1, hostX2, hostLabels = x1.get(), x2.get(), labels.get()

	hostG1, hostG2 = np.empty(g1.shape, dtype=np.float32), np.empty(g2.shape, dtype=np.float32)
	hostError = 0.0

	for b in range(batchsize):
		for n in range(size):
			diff = hostX1[b, n] - hostX2[b, n]
			sign = 1.0 if diff > 0.0 else -1.0

			if hostLabels[b] == 1:
				hostG1[b, n] = sign / batchsize / size
				hostG2[b, n] = -sign / batchsize / size
				hostError += np.abs(diff) / batchsize / size

			else:
				hostG1[b, n] = -sign / batchsize / size if np.abs(diff) < 1.0 else 0.0
				hostG2[b, n] = sign / batchsize / size if np.abs(diff) < 1.0 else 0.0
				hostError += max(0.0, 1.0 - np.abs(diff)) / batchsize / size

	assert np.allclose(hostG1, g1.get())
	assert np.allclose(hostG2, g2.get())
	assert np.isclose(hostError, error)


def valTest():
	batchsize, size = 20, 4

	x1 = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))
	x2 = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))

	labels = gpuarray.to_gpu(np.random.randint(low=0, high=2, size=(batchsize, )).astype(np.int32))

	l1Hinge = L1Hinge()
	error = l1Hinge.validate([x1, x2], labels)

	hostX1, hostX2, hostLabels = x1.get(), x2.get(), labels.get()

	dist = np.linalg.norm(hostX1 - hostX2, axis=1, ord=1) / size
	hostError = np.sum((dist <= 1.0) != labels.get()) / batchsize

	assert np.isclose(hostError, error)


def verifyLabelsTest():
	batchsize, size = 20, 4

	x1 = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))
	x2 = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))
	labels = gpuarray.to_gpu(np.random.randint(low=-1, high=3, size=(batchsize, size)).astype(np.int32))

	l1Hinge = L1Hinge()

	Config.verifyData = True

	try:
		l1Hinge([x1, x2], labels)

	except CostError as e:
		print("Caught labels verification error: %s" % e)


if __name__ == "__main__":
	unittest()
