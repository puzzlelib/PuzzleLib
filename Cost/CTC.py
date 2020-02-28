import math, random

import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Kernels.Costs import ctcLoss, ctcLossTest

from PuzzleLib.Cost.Cost import Cost


class CTC(Cost):
	def __init__(self, blank, vocabsize=None, normalized=False):
		super().__init__()
		self.normalized = normalized

		if vocabsize is not None:
			assert 0 <= blank <= vocabsize

		self.vocabsize = vocabsize
		self.blank = blank


	def calcGrad(self, pred, target):
		data, datalen = pred
		labels, lengths = target

		self.devErr.fill(0.0)

		_, grad = ctcLoss(data, datalen, labels, lengths, self.blank, error=self.devErr, normalized=self.normalized)
		return grad


	def calcError(self, scores, labels):
		self.accumErr += self.devErr


	def calcVal(self, pred, target):
		raise NotImplementedError()


	def checkDataShape(self, pred, target):
		data, datalen = pred
		labels, lengths = target

		assert datalen.dtype == labels.dtype and labels.dtype == lengths.dtype and lengths.dtype == np.int32
		assert datalen.shape[0] == lengths.shape[0] and lengths.shape[0] == data.shape[1]

		if self.vocabsize is not None:
			assert data.shape[2] == self.vocabsize


	def checkValDataShape(self, pred, target):
		pass


	def getBatchsize(self, pred):
		return pred[0].shape[1]


def unittest():
	smallTest()
	mediumTest()
	randomTest()


def smallTest():
	hostData = np.array([[[0.1, 0.6, 0.1, 0.1, 0.1]], [[0.1, 0.1, 0.6, 0.1, 0.1]]], dtype=np.float32)

	data = gpuarray.to_gpu(hostData)
	datalen = gpuarray.to_gpu(np.array([2], dtype=np.int32))

	labels = gpuarray.to_gpu(np.array([1, 2], dtype=np.int32))
	lengths = np.array([2], dtype=np.int32)

	ctc = CTC(blank=4, vocabsize=5, normalized=True)
	error, grad = ctc([data, datalen], [labels, lengths])

	hostScore = hostData[0, 0, 1] * hostData[1, 0, 2]
	assert np.isclose(math.exp(-error), hostScore)


def mediumTest():
	hostData = np.array([
		[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
		 [0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508]],

		[[0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
		 [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549]],

		[[0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
		 [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456]],

		[[0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
		 [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345]],

		[[0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107],
		 [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]
	], dtype=np.float32)

	data = gpuarray.to_gpu(hostData)
	datalen = gpuarray.to_gpu(np.array([5, 5], dtype=np.int32))

	labels = gpuarray.to_gpu(np.array([
		0, 1, 2, 1, 0,
		0, 1, 1, 0
	], dtype=np.int32))

	lengths = np.array([5, 4], dtype=np.int32)

	hostGrad = -np.array([
		[[-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
		 [-0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508]],

		[[0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436],
		 [0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549]],

		[[0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688],
		 [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544]],

		[[0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533],
		 [0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345]],

		[[-0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107],
		 [-0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]
	], dtype=np.float32)

	ctc = CTC(vocabsize=6, blank=5, normalized=True)
	error, grad = ctc([data, datalen], [labels, lengths])

	hostScore = np.empty((2, ), dtype=np.float32)

	hostScore[0] = -math.log(
		hostData[0, 0, 0] * hostData[1, 0, 1] * hostData[2, 0, 2] * hostData[3, 0, 1] * hostData[4, 0, 0]
	)
	hostScore[1] = 5.42262

	hostError = np.mean(hostScore)

	assert np.isclose(hostError, error)
	assert np.allclose(hostGrad, grad.get())


def randomTest():
	times, batchsize, vocabsize = 20, 3, 6
	hostData, hostDataLen, hostLabels, lengths = createData(times, batchsize, vocabsize)

	data, datalen, labels = gpuarray.to_gpu(hostData), gpuarray.to_gpu(hostDataLen), gpuarray.to_gpu(hostLabels)
	blank = 0

	ctc = CTC(blank=0, vocabsize=vocabsize)

	error, grad = ctc([data, datalen], [labels, lengths])
	hostError, hostGrad, _ = ctcLossTest(hostData, hostDataLen, hostLabels, lengths, blank)

	assert np.isclose(hostError / batchsize, error)
	assert np.allclose(hostGrad, grad.get(), atol=1e-5)


def createData(times, batchsize, vocabsize):
	data = np.random.randn(times, batchsize, vocabsize).astype(np.float32)
	datalen = np.array([times] * batchsize, dtype=np.int32)

	lengths = np.array([random.randint(a=times // 4, b=times // 2 - 1) for _ in range(batchsize)], dtype=np.int32)
	labels = np.concatenate([
		np.array([random.randint(a=1, b=vocabsize - 1) for _ in range(lengths[b])], dtype=np.int32)
		for b in range(batchsize)
	])

	return data, datalen, labels, lengths


if __name__ == "__main__":
	unittest()
