import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Dnn.Basic import crossMapLRN, crossMapLRNBackward

from PuzzleLib.Modules.LRN import LRN


class CrossMapLRN(LRN):
	def __init__(self, N=5, alpha=1e-4, beta=0.75, K=2.0, name=None):
		super().__init__(N, alpha, beta, K, name)
		self.gradUsesOutData = True


	def updateData(self, data):
		self.data, self.workspace = crossMapLRN(data, N=self.N, alpha=self.alpha, beta=self.beta, K=self.K,
												test=not self.train)


	def updateGrad(self, grad):
		self.grad = crossMapLRNBackward(self.inData, self.data, grad, self.workspace,
										N=self.N, alpha=self.alpha, beta=self.beta, K=self.K)


def unittest():
	maps = 10
	data = gpuarray.to_gpu(np.random.randn(1, maps, 1, 1).astype(np.float32))

	crossMapLrn = CrossMapLRN()
	crossMapLrn(data)

	lookBehind = int((crossMapLrn.N - 1) / 2)
	lookAhead = crossMapLrn.N - lookBehind

	hostData = data.get().reshape(maps, ).astype(np.float32)
	norms = np.empty((maps, ), dtype=np.float32)
	for i in range(maps):
		norm = 0.0
		for j in range(max(0, i - lookBehind), min(maps, i + lookAhead)):
			norm += hostData[j]**2
		norms[i] = crossMapLrn.K + norm * crossMapLrn.alpha / crossMapLrn.N

	hostOutData = hostData / norms**crossMapLrn.beta
	assert np.allclose(hostOutData, crossMapLrn.data.get().reshape(maps, ).astype(np.float32))

	grad = gpuarray.to_gpu(np.random.randn(1, maps, 1, 1).astype(np.float32))
	crossMapLrn.backward(grad)

	hostGrad = grad.get().reshape(maps, ).astype(np.float32)
	hostInGrad = np.zeros((maps, ), dtype=np.float32)

	k = 2.0 * crossMapLrn.alpha * crossMapLrn.beta / crossMapLrn.N
	for i in range(maps):
		hostInGrad[i] += hostGrad[i] / norms[i]**crossMapLrn.beta

		for j in range(max(0, i - lookBehind), min(maps, i + lookAhead)):
			hostInGrad[j] -= hostGrad[i] * k * hostData[i] * hostData[j] / norms[i]**(crossMapLrn.beta+1)
	assert np.allclose(hostInGrad, crossMapLrn.grad.get().reshape(maps, ).astype(np.float32))


if __name__ == "__main__":
	unittest()
