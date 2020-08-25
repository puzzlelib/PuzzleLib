import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Dnn import mapLRN, mapLRNBackward

from PuzzleLib.Modules.LRN import LRN


class MapLRN(LRN):
	def __init__(self, N=5, alpha=1e-4, beta=0.75, K=2.0, name=None):
		super().__init__(N, alpha, beta, K, name)

		if Config.backend != Config.Backend.cuda:
			self.gradUsesOutData = True


	def updateData(self, data):
		self.data, self.workspace = mapLRN(data, None, N=self.N, alpha=self.alpha, beta=self.beta, K=self.K,
										   test=not self.train)


	def updateGrad(self, grad):
		self.grad = mapLRNBackward(self.inData, self.data, grad, None, self.workspace,
								   N=self.N, alpha=self.alpha, beta=self.beta, K=self.K)


def unittest():
	h, w = 10, 10
	data = gpuarray.to_gpu(np.random.randn(1, 1, h, w).astype(np.float32))

	mapLrn = MapLRN()
	mapLrn(data)

	lookBehind = int((mapLrn.N - 1) / 2)
	lookAhead = mapLrn.N - lookBehind

	hostData = data.get().reshape(h, w).astype(np.float32)
	norms = np.empty((h, w), dtype=np.float32)
	for i in range(h):
		for j in range(w):
			norm = 0.0
			for m in range(max(0, i - lookBehind), min(h, i + lookAhead)):
				for n in range(max(0, j - lookBehind), min(w, j + lookAhead)):
					norm += hostData[m, n]**2
			norms[i, j] = mapLrn.K + norm * mapLrn.alpha / mapLrn.N**2

	hostOutData = hostData / norms**mapLrn.beta
	assert np.allclose(hostOutData, mapLrn.data.get()[0, 0])

	grad = gpuarray.to_gpu(np.random.randn(1, 1, h, w).astype(np.float32))
	mapLrn.backward(grad)

	hostGrad = grad.get().reshape(h, w).astype(np.float32)
	hostInGrad = np.zeros((h, w), dtype=np.float32)

	k = 2.0 * mapLrn.alpha * mapLrn.beta / mapLrn.N**2
	for i in range(h):
		for j in range(w):
			hostInGrad[i, j] += hostGrad[i, j] / norms[i, j]**mapLrn.beta

			for m in range(max(0, i - lookBehind), min(h, i + lookAhead)):
				for n in range(max(0, j - lookBehind), min(w, j + lookAhead)):
					hostInGrad[i, j] -= k*hostGrad[m, n]*hostData[i, j]*hostData[m, n]/norms[m, n]**(mapLrn.beta+1)

	assert np.allclose(hostInGrad, mapLrn.grad.get()[0, 0])


if __name__ == "__main__":
	unittest()
