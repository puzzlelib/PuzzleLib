import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.Dnn import PoolMode, poolNd, poolNdBackward, mapLRN, mapLRNBackward

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Modules.LRN import LRN


class LCN(LRN):
	def __init__(self, N=5, alpha=1e-4, beta=0.75, K=2.0, includePad=True, name=None):
		super().__init__(N, alpha, beta, K, name)
		self.registerBlueprint(locals())

		if N % 2 != 1 or N == 1:
			raise ModuleError("LCN size must be odd and > 1")

		self.size = self.repeat(N, 2)
		self.pad = (self.size[0] // 2, self.size[1] // 2)

		self.gradUsesOutData = Config.backend != Config.Backend.cuda

		self.includePad = includePad
		self.mode = PoolMode.avgWithPad if includePad else PoolMode.avgNoPad

		self.means = None
		self.poolspace = None
		self.lrnspace = None


	def updateData(self, data):
		self.means, self.poolspace = poolNd(
			data, size=self.size, stride=1, pad=self.pad, mode=self.mode, test=not self.train
		)
		self.data, self.lrnspace = mapLRN(
			data, self.means, N=self.N, alpha=self.alpha, beta=self.beta, K=self.K, test=not self.train
		)


	def updateGrad(self, grad):
		self.grad, meansGrad = mapLRNBackward(
			self.inData, self.data, grad, self.means, None, N=self.N, alpha=self.alpha, beta=self.beta, K=self.K
		)

		if self.includePad:
			meansGrad = poolNdBackward(
				self.inData, self.means, meansGrad, self.workspace, size=self.size, stride=1, pad=self.pad,
				mode=self.mode
			)
			Blas.addVectorToVector(self.grad.ravel(), meansGrad.ravel(), out=self.grad.ravel(), beta=-1.0)


	def reset(self):
		super().reset()
		self.means = None
		self.poolspace = None
		self.lrnspace = None


def unittest():
	batchsize, maps, h, w = 1, 1, 5, 5
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, h, w).astype(np.float32))

	lcn = LCN(N=5)
	lcn(data)

	lookBehind = int((lcn.N - 1) / 2)
	lookAhead = lcn.N - lookBehind

	hsize, wsize = lcn.size
	hpad, wpad = lcn.pad

	hostData = np.zeros(shape=(batchsize, maps, h + 2 * hpad, w + 2 * wpad), dtype=np.float32)
	hostData[:, :, hpad:-hpad, wpad:-wpad] = data.get()

	hostMeans = np.empty(lcn.data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(lcn.data.shape[2]):
				for x in range(lcn.data.shape[3]):
					hostMeans[b, c, y, x] = np.sum(hostData[b, c, y:y + hsize, x:x + wsize]) / (hsize * wsize)

	assert np.allclose(hostMeans, lcn.means.get())
	norms = np.empty(lcn.data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(h):
				for x in range(w):
					norm = 0.0

					for dy in range(max(0, y - lookBehind), min(h, y + lookAhead)):
						for dx in range(max(0, x - lookBehind), min(w, x + lookAhead)):
							norm += (hostData[b, c, dy, dx] - hostMeans[b, c, y, x])**2

					norms[b, c, y, x] = lcn.K + norm * lcn.alpha / lcn.N**2

	hostOutData = hostData[:, :, hpad:-hpad, wpad:-wpad] / (norms**lcn.beta)
	assert np.allclose(hostOutData, lcn.data.get(), atol=1e-5)

	grad = gpuarray.to_gpu(np.random.randn(*lcn.data.shape).astype(np.float32))
	lcn.backward(grad)

	hostGrad = grad.get()
	hostInGrad, hostMeansGrad = np.zeros(data.shape, dtype=np.float32), np.zeros(data.shape, dtype=np.float32)

	k = 2.0 * lcn.alpha * lcn.beta / lcn.N**2

	for b in range(batchsize):
		for c in range(maps):
			for y in range(h):
				for x in range(w):
					hostInGrad[b, c, y, x] += hostGrad[b, c, y, x] / norms[b, c, y, x]**lcn.beta

					for dy in range(max(0, y - lookBehind), min(h, y + lookAhead)):
						for dx in range(max(0, x - lookBehind), min(w, x + lookAhead)):
							hostInGrad[b, c, y, x] -= k * hostGrad[b, c, dy, dx] * (
									hostData[b, c, y, x] - hostMeans[b, c, dy, dx]
							) * hostData[b, c, dy, dx] / norms[b, c, dy, dx]**(lcn.beta + 1)

							hostMeansGrad[b, c, y, x] += hostData[b, c, dy, dx] - hostMeans[b, c, y, x]

					K = 2.0 * lcn.alpha * lcn.beta * hostData[b, c, y, x] / lcn.N**2 / \
						   norms[b, c, y, x]**(lcn.beta + 1)

					hostMeansGrad[b, c, y, x] *= K * hostGrad[b, c, y, x]

	extInGrad = np.zeros(hostData.shape, dtype=np.float32)
	extInGrad[:, :, hpad:-hpad, wpad:-wpad] = hostInGrad

	hostInGrad = extInGrad

	for b in range(batchsize):
		for c in range(maps):
			for y in range(hostGrad.shape[2]):
				for x in range(hostGrad.shape[3]):
					for dy in range(lcn.N):
						for dx in range(lcn.N):
							hostInGrad[b, c, y + dy, x + dx] -= hostMeansGrad[b, c, y, x] / lcn.N**2

	assert np.allclose(hostInGrad[:, :, hpad:-hpad, wpad:-wpad], lcn.grad.get(), atol=1e-4)


if __name__ == "__main__":
	unittest()
