import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Dnn import PoolMode, poolNd, poolNdBackward

from PuzzleLib.Modules.Pool2D import Pool2D


class AvgPool2D(Pool2D):
	def __init__(self, size=2, stride=2, pad=0, includePad=True, name=None):
		super().__init__(size, stride, pad, name)
		self.registerBlueprint(locals())

		self.mode = PoolMode.avgWithPad if includePad else PoolMode.avgNoPad


	def updateData(self, data):
		self.data, self.workspace = poolNd(data, size=self.size, stride=self.stride, pad=self.pad, mode=self.mode,
										   test=not self.train)


	def updateGrad(self, grad):
		self.grad = poolNdBackward(self.inData, self.data, grad, self.workspace, size=self.size, stride=self.stride,
								   pad=self.pad, mode=self.mode)


def unittest():
	batchsize, maps, h, w = 2, 3, 6, 6
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, h, w).astype(np.float32))

	size = 3
	stride, pad = 1, 1

	avgpool2d = AvgPool2D(size=size, stride=stride, pad=pad, includePad=True)
	avgpool2d(data)

	grad = gpuarray.to_gpu(np.random.randn(*avgpool2d.data.shape).astype(np.float32))
	avgpool2d.backward(grad)

	hostData = np.zeros(shape=(batchsize, maps, h + 2 * pad, w + 2 * pad), dtype=np.float32)
	hostData[:, :, pad:-pad, pad:-pad] = data.get()

	hostOutData = np.empty(avgpool2d.data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(avgpool2d.data.shape[2]):
				for x in range(avgpool2d.data.shape[3]):
					hostOutData[b,c,y,x] = np.sum(hostData[b,c,y*stride:y*stride+size, x*stride:x*stride+size])/size**2

	assert np.allclose(hostOutData, avgpool2d.data.get())

	hostGrad, hostInGrad = grad.get(), np.zeros(hostData.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(hostGrad.shape[2]):
				for x in range(hostGrad.shape[3]):
					for dy in range(size):
						for dx in range(size):
							hostInGrad[b, c, y * stride + dy, x * stride + dx] += hostGrad[b, c, y, x] / size**2

	assert np.allclose(hostInGrad[:, :, pad:-pad, pad:-pad], avgpool2d.grad.get())


if __name__ == "__main__":
	unittest()
