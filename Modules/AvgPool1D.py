import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Dnn import PoolMode, poolNd, poolNdBackward

from PuzzleLib.Modules.Pool1D import Pool1D


class AvgPool1D(Pool1D):
	def __init__(self, size=2, stride=2, pad=0, includePad=True, name=None):
		super().__init__(size, stride, pad, name)
		self.registerBlueprint(locals())

		self.mode = PoolMode.avgWithPad if includePad else PoolMode.avgNoPad


	def updateData(self, data):
		data = data.reshape(*data.shape[:2], 1, *data.shape[2:])
		self.data, self.workspace = poolNd(data, size=self.size, stride=self.stride, pad=self.pad, mode=self.mode,
										   test=not self.train)
		self.data = self.data.reshape(*self.data.shape[:2], *self.data.shape[3:])


	def updateGrad(self, grad):
		grad = grad.reshape(*grad.shape[:2], 1, *grad.shape[2:])

		indata = self.inData.reshape(*self.inData.shape[:2], 1, *self.inData.shape[2:])
		outdata = self.data.reshape(*self.data.shape[:2], 1, *self.data.shape[2:])

		self.grad = poolNdBackward(indata, outdata, grad, self.workspace, size=self.size, stride=self.stride,
								   pad=self.pad, mode=self.mode)
		self.grad = self.grad.reshape(*self.grad.shape[:2], *self.grad.shape[3:])


def unittest():
	batchsize, maps, insize = 2, 6, 5
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, insize).astype(np.float32))

	size = 3
	stride, pad = 2, 1

	avgpool1d = AvgPool1D(size=size, stride=stride, pad=pad, includePad=True)
	avgpool1d(data)

	hostData = np.zeros(shape=(batchsize, maps, insize + 2 * pad), dtype=np.float32)
	hostData[:, :, pad:-pad] = data.get()
	hostOutData = np.empty(avgpool1d.data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for x in range(hostOutData.shape[2]):
				hostOutData[b, c, x] = np.mean(hostData[b, c, x * stride:x * stride + size])

	assert np.allclose(hostOutData, avgpool1d.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*avgpool1d.data.shape).astype(np.float32))
	avgpool1d.backward(grad)

	hostGrad = grad.get()
	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for x in range(hostOutData.shape[2]):
				for dx in range(size):
					hostInGrad[b, c, x * stride+dx] += hostGrad[b, c, x] / size

	assert np.allclose(hostInGrad[:, :, pad:-pad], avgpool1d.grad.get())


if __name__ == "__main__":
	unittest()
