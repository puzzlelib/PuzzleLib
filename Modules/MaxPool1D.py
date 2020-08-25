import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Dnn import PoolMode, poolNd, poolNdBackward
from PuzzleLib.Backend.Kernels import Pool

from PuzzleLib.Modules.Pool1D import Pool1D


class MaxPool1D(Pool1D):
	def __init__(self, size=2, stride=2, pad=0, useMask=False, name=None):
		super().__init__(size, stride, pad, name)
		self.registerBlueprint(locals())

		self.useMask = useMask
		self.mask = None

		self.mode = PoolMode.max


	@property
	def withMask(self):
		return self.useMask


	@withMask.setter
	def withMask(self, val):
		self.useMask = val
		self.gradUsesOutData = False if val else True


	def updateData(self, data):
		data = data.reshape(*data.shape[:2], 1, *data.shape[2:])

		if self.useMask:
			self.data, self.mask = Pool.maxpool2d(data, size=self.size, stride=self.stride, pad=self.pad)
		else:
			self.data, self.workspace = poolNd(data, size=self.size, stride=self.stride, pad=self.pad,
											   test=not self.train, mode = self.mode)

		self.data = self.data.reshape(*self.data.shape[:2], *self.data.shape[3:])


	def updateGrad(self, grad):
		grad = grad.reshape(*grad.shape[:2], 1, *grad.shape[2:])

		indata = self.inData.reshape(*self.inData.shape[:2], 1, *self.inData.shape[2:])
		outdata = self.data.reshape(*self.data.shape[:2], 1, *self.data.shape[2:])

		if self.useMask:
			self.grad = Pool.maxpool2dBackward(grad, indata.shape, self.mask,
											   size=self.size, stride=self.stride, pad=self.pad)
		else:
			self.grad = poolNdBackward(indata, outdata, grad, self.workspace,
									   size=self.size, stride=self.stride, pad=self.pad, mode=self.mode)

		self.grad = self.grad.reshape(*self.grad.shape[:2], *self.grad.shape[3:])


	def reset(self):
		super().reset()
		self.mask = None


def unittest():
	batchsize, maps, insize = 1, 1, 6
	size, stride, pad = 3, 2, 1
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, insize).astype(np.float32))

	maxpool1d = MaxPool1D(size=size, stride=stride, pad=pad)
	maxpool1d(data)

	hostData = np.full(shape=(batchsize, maps, insize + 2 * pad), fill_value=np.finfo(np.float32).min, dtype=np.float32)
	hostData[:, :, pad:-pad] = data.get()
	hostOutData = np.empty(maxpool1d.data.shape)

	for b in range(batchsize):
		for c in range(maps):
			for x in range(hostOutData.shape[2]):
						hostOutData[b, c, x] = np.max(hostData[b, c, x * stride:x * stride + size])

	assert np.allclose(hostOutData, maxpool1d.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*maxpool1d.data.shape).astype(np.float32))
	maxpool1d.backward(grad)

	hostGrad = grad.get()
	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for x in range(hostOutData.shape[2]):
				for dx in range(size):
					if hostData[b, c, x * stride + dx] == hostOutData[b, c, x]:
						hostInGrad[b, c, x * stride + dx] += hostGrad[b, c, x]

	assert np.allclose(hostInGrad[:, :, pad:-pad], maxpool1d.grad.get())


if __name__ == "__main__":
	unittest()
