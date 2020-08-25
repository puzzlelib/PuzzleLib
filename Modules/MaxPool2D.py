import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Kernels import Pool
from PuzzleLib.Backend.Dnn import PoolMode, poolNd, poolNdBackward

from PuzzleLib.Modules.Pool2D import Pool2D


class MaxPool2D(Pool2D):
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
		if self.useMask:
			self.data, self.mask = Pool.maxpool2d(data, size=self.size, stride=self.stride, pad=self.pad)
		else:
			test = not self.train
			self.data, self.workspace = poolNd(data, size=self.size, stride=self.stride, pad=self.pad,
											   mode=self.mode, test=test)


	def updateGrad(self, grad):
		if self.useMask:
			self.grad = Pool.maxpool2dBackward(grad, self.inData.shape, self.mask,
											   size=self.size, stride=self.stride, pad=self.pad)
		else:
			self.grad = poolNdBackward(self.inData, self.data, grad, self.workspace,
									   size=self.size, stride=self.stride, pad=self.pad, mode=self.mode)


	def reset(self):
		super().reset()
		self.mask = None


def unittest():
	batchsize, maps, h, w = 1, 1, 6, 6
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, h, w).astype(np.float32))

	maxpool2d = MaxPool2D()
	maxpool2d(data)

	grad = gpuarray.to_gpu(np.random.randn(*maxpool2d.data.shape).astype(np.float32))
	maxpool2d.backward(grad)

	def maxDownSample2d(dat, factor):
		trimrows = dat.shape[0] // factor * factor
		trimcols = dat.shape[1] // factor * factor

		maxSoFar = None
		first = True

		for coff in range(factor):
			for roff in range(factor):
				hopped = dat[roff:trimrows:factor, coff:trimcols:factor]
				if first:
					maxSoFar = hopped
					first = False
				else:
					maxSoFar = np.maximum(maxSoFar, hopped)

		return maxSoFar

	hostOutData = maxDownSample2d(data.get()[0, 0], 2)
	assert np.allclose(hostOutData, maxpool2d.data.get()[0, 0])


if __name__ == "__main__":
	unittest()
