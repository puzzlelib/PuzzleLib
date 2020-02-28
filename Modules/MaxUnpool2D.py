import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Kernels import Pool

from PuzzleLib.Modules.Module import ModuleError, Module
from PuzzleLib.Modules.MaxPool2D import MaxPool2D


class MaxUnpool2D(Module):
	def __init__(self, maxpool2d, name=None):
		super().__init__(name)
		self.registerBlueprint(locals(), exclude=["maxpool2d"])

		self.maxpool2d = maxpool2d
		self.maxpool2d.withMask = True


	def updateData(self, data):
		self.data = Pool.maxunpool2d(data, self.maxpool2d.inData.shape, self.maxpool2d.mask)


	def updateGrad(self, grad):
		self.grad = Pool.maxunpool2dBackward(grad, self.maxpool2d.data.shape, self.maxpool2d.mask)


	def dataShapeFrom(self, shape):
		batchsize, maps, inh, inw = shape

		hsize, wsize = self.maxpool2d.size
		padh, padw = self.maxpool2d.pad
		hstride, wstride = self.maxpool2d.stride

		outh = (inh - 1) * hstride - 2 * padh + hsize
		outw = (inw - 1) * wstride - 2 * padw + wsize

		return batchsize, maps, outh, outw


	def checkDataShape(self, shape):
		if shape != self.maxpool2d.mask.shape:
			raise ModuleError("Data shape (current %s) must be equal to connected MaxPool2D mask shape (%s)" %
							  (shape, self.maxpool2d.mask.shape))


	def gradShapeFrom(self, shape):
		batchsize, maps, outh, outw = shape

		hsize, wsize = self.maxpool2d.size
		padh, padw = self.maxpool2d.pad
		hstride, wstride = self.maxpool2d.stride

		inh = (outh + 2 * padh - hsize) // hstride + 1
		inw = (outw + 2 * padw - wsize) // wstride + 1

		return batchsize, maps, inh, inw


	def checkGradShape(self, shape):
		if shape != self.maxpool2d.inData.shape:
			raise ModuleError("Grad shape (current %s) must be equal to connected MaxPool2D data shape (%s)" %
							  (shape, self.maxpool2d.inData.shape))


def unittest():
	batchsize, maps, h, w = 15, 3, 4, 5
	indata = gpuarray.to_gpu(np.random.randn(batchsize, maps, h, w).astype(np.float32))

	maxpool2d = MaxPool2D()
	maxunpool2d = MaxUnpool2D(maxpool2d)

	maxpool2d(indata)

	data = gpuarray.to_gpu(np.random.randn(*maxpool2d.data.shape).astype(np.float32))
	maxunpool2d(data)

	hostPoolData = data.get()
	hostMask = maxpool2d.mask.get()

	hostOutData = np.zeros(maxpool2d.inData.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(maxpool2d.data.shape[2]):
				for x in range(maxpool2d.data.shape[3]):
					maxidx = hostMask[b, c, y, x]
					hostOutData[b, c].ravel()[maxidx] = hostPoolData[b, c, y, x]

	assert np.allclose(hostOutData, maxunpool2d.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*maxunpool2d.data.shape).astype(np.float32))
	maxunpool2d.backward(grad)

	hostGrad = grad.get()
	hostInGrad = np.empty(maxunpool2d.grad.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(maxpool2d.data.shape[2]):
				for x in range(maxpool2d.data.shape[3]):
					maxidx = hostMask[b, c, y, x]
					hostInGrad[b, c, y, x] = hostGrad[b, c].ravel()[maxidx]

	assert np.allclose(hostInGrad, maxunpool2d.grad.get())


if __name__ == "__main__":
	unittest()
