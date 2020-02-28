import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import dtypesSupported, globalRng, copy, memoryPool as memPool
from PuzzleLib.Backend.Kernels.ElementWise import dropout2dKer

from PuzzleLib.Modules.Dropout import Dropout


class Dropout2D(Dropout):
	def __init__(self, p=0.5, rng=globalRng, slicing=None, inplace=False, name=None):
		super().__init__(p, rng, slicing, inplace, name)
		self.mapsize = None


	def updateData(self, data):
		if self.train:
			if self.inplace:
				self.data = data
			else:
				if self.slice is not None:
					self.data = copy(None, data)
				else:
					self.data = gpuarray.empty(data.shape, dtype=data.dtype, allocator=memPool)

			batchsize, maps, height, width = data.shape
			self.mapsize = height * width

			parttype = {
				np.float32: np.uint32,
				np.float16: np.uint16
			}[data.dtype.type]

			intsize = np.dtype(np.uint32).itemsize
			itemsize = np.dtype(parttype).itemsize

			nbytes = (batchsize * maps * itemsize + intsize - 1) // intsize * intsize
			self.rands = gpuarray.empty((nbytes // itemsize, ), dtype=parttype, allocator=memPool)

			self.rng.fillInteger(self.rands.view(np.uint32))

			p = 1.0 - self.p
			self.partition = int(p * np.iinfo(parttype).max)

			dropout2dKer(data.dtype)(self.data, data, self.rands, self.partition, p, self.mapsize, slice=self.slice)

		else:
			self.data = data


	def updateGrad(self, grad):
		if self.train:
			if self.inplace:
				self.grad = grad
			else:
				if self.slice is not None:
					self.grad = copy(None, grad)
				else:
					self.grad = gpuarray.empty(grad.shape, dtype=grad.dtype, allocator=memPool)

			dropout2dKer(grad.dtype)(self.grad, grad, self.rands, self.partition, 1.0 - self.p, self.mapsize)

		else:
			self.grad = grad


def unittest():
	for dtype, _ in dtypesSupported():
		dropout2dTest(dtype)


def dropout2dTest(dtype):
	batchsize, maps, height, width = 11, 13, 4, 3

	hostData = np.random.randn(batchsize, maps, height, width).astype(dtype)
	data = gpuarray.to_gpu(hostData)

	dropout2d = Dropout2D()
	dropout2d.calcMode(dtype)

	dropout2d(data)

	hostRands = dropout2d.rands.get()[:batchsize * maps].reshape(batchsize, maps)[:, :, np.newaxis, np.newaxis]

	hostOutData = hostData * (hostRands < dropout2d.partition) / (1.0 - dropout2d.p)
	assert np.allclose(hostOutData, dropout2d.data.get())

	hostGrad = np.random.randn(*dropout2d.data.shape).astype(dtype)
	grad = gpuarray.to_gpu(hostGrad)

	dropout2d.backward(grad)

	hostInGrad = hostGrad * (hostRands < dropout2d.partition) / (1.0 - dropout2d.p)
	assert np.allclose(hostInGrad, dropout2d.grad.get())

	dropout2d.evalMode()
	dropout2d(data)

	hostOutData = hostData
	assert np.allclose(hostOutData, dropout2d.data.get())


if __name__ == "__main__":
	unittest()
