import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import dtypesSupported, globalRng, copy, memoryPool as memPool
from PuzzleLib.Backend.Kernels.ElementWise import dropoutKer

from PuzzleLib.Modules.Module import ModuleError, Module


class Dropout(Module):
	def __init__(self, p=0.5, rng=globalRng, slicing=None, inplace=False, name=None):
		super().__init__(name)
		self.registerBlueprint(locals(), exclude=["rng"])

		if rng is None:
			rng = globalRng

		self.p = p
		self.partition = None

		self.rng = rng
		self.rands = None

		self.slice = slicing

		self.inplace = inplace
		if inplace and Config.showWarnings:
			print("[%s] Warning: %s is using inplace flag" % (Config.libname, self))


	def updateData(self, data):
		if self.train:
			if self.inplace:
				self.data = data
			else:
				if self.slice is not None:
					self.data = copy(None, data)
				else:
					self.data = gpuarray.empty(data.shape, dtype=data.dtype, allocator=memPool)

			parttype = {
				np.float32: np.uint32,
				np.float16: np.uint16
			}[data.dtype.type]

			intsize = np.dtype(np.uint32).itemsize

			nbytes = (data.nbytes + intsize - 1) // intsize * intsize
			self.rands = gpuarray.empty((nbytes // np.dtype(parttype).itemsize, ), dtype=parttype, allocator=memPool)

			self.rng.fillInteger(self.rands.view(np.uint32))

			p = 1.0 - self.p
			self.partition = int(p * np.iinfo(parttype).max)

			dropoutKer(data.dtype)(self.data, data, self.rands, self.partition, np.float32(p), slice=self.slice)

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

			dropoutKer(grad.dtype)(self.grad, grad, self.rands, self.partition, 1.0 - self.p, slice=self.slice)

		else:
			self.grad = grad



	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return shape


	def reset(self):
		super().reset()
		self.rands = None


	def calcMode(self, T):
		if Config.backend == Config.Backend.cuda:
			if T not in {np.float16, np.float32}:
				raise ModuleError("Unsupported dtype %s" % T)

		elif T != np.float32:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, _ in dtypesSupported():
		dropoutTest(dtype)


def dropoutTest(dtype):
	hostData = np.random.randn(11, 13, 4, 3).astype(dtype)
	data = gpuarray.to_gpu(hostData)

	dropout = Dropout()
	dropout.calcMode(dtype)

	dropout(data)

	hostRands = dropout.rands.get()[:data.size].reshape(data.shape)

	hostOutData = hostData * (hostRands < dropout.partition) / (1.0 - dropout.p)
	assert np.allclose(hostOutData, dropout.data.get())

	hostGrad = np.random.randn(*dropout.data.shape).astype(dtype)
	grad = gpuarray.to_gpu(hostGrad)

	dropout.backward(grad)

	hostInGrad = hostGrad * (hostRands < dropout.partition) / (1.0 - dropout.p)
	assert np.allclose(hostInGrad, dropout.grad.get())

	dropout.evalMode()
	dropout(data)

	hostOutData = hostData
	assert np.allclose(hostOutData, dropout.data.get())


if __name__ == "__main__":
	unittest()
