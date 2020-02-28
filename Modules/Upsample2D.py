from enum import Enum

import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Kernels import Upsample

from PuzzleLib.Modules.Module import ModuleError, Module


class UpsampleMode(str, Enum):
	nearest = "nearest"
	linear = "linear"


class Upsample2D(Module):
	def __init__(self, scale=2, mode="nearest", name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.scale = scale
		self.mode = UpsampleMode(mode)


	def updateData(self, data):
		self.data = Upsample.upsample2d(data, self.scale, mode=self.mode.value)


	def updateGrad(self, grad):
		self.grad = Upsample.upsample2dBackward(grad, self.scale, mode=self.mode.value)


	def checkDataShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Data must be 4d tensor")


	def checkGradShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Grad must be 4d tensor")

		_, _, h, w = shape
		if h % self.scale != 0 or w % self.scale != 0:
			raise ModuleError("Grad map size is not divisible by scale %s" % self.scale)


	def dataShapeFrom(self, shape):
		batchsize, maps, h, w = shape
		return batchsize, maps, self.scale * h, self.scale * w


	def gradShapeFrom(self, shape):
		batchsize, maps, h, w = shape
		return batchsize, maps, h // self.scale, w // self.scale


def unittest():
	batchsize, maps, inh, inw = 3, 4, 5, 6
	scale = 2

	data = gpuarray.to_gpu(np.random.uniform(low=-1.0, high=1.0, size=(batchsize, maps, inh, inw)).astype(np.float32))

	upsample2d = Upsample2D(scale=scale, mode="nearest")
	upsample2d(data)

	hostData = data.get()
	hostOutData = np.empty(upsample2d.data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(inh):
				for x in range(inw):
					hostOutData[b, c, y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = hostData[b, c, y, x]

	assert np.allclose(hostOutData, upsample2d.data.get())

	grad = gpuarray.to_gpu(np.random.uniform(low=-1.0, high=1.0, size=upsample2d.data.shape).astype(np.float32))
	upsample2d.backward(grad)

	hostGrad = grad.get()
	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(inh):
				for x in range(inw):
					for dy in range(scale):
						for dx in range(scale):
							hostInGrad[b, c, y, x] += hostGrad[b, c, y * scale + dy, x * scale + dx]

	assert np.allclose(hostInGrad, upsample2d.grad.get())


if __name__ == "__main__":
	unittest()
