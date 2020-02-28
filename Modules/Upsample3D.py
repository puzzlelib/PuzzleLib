import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Kernels import Upsample

from PuzzleLib.Modules.Module import ModuleError, Module
from PuzzleLib.Modules.Upsample2D import UpsampleMode


class Upsample3D(Module):
	def __init__(self, scale=2, mode="nearest", name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.scale = scale
		self.mode = UpsampleMode(mode)


	def updateData(self, data):
		self.data = Upsample.upsample3d(data, self.scale, mode=self.mode.value)


	def updateGrad(self, grad):
		self.grad = Upsample.upsample3dBackward(grad, self.scale, mode=self.mode.value)


	def checkDataShape(self, shape):
		if len(shape) != 5:
			raise ModuleError("Data must be 5d tensor")


	def checkGradShape(self, shape):
		if len(shape) != 5:
			raise ModuleError("Grad must be 5d tensor")

		_, _, d, h, w = shape
		if d % self.scale != 0 or h % self.scale != 0 or w % self.scale != 0:
			raise ModuleError("Grad map size is not divisible by scale %s (got %s, %s, %s)" % (self.scale, d, h, w))


	def dataShapeFrom(self, shape):
		batchsize, maps, d, h, w = shape
		return batchsize, maps, self.scale * d, self.scale * h, self.scale * w


	def gradShapeFrom(self, shape):
		batchsize, maps, d, h, w = shape
		return batchsize, maps, d // self.scale, h // self.scale, w // self.scale


def unittest():
	batchsize, maps, ind, inh, inw = 2, 2, 2, 2, 2
	scale = 2

	data = gpuarray.to_gpu(np.random.uniform(low=-1.0, high=1.0,
											 size=(batchsize, maps, ind, inh, inw)).astype(np.float32))

	upsample3d = Upsample3D(scale=scale, mode="nearest")
	upsample3d(data)

	hostData = data.get()
	hostOutData = np.empty(upsample3d.data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for z in range(ind):
				for y in range(inh):
					for x in range(inw):
						hostOutData[b, c, z * scale:(z+1) * scale, y * scale:(y+1) * scale, x * scale:(x+1) * scale] = \
							hostData[b, c, z, y, x]

	assert np.allclose(hostOutData, upsample3d.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*upsample3d.data.shape).astype(np.float32))
	upsample3d.backward(grad)

	hostGrad = grad.get()
	hostInGrad = np.zeros(data.shape, dtype=np.float32)
	for b in range(batchsize):
		for c in range(maps):
			for z in range(ind):
				for y in range(inh):
					for x in range(inw):
						for dz in range(scale):
							for dy in range(scale):
								for dx in range(scale):
									hostInGrad[b, c, z, y, x] += \
										hostGrad[b, c, z * scale + dz, y * scale + dy, x * scale + dx]

	assert np.allclose(hostInGrad, upsample3d.grad.get())


if __name__ == "__main__":
	unittest()
