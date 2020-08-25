import numpy as np

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.Dnn import PoolMode, poolNd, poolNdBackward

from PuzzleLib.Modules.Module import ModuleError, Module


class SubtractMean(Module):
	def __init__(self, size=5, includePad=True, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		if size % 2 != 1 or size == 1:
			raise ModuleError("Subtractive norm size must be odd and > 1")

		self.size = self.repeat(size, 2)
		self.pad = (self.size[0] // 2, self.size[1] // 2)

		self.mode = PoolMode.avgWithPad if includePad else PoolMode.avgNoPad

		self.means = None
		self.workspace = None


	def updateData(self, data):
		self.means, self.workspace = poolNd(
			data, size=self.size, stride=1, pad=self.pad, mode=self.mode, test=not self.train
		)
		self.data = Blas.addVectorToVector(data.ravel(), self.means.ravel(), beta=-1.0).reshape(*data.shape)


	def updateGrad(self, grad):
		meansGrad = poolNdBackward(
			self.inData, self.means, grad, self.workspace, size=self.size, stride=1, pad=self.pad, mode=self.mode
		)

		Blas.addVectorToVector(grad.ravel(), meansGrad.ravel(), out=meansGrad.ravel(), beta=-1.0)
		self.grad = meansGrad


	def dataShapeFrom(self, shape):
		return shape


	def checkDataShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Data must be 4d tensor")


	def gradShapeFrom(self, shape):
		return shape


	def checkGradShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Grad must be 4d tensor")


	def reset(self):
		super().reset()
		self.means = None
		self.workspace = None


def unittest():
	batchsize, maps, h, w = 1, 1, 6, 6
	size = 3

	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, h, w).astype(np.float32))

	subtractMean = SubtractMean(size=size)
	subtractMean(data)

	hpad, wpad = subtractMean.pad
	hostData = np.zeros(shape=(batchsize, maps, h + 2 * hpad, w + 2 * wpad), dtype=np.float32)
	hostData[:, :, hpad:-hpad, wpad:-wpad] = data.get()

	hostOutData = np.empty(subtractMean.data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(data.shape[2]):
				for x in range(data.shape[3]):
					hostOutData[b, c, y, x] -= np.sum(hostData[b, c, y:y + size, x:x + size]) / size**2

	assert np.allclose(hostOutData, subtractMean.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*subtractMean.data.shape).astype(np.float32))
	subtractMean.backward(grad)

	hostGrad = grad.get()
	hostInGrad = np.zeros(shape=hostData.shape, dtype=np.float32)
	hostInGrad[:, :, hpad:-hpad, wpad:-wpad] = hostGrad

	for b in range(batchsize):
		for c in range(maps):
			for y in range(hostGrad.shape[2]):
				for x in range(hostGrad.shape[3]):
					for dy in range(size):
						for dx in range(size):
							hostInGrad[b, c, y + dy, x + dx] -= hostGrad[b, c, y, x] / size**2

	assert np.allclose(hostInGrad[:, :, hpad:-hpad, wpad:-wpad], subtractMean.grad.get())


if __name__ == "__main__":
	unittest()
