import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.gpuarray import memoryPool as memPool
from PuzzleLib.Backend.Kernels import Pad

from PuzzleLib.Modules.Module import ModuleError, Module
from PuzzleLib.Modules.Pad2D import PadMode


class Pad1D(Module):
	def __init__(self, pad, mode="constant", fillValue=None, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.mode = PadMode(mode)

		self.pad = self.repeat(pad, 2)
		self.fillValue = 0 if fillValue is None else fillValue


	def updateData(self, data):
		if self.mode == PadMode.constant:
			insize = data.shape[2]
			lpad, rpad = self.pad

			outsize = insize + lpad + rpad
			self.data = gpuarray.empty(data.shape[:2] + (outsize, ), dtype=np.float32, allocator=memPool)

			self.data.fill(self.fillValue)
			self.data[:, :, lpad:self.data.shape[2] - rpad] = data

		elif self.mode == PadMode.reflect:
			self.data = Pad.reflectpad1d(data, self.pad)

		else:
			raise NotImplementedError(self.mode)


	def updateGrad(self, grad):
		if self.mode == PadMode.constant:
			size = grad.shape[2]
			lpad, rpad = self.pad

			self.grad = grad[:, :, lpad:size - rpad].copy(allocator=memPool)

		elif self.mode == PadMode.reflect:
			self.grad = Pad.reflectpad1dBackward(grad, self.pad)

		else:
			raise NotImplementedError(self.mode)


	def checkDataShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Data must be 3d tensor")

		lpad, rpad = self.pad
		size = shape[2]

		pad = max(lpad, rpad)

		if size < pad + 1:
			raise ModuleError("Data maps size is too small (got %d, expected >= %d)" % (size, pad + 1))


	def checkGradShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Grad must be 3d tensor")

		lpad, rpad = self.pad
		size = shape[2]

		if size < lpad + rpad + 1:
			raise ModuleError("Grad maps size is too small (got %d, expected >= %d)" % (size, lpad + rpad + 1))


	def dataShapeFrom(self, shape):
		batchsize, maps, size = shape
		lpad, rpad = self.pad

		return batchsize, maps, size + lpad + rpad


	def gradShapeFrom(self, shape):
		batchsize, maps, size = shape
		lpad, rpad = self.pad

		return batchsize, maps, size - lpad - rpad


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in gpuarray.dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	constantTest()
	reflectTest()


def constantTest():
	data = gpuarray.to_gpu(np.random.randn(3, 4, 5).astype(np.float32))

	lpad, rpad = 0, 1
	fillValue = -0.1

	padmod = Pad1D(pad=(lpad, rpad), mode=PadMode.constant, fillValue=fillValue)
	padmod(data)

	assert padmod.dataShapeFrom(data.shape) == padmod.data.shape

	hostData, hostOutData = data.get(), padmod.data.get()
	assert np.allclose(hostOutData[:, :, lpad:hostOutData.shape[2] - rpad], hostData)

	assert np.isclose(hostOutData[0, 0, hostOutData.shape[2] - 1], fillValue)

	grad = gpuarray.to_gpu(np.random.randn(*hostOutData.shape).astype(np.float32))
	padmod.backward(grad)

	assert padmod.gradShapeFrom(grad.shape) == data.shape
	assert np.allclose(padmod.grad.get(), grad.get()[:, :, lpad:grad.shape[2] - rpad])


def reflectTest():
	batchsize, maps, insize = 4, 8, 48
	lpad, rpad = 2, 3

	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, insize).astype(np.float32))

	reflectpad = Pad1D(pad=(lpad, rpad), mode=PadMode.reflect)
	reflectpad(data)

	hostData, hostOutData = data.get(), reflectpad.data.get()
	outsize = hostOutData.shape[2]

	assert np.allclose(hostOutData[:, :, lpad:insize + lpad], hostData)
	assert np.allclose(hostOutData[:, :, :lpad][:, :, ::-1], hostData[:, :, 1:lpad+1])
	assert np.allclose(hostOutData[:, :, insize + lpad:][:, :, ::-1], hostData[:, :, insize - 1 - rpad:insize - 1])

	grad = gpuarray.to_gpu(np.random.randn(batchsize, maps, outsize).astype(np.float32))
	reflectpad.backward(grad)

	hostGrad, hostInGrad = grad.get(), reflectpad.grad.get()

	assert np.allclose(hostInGrad[:, :, lpad + 1:insize - rpad - 1],
					   hostGrad[:, :, 2 * lpad + 1:outsize - 2 * rpad - 1])
	assert np.allclose(hostInGrad[:, :, 1:lpad + 1], hostGrad[:, :, :lpad][:, :, ::-1] +
					   hostGrad[:, :, lpad + 1:2 * lpad + 1])
	assert np.allclose(hostInGrad[:, :, insize - rpad - 1:insize - 1], hostGrad[:, :, outsize - rpad:][:, :, ::-1] +
					   hostGrad[:, :, outsize - 2 * rpad - 1:outsize - rpad - 1])


if __name__ == "__main__":
	unittest()
