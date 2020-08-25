from enum import Enum

import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.gpuarray import memoryPool as memPool
from PuzzleLib.Backend.Kernels import Pad

from PuzzleLib.Modules.Module import ModuleError, Module


class PadMode(str, Enum):
	constant = "constant"
	reflect = "reflect"


class Pad2D(Module):
	def __init__(self, pad, mode="constant", fillValue=None, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.mode = PadMode(mode)

		self.pad = self.repeat(pad, 4)
		self.fillValue = 0 if fillValue is None else fillValue


	def updateData(self, data):
		if self.mode == PadMode.constant:
			inh, inw = data.shape[2:]
			upad, bpad, lpad, rpad = self.pad

			outh, outw = inh + upad + bpad, inw + lpad + rpad
			self.data = gpuarray.empty(data.shape[:2] + (outh, outw), dtype=np.float32, allocator=memPool)

			self.data.fill(self.fillValue)
			self.data[:, :, upad:self.data.shape[2] - bpad, lpad:self.data.shape[3] - rpad] = data

		elif self.mode == PadMode.reflect:
			self.data = Pad.reflectpad2d(data, self.pad)

		else:
			raise NotImplementedError(self.mode)


	def updateGrad(self, grad):
		if self.mode == PadMode.constant:
			height, width = grad.shape[2:]
			upad, bpad, lpad, rpad = self.pad

			self.grad = grad[:, :, upad:height - bpad, lpad:width - rpad].copy(allocator=memPool)

		elif self.mode == PadMode.reflect:
			self.grad = Pad.reflectpad2dBackward(grad, self.pad)

		else:
			raise NotImplementedError(self.mode)


	def checkDataShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Data must be 4d tensor")


	def checkGradShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Grad must be 4d tensor")

		upad, bpad, lpad, rpad = self.pad
		height, width = shape[2:]

		if height < upad + bpad + 1:
			raise ModuleError("Grad maps height is too small (got %d, expected >= %d)" % (height, upad + bpad + 1))

		if width < lpad + rpad + 1:
			raise ModuleError("Grad maps width is too small (got %d, expected >= %d)" % (width, lpad + rpad + 1))


	def dataShapeFrom(self, shape):
		batchsize, maps, inh, inw = shape
		upad, bpad, lpad, rpad = self.pad

		return batchsize, maps, inh + upad + bpad, inw + lpad + rpad


	def gradShapeFrom(self, shape):
		batchsize, maps, outh, outw = shape
		upad, bpad, lpad, rpad = self.pad

		return batchsize, maps, outh - upad - bpad, outw - lpad - rpad


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in gpuarray.dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	constantTest()
	reflectTest()


def constantTest():
	data = gpuarray.to_gpu(np.random.randn(3, 4, 5, 6).astype(np.float32))

	upad, bpad, lpad, rpad = 0, 1, 0, 1
	fillValue = -0.1

	padmod = Pad2D(pad=(upad, bpad, lpad, rpad), fillValue=fillValue)
	padmod(data)

	assert padmod.dataShapeFrom(data.shape) == padmod.data.shape

	hostData, hostOutData = data.get(), padmod.data.get()
	assert np.allclose(hostOutData[:, :, upad:hostOutData.shape[2] - bpad, lpad:hostOutData.shape[3] - rpad], hostData)

	assert np.isclose(hostOutData[0, 0, hostOutData.shape[2] - 1, hostOutData.shape[3] - 1], fillValue)

	grad = gpuarray.to_gpu(np.random.randn(*hostOutData.shape).astype(np.float32))
	padmod.backward(grad)

	assert padmod.gradShapeFrom(grad.shape) == data.shape
	assert np.allclose(padmod.grad.get(), grad.get()[:, :, upad:grad.shape[2] - bpad, lpad:grad.shape[3] - rpad])


def reflectTest():
	batchsize, maps, inh, inw = 4, 8, 12, 15
	upad, bpad, lpad, rpad = 2, 3, 2, 3

	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, inh, inw).astype(np.float32))

	reflectpad = Pad2D(pad=(upad, bpad, lpad, rpad), mode=PadMode.reflect)
	reflectpad(data)

	hostData, hostOutData = data.get(), reflectpad.data.get()

	assert np.allclose(hostOutData[:, :, upad:inh + upad, lpad:inw + lpad], hostData)
	assert np.allclose(hostOutData[:, :, :upad, :lpad][:, :, ::-1, ::-1], hostData[:, :, 1:upad + 1, 1:lpad + 1])
	assert np.allclose(
		hostOutData[:, :, inh + upad:, inw + lpad:][:, :, ::-1, ::-1],
		hostData[:, :, inh - 1 - bpad:inh - 1, inw - 1 - rpad:inw - 1]
	)

	outh, outw = hostOutData.shape[2:]

	grad = gpuarray.to_gpu(np.random.randn(batchsize, maps, outh, outw).astype(np.float32))
	reflectpad.backward(grad)

	hostGrad, hostInGrad = grad.get(), reflectpad.grad.get()

	assert np.allclose(
		hostInGrad[:, :, upad + 1:inh - bpad - 1, lpad + 1:inw - rpad - 1],
		hostGrad[:, :, 2 * upad + 1:outh - 2 * bpad - 1, 2 * lpad + 1:outw - 2 * rpad - 1]
	)
	assert np.allclose(
		hostInGrad[:, :, 1:upad + 1, 1:lpad + 1],
		hostGrad[:, :, :upad, :lpad][:, :, ::-1, ::-1] +
		hostGrad[:, :, upad + 1:2 * upad + 1, lpad + 1:2 * lpad + 1] +
		hostGrad[:, :, :upad, lpad + 1:2 * lpad + 1][:, :, ::-1, :] +
		hostGrad[:, :, upad + 1:2 * upad + 1, :lpad][:, :, :, ::-1]
	)
	assert np.allclose(
		hostInGrad[:, :, inh - bpad - 1:inh - 1, inw - rpad - 1:inw - 1],
		hostGrad[:, :, outh - bpad:, outw - rpad:][:, :, ::-1, ::-1] +
		hostGrad[:, :, outh - 2 * bpad - 1:outh - bpad - 1, outw - 2 * rpad - 1:outw - rpad - 1] +
		hostGrad[:, :, outh - bpad:, outw - 2 * rpad - 1:outw - rpad - 1][:, :, ::-1, :] +
		hostGrad[:, :, outh - 2 * bpad - 1:outh - bpad - 1, outw - rpad:][:, :, :, ::-1]
	)


if __name__ == "__main__":
	unittest()
