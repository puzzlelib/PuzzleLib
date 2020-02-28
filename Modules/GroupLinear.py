from enum import Enum

import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import copy
from PuzzleLib.Backend.Kernels.MatVec import addVecToMat
from PuzzleLib.Backend.Kernels.MatVecBatch import addVecToMatBatch
from PuzzleLib.Backend import Blas, BlasGroup

from PuzzleLib.Variable import Variable
from PuzzleLib.Modules.Module import ModuleError, Module


class GroupMode(str, Enum):
	full = "full"
	one = "one"


class GroupLinear(Module):
	def __init__(self, groups, insize, outsize, wscale=1.0, useW=True, useBias=True, initscheme=None,
				 inmode="full", wmode="full", batchDim=0, name=None, empty=False, transpW=False):
		super().__init__(name)
		self.registerBlueprint(locals())

		if not(useW or useBias):
			raise ModuleError("Not using W and bias is not supported")

		self.transpW = transpW
		self.useW = useW
		self.useBias = useBias

		self.inmode = GroupMode(inmode)
		self.wmode = GroupMode(wmode)

		if batchDim == 0:
			self.format = "bgp"
		elif batchDim == 1:
			self.format = "gbp"
		else:
			raise ModuleError("Unsupported batch dimension")

		self.groupDim = 1 if batchDim == 0 else 0
		self.groups = 1 if groups is None else groups

		self.W = None
		self.b = None

		if empty:
			return

		self.setupW(insize, outsize, initscheme, wscale)
		self.setupBias(insize, outsize)


	def setupW(self, insize, outsize, initscheme, wscale):
		if not self.useW:
			return

		asize, bsize = (outsize, insize) if self.transpW else (insize, outsize)
		groups = self.groups if self.wmode == GroupMode.full else 1

		Wshape = (groups, asize, bsize)

		W = self.createTensorWithScheme(initscheme, Wshape, wscale, self.calcNeuronsNumber(Wshape, self.transpW))
		W = gpuarray.empty(Wshape, dtype=np.float32) if W is None else gpuarray.to_gpu(W)

		self.setVar("W", Variable(W))


	def setupBias(self, insize, outsize):
		if not self.useBias:
			return

		size = outsize if self.useW else insize
		bshape = (self.groups, size) if self.wmode == GroupMode.full else (1, size)

		self.setVar("b", Variable(gpuarray.zeros(bshape, dtype=np.float32)))


	def updateData(self, data):
		if self.useW:
			self.data = BlasGroup.mulTensorBatch(
				data, self.W, formatA=self.format, formatB="gbp", transpB=self.transpW, formatOut=self.format
			)
		else:
			self.data = copy(None, data)

		if self.useBias:
			if self.groupDim == 1:
				b = self.b.reshape(int(np.prod(self.b.shape)))
				outdata = self.data.reshape(self.data.shape[0], int(np.prod(self.data.shape[1:])))

				addVecToMat(b, outdata, axis=1, out=outdata)

			else:
				addVecToMatBatch(self.b, self.data, axis=1, out=self.data)


	def updateGrad(self, grad):
		if self.useW:
			formatOut = self.format if self.inmode == GroupMode.full else "gbp"

			self.grad = BlasGroup.mulTensorBatch(
				grad, self.W, formatA=self.format, formatB="gbp", transpB=not self.transpW, formatOut=formatOut
			)

			if self.inmode != GroupMode.full:
				self.grad = Blas.sumOnMatrix(self.grad.reshape(self.groups, grad.shape[0] * self.W.shape[1]))
				self.grad = self.grad.reshape(grad.shape[0], 1, self.W.shape[1])

		else:
			self.grad = grad


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		if self.wmode == GroupMode.full:
			if self.useW:
				A, B = (grad, self.inData) if self. transpW else (self.inData, grad)

				BlasGroup.mulTensorBatch(
					A, B, out=self.vars["W"].grad, formatA=self.format, formatB=self.format,
					formatOut="gbp", transpA=True, alpha=scale, beta=momentum
				)

			if self.useBias:
				BlasGroup.sumOnTensorGroup(grad, out=self.vars["b"].grad, formatT=self.format)

		else:
			if self.useW:
				A, B = (grad, self.inData) if self.transpW else (self.inData, grad)

				wgrad = BlasGroup.mulTensorBatch(
					A, B, transpA=True, formatA=self.format, formatB=self.format, formatOut="gbp",
					alpha=scale, beta=momentum
				)

				Blas.sumOnMatrix(wgrad.reshape(wgrad.shape[0], -1), out=self.vars["W"].grad.ravel())

			if self.useBias:
				Blas.sumOnMatrix(grad.reshape(grad.shape[0] * grad.shape[1], grad.shape[2]), out=self.vars["b"].grad[0])


	def dataShapeFrom(self, shape):
		groups = shape[self.groupDim] if self.inmode == GroupMode.full else self.groups
		beg = (shape[0], groups) if self.groupDim == 1 else (groups, shape[1])

		if self.useW:
			return beg + (self.W.shape[1], ) if self.transpW else beg + (self.W.shape[2], )
		else:
			return beg + (shape[2], )


	def checkDataShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Data must be 3d tensor")

		if self.inmode == GroupMode.one and shape[1] != 1:
			raise ModuleError("Expected 1 group in data, %d were given" % (shape[1]))

		if self.inmode != GroupMode.one and self.wmode != GroupMode.one and shape[self.groupDim] != self.groups:
			raise ModuleError("Expected %d groups in data, %d were given" % (self.groups, shape[self.groupDim]))

		if self.useW:
			if self.transpW and shape[2] != self.W.shape[2]:
				raise ModuleError("Expected %d data dimensions, %d were given" % (self.W.shape[2], shape[2]))

			elif shape[2] != self.W.shape[1]:
				raise ModuleError("Expected %d data dimensions, %d were given" % (self.W.shape[1], shape[2]))


	def gradShapeFrom(self, shape):
		beg = (shape[0], self.groups) if self.groupDim == 1 else (self.groups, shape[1])
		onebeg = (shape[0], 1) if self.groupDim == 1 else (1, shape[1])

		if self.useW:
			size = self.W.shape[2 if self.transpW else 1]
			return beg + (size, ) if self.inmode == GroupMode.full else onebeg + (size, )

		else:
			return beg + (shape[2], ) if self.inmode == GroupMode.full else onebeg + (shape[2], )


	def checkGradShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Grad must be 3d tensor")

		if self.wmode == GroupMode.full and shape[self.groupDim] != self.groups:
			raise ModuleError("Expected %d groups in grad, %d were given" % (self.groups, shape[self.groupDim]))

		if self.useW:
			if self.transpW and shape[2] != self.W.shape[1]:
				raise ModuleError("Expected %d grad dimensions, %d were given" % (self.W.shape[1], shape[2]))

			elif shape[2] != self.W.shape[2]:
				raise ModuleError("Expected %d grad dimensions, %d were given" % (self.W.shape[2], shape[2]))


	@classmethod
	def calcNeuronsNumber(cls, shape, transpose=False):
		shape = shape[1:]
		return super().calcNeuronsNumber(shape, transpose)


def unittest():
	stdCalcTest()
	oneInCalcTest()
	oneWCalcTest()
	batchDimTest()
	trainTest()


def stdCalcTest():
	groups, insize, outsize = 2, 5, 4
	batchsize = 3

	data = gpuarray.to_gpu(np.random.randn(batchsize, groups, insize).astype(np.float32))

	grpLinear = GroupLinear(groups, insize, outsize)
	grpLinear.b.fill(0.5)
	grpLinear(data)

	hostOutData = np.empty(grpLinear.data.shape, dtype=np.float32)
	for i in range(groups):
		hostOutData[:, i, :] = np.dot(data.get()[:, i, :], grpLinear.W.get()[i])
	hostOutData += grpLinear.b.get()

	assert np.allclose(hostOutData, grpLinear.data.get())

	grad = gpuarray.to_gpu(np.random.randn(batchsize, groups, outsize).astype(np.float32))
	grpLinear.backward(grad)

	hostInGrad = np.empty(grpLinear.grad.shape, dtype=np.float32)
	for i in range(groups):
		hostInGrad[:, i, :] = np.dot(grad.get()[:, i, :], grpLinear.W.get()[i].T)

	assert np.allclose(hostInGrad, grpLinear.grad.get())

	hostWGrad = np.empty(grpLinear.W.shape, dtype=np.float32)
	for i in range(groups):
		hostWGrad[i] = np.dot(data.get()[:, i, :].T, grad.get()[:, i, :])

	hostBGrad = np.empty(grpLinear.b.shape, dtype=np.float32)
	for i in range(groups):
		hostBGrad[i] = np.sum(grad.get()[:, i, :], axis=0)

	assert np.allclose(hostWGrad, grpLinear.vars["W"].grad.get())
	assert np.allclose(hostBGrad, grpLinear.vars["b"].grad.get())


def oneInCalcTest():
	batchsize, insize, outsize = 4, 5, 3
	groups = 4

	data = gpuarray.to_gpu(np.random.randn(batchsize, 1, insize).astype(np.float32))

	grpLinear = GroupLinear(groups, insize, outsize, inmode="one")
	grpLinear(data)

	hostOutData = np.empty(grpLinear.data.shape, dtype=np.float32)
	for i in range(groups):
		hostOutData[:, i, :] = np.dot(data.get()[:, 0, :], grpLinear.W.get()[i])
	hostOutData += grpLinear.b.get()[np.newaxis, :, :]

	assert np.allclose(hostOutData, grpLinear.data.get())

	grad = gpuarray.to_gpu(np.random.randn(batchsize, groups, outsize).astype(np.float32))
	grpLinear.backward(grad)

	hostInGrad = np.zeros(data.shape, dtype=np.float32)
	for i in range(groups):
		hostInGrad[:, 0, :] += np.dot(grad.get()[:, i, :], grpLinear.W.get()[i].T)

	assert np.allclose(hostInGrad, grpLinear.grad.get())

	hostWGrad = np.empty(grpLinear.W.shape, dtype=np.float32)
	for i in range(groups):
		hostWGrad[i] = np.dot(data.get()[:, 0, :].T, grad.get()[:, i, :])

	hostBGrad = np.empty(grpLinear.b.shape, dtype=np.float32)
	for i in range(groups):
		hostBGrad[i] = np.sum(grad.get()[:, i, :], axis=0)

	assert np.allclose(hostWGrad, grpLinear.vars["W"].grad.get())
	assert np.allclose(hostBGrad, grpLinear.vars["b"].grad.get())


def oneWCalcTest():
	batchsize, insize, outsize = 4, 3, 4
	groups = 3

	data = gpuarray.to_gpu(np.random.randn(batchsize, groups, insize).astype(np.float32))

	grpLinear = GroupLinear(None, insize, outsize, wmode="one")
	grpLinear(data)

	hostOutData = np.empty(grpLinear.data.shape, dtype=np.float32)
	for i in range(groups):
		hostOutData[:, i, :] = np.dot(data.get()[:, i, :], grpLinear.W.get()[0])
	hostOutData += grpLinear.b.get()[np.newaxis, :, :]

	assert np.allclose(hostOutData, grpLinear.data.get())

	grad = gpuarray.to_gpu(np.random.randn(batchsize, groups, outsize).astype(np.float32))
	grpLinear.backward(grad)

	hostInGrad = np.empty(grpLinear.grad.shape, dtype=np.float32)
	for i in range(groups):
		hostInGrad[:, i, :] = np.dot(grad.get()[:, i, :], grpLinear.W.get()[0].T)

	assert np.allclose(hostInGrad, grpLinear.grad.get())

	hostWGrad = np.zeros(grpLinear.W.shape, dtype=np.float32)
	for i in range(groups):
		hostWGrad += np.dot(data.get()[:, i, :].T, grad.get()[:, i, :])

	hostBGrad = np.sum(grad.get().reshape(batchsize * groups, outsize), axis=0)

	assert np.allclose(hostWGrad, grpLinear.vars["W"].grad.get())
	assert np.allclose(hostBGrad, grpLinear.vars["b"].grad.get())


def batchDimTest():
	groups, insize, outsize = 2, 5, 4
	batchsize = 3

	data = gpuarray.to_gpu(np.random.randn(groups, batchsize, insize).astype(np.float32))

	grpLinear = GroupLinear(groups, insize, outsize, batchDim=1)
	grpLinear.b.fill(0.5)
	grpLinear(data)

	hostOutData = np.empty(grpLinear.data.shape, dtype=np.float32)
	for i in range(groups):
		hostOutData[i] = np.dot(data.get()[i], grpLinear.W.get()[i])

	for i in range(batchsize):
		hostOutData[:, i, :] += grpLinear.b.get()

	assert np.allclose(hostOutData, grpLinear.data.get())

	grad = gpuarray.to_gpu(np.random.randn(groups, batchsize, outsize).astype(np.float32))
	grpLinear.backward(grad)

	hostInGrad = np.empty(grpLinear.grad.shape, dtype=np.float32)
	for i in range(groups):
		hostInGrad[i] = np.dot(grad.get()[i], grpLinear.W.get()[i].T)

	assert np.allclose(hostInGrad, grpLinear.grad.get())

	hostWGrad = np.empty(grpLinear.W.shape, dtype=np.float32)
	for i in range(groups):
		hostWGrad[i] = np.dot(data.get()[i].T, grad.get()[i])

	hostBGrad = np.empty(grpLinear.b.shape, dtype=np.float32)
	for i in range(groups):
		hostBGrad[i] = np.sum(grad.get()[i], axis=0)

	assert np.allclose(hostWGrad, grpLinear.vars["W"].grad.get())
	assert np.allclose(hostBGrad, grpLinear.vars["b"].grad.get())


def trainTest():
	groups, insize, outsize = 16, 128, 32
	batchsize = 32

	data = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, groups, insize)).astype(np.float32))
	target = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, groups, outsize)).astype(np.float32))

	grpLinear = GroupLinear(groups, insize, outsize)

	from PuzzleLib.Cost.MSE import MSE
	mse = MSE()

	for i in range(100):
		learnRate = 1e-1

		grpLinear(data)
		error, grad = mse(grpLinear.data, target)

		grpLinear.backward(grad)
		grpLinear.updateParams(learnRate)

		if (i + 1) % 5 == 0:
			print("Iteration #%d error: %s" % (i + 1, error))


if __name__ == "__main__":
	unittest()
