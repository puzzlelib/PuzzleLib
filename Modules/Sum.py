import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend import BlasGroup

from PuzzleLib.Modules.Module import ModuleError, Module


class Sum(Module):
	def __init__(self, axis, useWeights=True, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.useWeights = useWeights
		self.axis = axis

		self.v = None
		self.axisSize = None


	def updateData(self, batch):
		data, self.v = batch if self.useWeights else (batch, None)

		preAxis, postAxis = int(np.prod(data.shape[:self.axis])), int(np.prod(data.shape[self.axis + 1:]))
		self.axisSize = data.shape[self.axis]

		indata = data.reshape(preAxis, self.axisSize, postAxis)

		if self.useWeights:
			self.data = BlasGroup.mulTensorOnVecGroup(indata, self.v, formatT="gbp", transpT=True)
		else:
			self.data = BlasGroup.sumOnTensorGroup(indata, formatT="gbp", cols=True)

		self.data = self.data.reshape(*data.shape[:self.axis], *data.shape[self.axis + 1:])


	def updateGrad(self, grad):
		preAxis, postAxis = int(np.prod(grad.shape[:self.axis])), int(np.prod(grad.shape[self.axis:]))

		outgrad = grad.reshape(preAxis, 1, postAxis)
		wgrad = None

		if self.useWeights:
			v = self.v.reshape(preAxis, self.axisSize, 1)

			datagrad = BlasGroup.mulTensorBatch(v, outgrad, formatA="gbp", formatB="gbp", formatOut="gbp")
			wgrad = BlasGroup.mulTensorOnVecGroup(self.inData[0], grad, formatT="gbp")

		else:
			ones = gpuarray.empty(shape=(1, self.axisSize, 1), dtype=np.float32).fill(1.0)
			datagrad = BlasGroup.mulTensorBatch(ones, outgrad, formatA="gbp", formatB="gbp", formatOut="gbp")

		datagrad = datagrad.reshape(*grad.shape[:self.axis], self.axisSize, *grad.shape[self.axis:])
		self.grad = [datagrad, wgrad] if self.useWeights else datagrad


	def dataShapeFrom(self, shapes):
		shape = shapes[0] if self.useWeights else shapes
		return shape[:self.axis] + shape[self.axis + 1:]


	def gradShapeFrom(self, shape):
		inshape = shape[:self.axis] + (self.axisSize, ) + shape[self.axis + 1:]
		return [inshape, (self.axisSize, )] if self.useWeights else inshape


	def checkDataShape(self, shapes):
		if self.useWeights:
			shape, wshape = shapes

			if len(wshape) != self.axis + 1:
				raise ModuleError("Not enough dims in weights (%d were given, need at least %d)" %
								  (len(wshape), self.axis + 1))

			if shape[:self.axis + 1] != wshape:
				raise ModuleError("Inconsistency in data and weights shapes (%s with %s)" % (shape, wshape))

		else:
			shape = shapes

		if self.axis > len(shape) - 1:
			raise ModuleError("Not enough dims in data (%d were given, need at least %d)" % (len(shape), self.axis + 1))


	def checkGradShape(self, shape):
		if self.axis >= len(shape):
			raise ModuleError("Not enough dims in grad (%d were given, need at least %d)" % (len(shape), self.axis))

		if self.useWeights:
			if shape[:self.axis] != self.v.shape[:self.axis]:
				raise ModuleError("Inconsistency in grad and weights shapes (%s  with %s)" % (shape, self.v.shape))


	def reset(self):
		super().reset()

		self.v = None
		self.axisSize = None


def groupAxisTest():
	batchsize, groups, size = 5, 3, 4
	data = gpuarray.to_gpu(np.random.randn(batchsize, groups, size).astype(np.float32))

	summod = Sum(axis=1, useWeights=False)
	summod(data)

	hostOutData = np.sum(data.get(), axis=1)
	assert np.allclose(hostOutData, summod.data.get())

	grad = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))
	summod.backward(grad)

	hostInGrad = np.repeat(grad.get().reshape(batchsize, 1, size), repeats=groups, axis=1)
	assert np.allclose(hostInGrad, summod.grad.get())

	weights = gpuarray.to_gpu(np.random.randn(batchsize, groups).astype(np.float32))
	hostWeights = weights.get().reshape(*weights.shape, 1)

	summod = Sum(axis=1, useWeights=True)
	summod([data, weights])

	hostOutData = np.sum(data.get() * hostWeights, axis=1)
	assert np.allclose(hostOutData, summod.data.get())

	summod.backward(grad)

	hostInGrad = np.repeat(grad.get().reshape(batchsize, 1, size), repeats=groups, axis=1) * hostWeights
	assert np.allclose(hostInGrad, summod.grad[0].get())

	hostWGrad = np.sum(data.get() * grad.get().reshape(batchsize, 1, size), axis=2)
	assert np.allclose(hostWGrad, summod.grad[1].get())


def preLastAxisTest():
	batchsize, seqlen, groups, size = 5, 3, 2, 6

	data = gpuarray.to_gpu(np.random.randn(batchsize, seqlen, groups, size).astype(np.float32))

	summod = Sum(axis=2, useWeights=False)
	summod(data)

	hostOutData = np.sum(data.get(), axis=2)
	assert np.allclose(hostOutData, summod.data.get())

	grad = gpuarray.to_gpu(np.random.randn(batchsize, seqlen, size).astype(np.float32))
	summod.backward(grad)

	hostInGrad = np.repeat(grad.get().reshape(batchsize, seqlen, 1, size), repeats=groups, axis=2)
	assert np.allclose(hostInGrad, summod.grad.get())


def unittest():
	groupAxisTest()
	preLastAxisTest()


if __name__ == "__main__":
	unittest()
