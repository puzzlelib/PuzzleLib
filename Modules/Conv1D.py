import numpy as np

from PuzzleLib import Config
from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Modules.ConvND import ConvND


class Conv1D(ConvND):
	def __init__(self, inmaps, outmaps, size, stride=1, pad=0, dilation=1, wscale=1.0, useBias=True,
				 name=None, initscheme=None, empty=False, groups=1):
		super().__init__(
			2, inmaps, outmaps, (1, size), (1, stride), (0, pad), (1, dilation), wscale, useBias,
			name, initscheme, empty, groups
		)
		self.registerBlueprint(locals())


	def optimizeForShape(self, shape, memlimit=None):
		shape = shape[:2] + (1, ) + shape[2:]
		super().optimizeForShape(shape, memlimit)


	def updateData(self, data):
		data = data.reshape(*data.shape[:2], 1, *data.shape[2:])
		super().updateData(data)
		self.data = self.data.reshape(*self.data.shape[:2], *self.data.shape[3:])


	def updateGrad(self, grad):
		data = self.inData
		self.inData = data.reshape(*data.shape[:2], 1, *data.shape[2:])

		super().updateGrad(grad.reshape(*grad.shape[:2], 1, *grad.shape[2:]))

		self.inData = data
		self.grad = self.grad.reshape(*self.grad.shape[:2], *self.grad.shape[3:])


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		data = self.inData
		self.inData = data.reshape(*data.shape[:2], 1, *data.shape[2:])

		super().accGradParams(grad.reshape(*grad.shape[:2], 1, *grad.shape[2:]), scale, momentum)
		self.inData = data


	def checkDataShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Data must be 3d tensor")

		_, inmaps, size = shape

		if inmaps != self.W.shape[1] * self.groups:
			raise ModuleError("Data has %d maps (expected: %d)" % (inmaps, self.W.shape[1] * self.groups))

		extsize = size + 2 * self.pad[1]
		extfsize = self.dilation[1] * (self.W.shape[3] - 1) + 1

		if extsize < extfsize:
			raise ModuleError("Data maps size is too small (got %d, expected at least %d)" % (extsize, extfsize))


	def dataShapeFrom(self, shape):
		batchsize, inmaps, insize = shape
		outmaps, _, _, fsize = self.W.shape

		_, pad = self.pad
		_, dilation = self.dilation
		_, stride = self.stride

		outsize = (insize + 2 * pad - dilation * (fsize - 1) - 1) // stride + 1
		return batchsize, outmaps, outsize


	def checkGradShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Grad must be 3d tensor")

		_, outmaps, _ = shape
		if outmaps != self.W.shape[0]:
			raise ModuleError("Grad has %d maps (expected: %d)" % (outmaps, self.W.shape[0]))


	def gradShapeFrom(self, shape):
		batchsize, outmaps, outsize = shape
		_, inmaps, _, fsize = self.W.shape

		_, pad = self.pad
		_, dilation = self.dilation
		_, stride = self.stride

		inmaps *= self.groups
		insize = (outsize - 1) * stride + dilation * (fsize - 1) - 2 * pad + 1

		return batchsize, inmaps, insize


def unittest():
	if Config.backend in {Config.Backend.cuda, Config.Backend.hip}:
		multiMapsWithPadsTest()

	trainTest()


def multiMapsWithPadsTest():
	batchsize, inmaps, size = 5, 4, 3
	outmaps, fsize, stride, pad, dilation = 4, 2, 2, 2, 2

	hostData = np.random.randn(batchsize, inmaps, size).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	conv = Conv1D(inmaps, outmaps, size=fsize, stride=stride, pad=pad, dilation=dilation, initscheme="gaussian")
	conv(data)

	hostW, hostBias = conv.W.get(), conv.b.get()

	hostExtData = np.zeros(shape=(batchsize, inmaps, size + 2 * pad))

	hostExtData[:, :, pad:-pad] = hostData
	hostData = hostExtData

	hostOutData = np.empty(conv.data.shape, dtype=np.float32)
	for c in range(outmaps):
		hostOutData[:, c, :] = hostBias[0, c, 0, 0]

	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for x in range(conv.data.shape[2]):
					for dx in range(fsize):
						hostOutData[b, oc, x] += hostData[b, ic, x * stride + dx * dilation] * hostW[oc, ic, 0, dx]

	assert np.allclose(hostOutData, conv.data.get())

	hostGrad = np.random.randn(*conv.data.shape).astype(np.float32)
	grad = gpuarray.to_gpu(hostGrad)

	conv.backward(grad)
	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for x in range(hostGrad.shape[2]):
					for dx in range(fsize):
						hostInGrad[b, ic, x * stride + dx * dilation] += hostW[oc, ic, 0, dx] * hostGrad[b, oc, x]

	assert np.allclose(hostInGrad[:, :, pad:-pad], conv.grad.get())

	hostWGrad = np.zeros(conv.getVar("W").grad.shape, dtype=np.float32)
	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for dx in range(fsize):
					for x in range(hostGrad.shape[2]):
						hostWGrad[oc, ic, 0, dx] += hostData[b, ic, x * stride + dx * dilation] * hostGrad[b, oc, x]

	assert np.allclose(hostWGrad, conv.getVar("W").grad.get())

	hostBGrad = np.empty(hostBias.shape, dtype=np.float32)
	for oc in range(outmaps):
		hostBGrad[0, oc, 0, 0] = np.sum(hostGrad[:, oc, :])

	assert np.allclose(hostBGrad, conv.getVar("b").grad.get())


def trainTest():
	batchsize, inmaps, size = 5, 1, 3
	outmaps = 1
	fsize = 3

	data = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, inmaps, size)).astype(np.float32))
	conv = Conv1D(inmaps, outmaps, fsize)

	from PuzzleLib.Cost.MSE import MSE
	mse = MSE()

	target = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, outmaps, 1)).astype(np.float32))

	for i in range(100):
		learnRate = 1e-1

		conv(data)
		error, grad = mse(conv.data, target)

		conv.backward(grad)
		conv.updateParams(learnRate)

		if (i + 1) % 5 == 0:
			print("Iteration #%d error: %s" % (i + 1, error))


if __name__ == "__main__":
	unittest()
