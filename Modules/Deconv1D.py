import numpy as np

from PuzzleLib import Config
from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Modules.DeconvND import DeconvND


class Deconv1D(DeconvND):
	def __init__(self, inmaps, outmaps, size, stride=1, pad=0, dilation=1, postpad=0, wscale=1.0, useBias=True,
				 name=None, initscheme=None, empty=False, groups=1):
		super().__init__(
			2, inmaps, outmaps, (1, size), (1, stride), (0, pad), (1, dilation), (0, postpad), wscale, useBias,
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
		grad = grad.reshape(*grad.shape[:2], 1, *grad.shape[2:])

		data = self.inData
		self.inData = data.reshape(*data.shape[:2], 1, *data.shape[2:])
		super().updateGrad(grad)
		self.inData = data

		self.grad = self.grad.reshape(*self.grad.shape[:2], *self.grad.shape[3:])


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		grad = grad.reshape(*grad.shape[:2], 1, *grad.shape[2:])

		data = self.inData
		self.inData = data.reshape(*data.shape[:2], 1, *data.shape[2:])
		super().accGradParams(grad, scale, momentum)
		self.inData = data


	def checkDataShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Data must be 3d tensor")

		_, inmaps, _ = shape
		if inmaps != self.W.shape[0]:
			raise ModuleError("Data has %d maps (expected: %d)" % (inmaps, self.W.shape[0]))


	def dataShapeFrom(self, shape):
		batchsize, inmaps, insize = shape
		_, outmaps, _, fsize = self.W.shape

		_, pad = self.pad
		_, postpad = self.postpad

		_, dilation = self.dilation
		_, stride = self.stride

		outmaps *= self.groups
		outsize = (insize - 1) * stride + dilation * (fsize - 1) - 2 * pad + 1 + postpad

		return batchsize, outmaps, outsize


	def checkGradShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Grad must be 3d tensor")

		_, outmaps, size = shape

		if outmaps != self.W.shape[1] * self.groups:
			raise ModuleError("Grad has %d maps (expected: %d)" % (outmaps, self.W.shape[1] * self.groups))

		if size + 2 * self.pad[1] < self.dilation[1] * (self.W.shape[3] - 1) + 1:
			raise ModuleError(
				"Grad maps height is too small (got %d, expected at least %d)" %
				(size + 2 * self.pad[1], self.dilation[1] * (self.W.shape[3] - 1) + 1)
			)


	def gradShapeFrom(self, shape):
		batchsize, outmaps, outsize = shape
		inmaps, _, _, fsize = self.W.shape

		_, pad = self.pad
		_, dilation = self.dilation
		_, stride = self.stride

		insize = (outsize + 2 * pad - dilation * (fsize - 1) - 1) // stride + 1
		return batchsize, inmaps, insize


def unittest():
	if Config.backend in {Config.Backend.cuda, Config.Backend.hip}:
		multiMapsWithPadsTest()

	trainTest()


def multiMapsWithPadsTest():
	batchsize, inmaps, size = 5, 4, 2
	outmaps, fsize, stride, pad, dilation = 4, 2, 2, 1, 2

	hostData = np.random.randn(batchsize, inmaps, size).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	deconv = Deconv1D(inmaps, outmaps, size=size, stride=stride, pad=pad, dilation=dilation, initscheme="gaussian")
	deconv(data)

	hostW, hostBias = deconv.W.get(), deconv.b.get()
	hostOutData = np.zeros(deconv.data.shape[:2]+(deconv.data.shape[2]+2*pad, ), dtype=np.float32)

	for c in range(outmaps):
		hostOutData[:, c, :] = hostBias[0, c, 0, 0]

	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for x in range(size):
					for dx in range(fsize):
						hostOutData[b, oc, x * stride + dx * dilation] += hostW[ic, oc, 0, dx] * hostData[b, ic, x]

	assert np.allclose(hostOutData[:, :, pad:-pad], deconv.data.get())

	hostGrad = np.random.randn(*deconv.data.shape).astype(np.float32)
	grad = gpuarray.to_gpu(hostGrad)

	deconv.backward(grad)

	hostExtGrad = np.zeros(grad.shape[:2] + (grad.shape[2] + 2 * pad, ), dtype=np.float32)

	hostExtGrad[:, :, pad:-pad] = hostGrad
	hostGrad = hostExtGrad

	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for x in range(size):
					for dx in range(fsize):
						hostInGrad[b, ic, x] += hostGrad[b, oc, x * stride + dx * dilation] * hostW[ic, oc, 0, dx]

	assert np.allclose(hostInGrad, deconv.grad.get())

	hostWGrad = np.zeros(deconv.getVar("W").grad.shape, dtype=np.float32)
	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for dx in range(fsize):
					for x in range(size):
						hostWGrad[ic, oc, 0, dx] += hostGrad[b, oc, x * stride + dx * dilation] * hostData[b, ic, x]

	assert np.allclose(hostWGrad, deconv.getVar("W").grad.get())

	hostBGrad = np.empty(hostBias.shape, dtype=np.float32)
	for oc in range(outmaps):
		hostBGrad[0, oc, 0, 0] = np.sum(hostGrad[:, oc, :])

	assert np.allclose(hostBGrad, deconv.getVar("b").grad.get())


def trainTest():
	batchsize, inmaps, size = 5, 5, 2
	outmaps = 1
	fsize = 3

	data = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, inmaps, size)).astype(np.float32))
	deconv = Deconv1D(inmaps, outmaps, fsize)

	from PuzzleLib.Cost.MSE import MSE
	mse = MSE()

	target = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, outmaps, 4)).astype(np.float32))

	for i in range(100):
		learnRate = 1e-2

		deconv(data)
		error, grad = mse(deconv.data, target)

		deconv.backward(grad)
		deconv.updateParams(learnRate)

		if (i + 1) % 5 == 0:
			print("Iteration #%d error: %s" % (i + 1, error))


if __name__ == "__main__":
	unittest()
