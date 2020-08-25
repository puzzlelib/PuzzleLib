import numpy as np

from PuzzleLib import Config
from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Modules.ConvND import ConvND


class Conv3D(ConvND):
	def __init__(self, inmaps, outmaps, size, stride=1, pad=0, dilation=1, wscale=1.0, useBias=True,
				 name=None, initscheme=None, empty=False, groups=1):
		super().__init__(
			3, inmaps, outmaps, size, stride, pad, dilation, wscale, useBias, name, initscheme, empty, groups
		)
		self.registerBlueprint(locals())


	def checkDataShape(self, shape):
		if len(shape) != 5:
			raise ModuleError("Data must be 5d tensor")

		_, inmaps, ind, inh, inw = shape
		_, _, fd, fh, fw = self.W.shape

		dpad, hpad, wpad = self.pad
		ddilation, hdilation, wdilation = self.dilation

		if inmaps != self.W.shape[1] * self.groups:
			raise ModuleError("Data has %d maps (expected: %d)" % (inmaps, self.W.shape[1] * self.groups))

		extd, exth, extw = ind + 2 * dpad, inh + 2 * hpad, inw + 2 * wpad
		extfd, extfh, extfw = ddilation * (fd - 1) + 1, hdilation * (fh - 1) + 1, wdilation * (fw - 1) + 1

		if extd < extfd:
			raise ModuleError("Data maps depth is too small (got %d, expected >= %d)" % (extd, extfd))

		if exth < extfh:
			raise ModuleError("Data maps height is too small (got %d, expected >= %d)" % (exth, extfh))

		if extw < extfw:
			raise ModuleError("Data maps width is too small (got %d, expected >= %d)" % (extw, extfw))


	def dataShapeFrom(self, shape):
		batchsize, inmaps, ind, inh, inw = shape
		outmaps, _, fd, fh, fw = self.W.shape

		dpad, hpad, wpad = self.pad
		ddilation, hdilation, wdilation = self.dilation
		dstride, hstride, wstride = self.stride

		outd = (ind + 2 * dpad - ddilation * (fd - 1) - 1) // dstride + 1
		outh = (inh + 2 * hpad - hdilation * (fh - 1) - 1) // hstride + 1
		outw = (inw + 2 * wpad - wdilation * (fw - 1) - 1) // wstride + 1

		return batchsize, outmaps, outd, outh, outw


	def checkGradShape(self, shape):
		if len(shape) != 5:
			raise ModuleError("Grad must be 5d tensor")

		_, outmaps, _, _, _ = shape
		if outmaps != self.W.shape[0]:
			raise ModuleError("Grad has %d maps (expected: %d)" % (outmaps, self.W.shape[0]))


	def gradShapeFrom(self, shape):
		batchsize, outmaps, outd, outh, outw = shape
		_, inmaps, fd, fh, fw = self.W.shape

		dpad, hpad, wpad = self.pad
		ddilation, hdilation, wdilation = self.dilation
		dstride, hstride, wstride = self.stride

		inmaps *= self.groups
		ind = (outd - 1) * dstride + ddilation * (fd - 1) - 2 * dpad + 1
		inh = (outh - 1) * hstride + hdilation * (fh - 1) - 2 * hpad + 1
		inw = (outw - 1) * wstride + wdilation * (fw - 1) - 2 * wpad + 1

		return batchsize, inmaps, ind, inh, inw


def unittest():
	if Config.backend in {Config.Backend.cuda, Config.Backend.hip}:
		multiMapsWithPadsTest()

	trainTest()


def multiMapsWithPadsTest():
	batchsize, inmaps, d, h, w = 2, 4, 2, 3, 3
	outmaps, size, stride, pad, dilation = 4, 2, 2, 2, 2

	hostData = np.random.randn(batchsize, inmaps, d, h, w).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	conv = Conv3D(inmaps, outmaps, size=size, stride=stride, pad=pad, dilation=dilation, initscheme="gaussian")
	conv(data)

	hostW, hostBias = conv.W.get(), conv.b.get()
	dl = dilation

	hostExtData = np.zeros(shape=(batchsize, inmaps, d + 2 * pad, h + 2 * pad, w + 2 * pad))

	hostExtData[:, :, pad:-pad, pad:-pad, pad:-pad] = hostData
	hostData = hostExtData

	hostOutData = np.empty(conv.data.shape, dtype=np.float32)
	for c in range(outmaps):
		hostOutData[:, c, :, :, :] = hostBias[0, c, 0, 0, 0]

	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for z in range(conv.data.shape[2]):
					for y in range(conv.data.shape[3]):
						for x in range(conv.data.shape[4]):
							for dz in range(size):
								for dy in range(size):
									for dx in range(size):
										hostOutData[b, oc, z, y, x] += \
											hostData[b, ic, z*stride + dz*dl, y*stride + dy*dl, x*stride + dx*dl] * \
											hostW[oc, ic, dz, dy, dx]

	assert np.allclose(hostOutData, conv.data.get())

	hostGrad = np.random.randn(*conv.data.shape).astype(np.float32)
	grad = gpuarray.to_gpu(hostGrad)

	conv.backward(grad)
	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for z in range(hostGrad.shape[2]):
					for y in range(hostGrad.shape[3]):
						for x in range(hostGrad.shape[4]):
							for dz in range(size):
								for dy in range(size):
									for dx in range(size):
										hostInGrad[b, ic, z*stride + dz*dl, y*stride + dy*dl, x*stride + dx*dl] += \
											hostW[oc, ic, dz, dy, dx] * hostGrad[b, oc, z, y, x]

	assert np.allclose(hostInGrad[:, :, pad:-pad, pad:-pad, pad:-pad], conv.grad.get())

	hostWGrad = np.zeros(conv.getVar("W").grad.shape, dtype=np.float32)
	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for dz in range(size):
					for dy in range(size):
						for dx in range(size):
							for z in range(hostGrad.shape[2]):
								for y in range(hostGrad.shape[3]):
									for x in range(hostGrad.shape[4]):
										hostWGrad[oc, ic, dz, dy, dx] += \
											hostData[b, ic, z*stride + dz*dl, y*stride + dy*dl, x*stride + dx*dl] * \
											hostGrad[b, oc, z, y, x]

	assert np.allclose(hostWGrad, conv.getVar("W").grad.get())

	hostBGrad = np.empty(hostBias.shape, dtype=np.float32)
	for oc in range(outmaps):
		hostBGrad[0, oc, 0, 0, 0] = np.sum(hostGrad[:, oc, :, :, :])

	assert np.allclose(hostBGrad, conv.getVar("b").grad.get())


def trainTest():
	batchsize, inmaps, d, h, w = 5, 1, 3, 3, 3
	outmaps = 1
	size = 3

	data = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, inmaps, d, h, w)).astype(np.float32))
	conv = Conv3D(inmaps, outmaps, size)

	from PuzzleLib.Cost.MSE import MSE
	mse = MSE()

	target = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, outmaps, 1, 1, 1)).astype(np.float32))

	for i in range(100):
		learnRate = 1e-2

		conv(data)
		error, grad = mse(conv.data, target)

		conv.backward(grad)
		conv.updateParams(learnRate)

		if (i + 1) % 5 == 0:
			print("Iteration #%d error: %s" % (i + 1, error))


if __name__ == "__main__":
	unittest()
