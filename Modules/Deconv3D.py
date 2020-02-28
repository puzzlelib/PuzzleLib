import numpy as np

from PuzzleLib import Config
from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Modules.DeconvND import DeconvND


class Deconv3D(DeconvND):
	def __init__(self, inmaps, outmaps, size, stride=1, pad=0, dilation=1, wscale=1.0, useBias=True, name=None,
				 initscheme=None, empty=False, groups=1):
		super().__init__(
			3, inmaps, outmaps, size, stride, pad, dilation, wscale, useBias, name, initscheme, empty, groups
		)
		self.registerBlueprint(locals())


	def checkDataShape(self, shape):
		if len(shape) != 5:
			raise ModuleError("Data must be 5d tensor")

		_, inmaps, _, _, _ = shape
		if inmaps != self.W.shape[0]:
			raise ModuleError("Data has %d maps (expected: %d)" % (inmaps, self.W.shape[0]))


	def dataShapeFrom(self, shape):
		batchsize, inmaps, ind, inh, inw = shape
		_, outmaps, fd, fh, fw = self.W.shape

		dpad, hpad, wpad = self.pad
		ddilation, hdilation, wdilation = self.dilation
		dstride, hstride, wstride = self.stride

		outmaps *= self.groups
		outd = (ind - 1) * dstride + ddilation * (fd - 1) - 2 * dpad + 1
		outh = (inh - 1) * hstride + hdilation * (fh - 1) - 2 * hpad + 1
		outw = (inw - 1) * wstride + wdilation * (fw - 1) - 2 * wpad + 1

		return batchsize, outmaps, outd, outh, outw


	def checkGradShape(self, shape):
		if len(shape) != 5:
			raise ModuleError("Grad must be 5d tensor")

		_, outmaps, outd, outh, outw = shape
		_, _, fd, fh, fw = self.W.shape

		dpad, hpad, wpad = self.pad
		ddilation, hdilation, wdilation = self.dilation

		if outmaps != self.W.shape[1] * self.groups:
			raise ModuleError("Grad has %d maps (expected: %d)" % (outmaps, self.W.shape[1] * self.groups))

		if outd + 2 * dpad < ddilation * (fd - 1) + 1:
			raise ModuleError(
				"Grad maps depth is too small (got %d, expected at least %d)" %
				(outd + 2 * dpad, ddilation * (fd - 1) + 1)
			)

		if outh + 2 * hpad < hdilation * (fh - 1) + 1:
			raise ModuleError(
				"Grad maps height is too small (got %d, expected at least %d)" %
				(outh + 2 * hpad, hdilation * (fh - 1) + 1)
			)

		if outw + 2 * wpad < wdilation * (fw - 1) + 1:
			raise ModuleError(
				"Grad maps width is too small (got %d, expected at least %d)" %
				(outw + 2 * wpad, wdilation * (fw - 1) + 1)
			)


	def gradShapeFrom(self, shape):
		batchsize, outmaps, outd, outh, outw = shape
		inmaps, _, fd, fh, fw = self.W.shape

		dpad, hpad, wpad = self.pad
		ddilation, hdilation, wdilation = self.dilation
		dstride, hstride, wstride = self.stride

		ind = (outd + 2 * dpad - ddilation * (fd - 1) - 1) // dstride + 1
		inh = (outh + 2 * hpad - hdilation * (fh - 1) - 1) // hstride + 1
		inw = (outw + 2 * wpad - wdilation * (fw - 1) - 1) // wstride + 1

		return batchsize, inmaps, ind, inh, inw


def unittest():
	if Config.backend == Config.Backend.cuda:
		multiMapsWithPadsTest()

	trainTest()


def multiMapsWithPadsTest():
	batchsize, inmaps, d, h, w = 5, 4, 2, 2, 2
	outmaps, size, stride, pad, dilation = 4, 2, 2, 1, 2

	data = gpuarray.to_gpu(np.random.randn(batchsize, inmaps, d, h, w).astype(np.float32))

	deconv = Deconv3D(inmaps, outmaps, size=size, stride=stride, pad=pad, dilation=dilation, initscheme="gaussian")
	deconv(data)

	hostW, hostBias = deconv.W.get(), deconv.b.get()
	dl = dilation

	hostData, hostOutData = data.get(), np.zeros(deconv.data.shape[:2] + (deconv.data.shape[2] + 2 * pad,
										deconv.data.shape[3] + 2 * pad, deconv.data.shape[4] + 2 * pad),
												 dtype=np.float32)

	for c in range(outmaps):
		hostOutData[:, c, :, :, :] = hostBias[0, c, 0, 0, 0]

	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for z in range(d):
					for y in range(h):
						for x in range(w):
							for dz in range(size):
								for dy in range(size):
									for dx in range(size):
										hostOutData[b, oc, z*stride + dz*dl, y*stride + dy*dl, x*stride + dx*dl] += \
											hostW[ic, oc, dz, dy, dx] * hostData[b, ic, z, y, x]

	assert np.allclose(hostOutData[:, :, pad:-pad, pad:-pad, pad:-pad], deconv.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*deconv.data.shape).astype(np.float32))
	deconv.backward(grad)

	hostGrad = np.zeros(grad.shape[:2]+(grad.shape[2]+2*pad,grad.shape[3]+2*pad,grad.shape[4]+2*pad), dtype=np.float32)
	hostGrad[:, :, pad:-pad, pad:-pad, pad:-pad] = grad.get()

	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for z in range(d):
					for y in range(h):
						for x in range(w):
							for dz in range(size):
								for dy in range(size):
									for dx in range(size):
										hostInGrad[b, ic, z, y, x] += \
											hostGrad[b, oc, z*stride + dz*dl, y*stride + dy*dl, x*stride+dx*dl] * \
											hostW[ic, oc, dz, dy, dx]

	assert np.allclose(hostInGrad, deconv.grad.get())

	hostWGrad = np.zeros(deconv.getVar("W").grad.shape, dtype=np.float32)
	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for dz in range(size):
					for dy in range(size):
						for dx in range(size):
							for z in range(d):
								for y in range(h):
									for x in range(w):
										hostWGrad[ic, oc, dz, dy, dx] += \
											hostGrad[b, oc, z*stride + dz*dl, y*stride + dy*dl, x*stride + dx*dl] * \
											hostData[b, ic, z, y, x]

	assert np.allclose(hostWGrad, deconv.getVar("W").grad.get())

	hostBGrad = np.empty(hostBias.shape, dtype=np.float32)
	for oc in range(outmaps):
		hostBGrad[0, oc, 0, 0, 0] = np.sum(hostGrad[:, oc, :, :, :])

	assert np.allclose(hostBGrad, deconv.getVar("b").grad.get())


def trainTest():
	batchsize, inmaps, d, h, w = 5, 5, 2, 2, 2
	outmaps = 1
	size = 3

	data = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, inmaps, d, h, w)).astype(np.float32))
	deconv = Deconv3D(inmaps, outmaps, size)

	from PuzzleLib.Cost.MSE import MSE
	mse = MSE()

	target = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, outmaps, 4, 4, 4)).astype(np.float32))

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
