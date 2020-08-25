import numpy as np

from PuzzleLib import Config
from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Modules.ConvND import ConvND


class Conv2D(ConvND):
	def __init__(self, inmaps, outmaps, size, stride=1, pad=0, dilation=1, wscale=1.0, useBias=True,
				 name=None, initscheme=None, empty=False, groups=1):
		super().__init__(
			2, inmaps, outmaps, size, stride, pad, dilation, wscale, useBias, name, initscheme, empty, groups
		)
		self.registerBlueprint(locals())


	def checkDataShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Data must be 4d tensor")

		_, inmaps, inh, inw = shape
		_, _, fh, fw = self.W.shape

		hpad, wpad = self.pad
		hdilation, wdilation = self.dilation

		if inmaps != self.W.shape[1] * self.groups:
			raise ModuleError("Data has %d maps (expected: %d)" % (inmaps, self.W.shape[1] * self.groups))

		exth, extw = inh + 2 * hpad, inw + 2 * wpad
		extfh, extfw = hdilation * (fh - 1) + 1, wdilation * (fw - 1) + 1

		if exth < extfh:
			raise ModuleError("Data maps height is too small (got %d, expected at least %d)" % (exth, extfh))

		if extw < extfw:
			raise ModuleError("Data maps width is too small (got %d, expected at least %d)" % (extw, extfw))


	def dataShapeFrom(self, shape):
		batchsize, inmaps, inh, inw = shape
		outmaps, _, fh, fw = self.W.shape

		hpad, wpad = self.pad
		hdilation, wdilation = self.dilation
		hstride, wstride = self.stride

		outh = (inh + 2 * hpad - hdilation * (fh - 1) - 1) // hstride + 1
		outw = (inw + 2 * wpad - wdilation * (fw - 1) - 1) // wstride + 1

		return batchsize, outmaps, outh, outw


	def checkGradShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Grad must be 4d tensor")

		_, outmaps, _, _ = shape
		if outmaps != self.W.shape[0]:
			raise ModuleError("Grad has %d maps (expected: %d)" % (outmaps, self.W.shape[0]))


	def gradShapeFrom(self, shape):
		batchsize, outmaps, outh, outw = shape
		_, inmaps, fh, fw = self.W.shape

		hpad, wpad = self.pad
		hdilation, wdilation = self.dilation
		hstride, wstride = self.stride

		inmaps *= self.groups
		inh = (outh - 1) * hstride + hdilation * (fh - 1) - 2 * hpad + 1
		inw = (outw - 1) * wstride + wdilation * (fw - 1) - 2 * wpad + 1

		return batchsize, inmaps, inh, inw


def unittest():
	oneMapTest()
	multiOutMapsTest()
	multiInMapsTest()

	if Config.backend in {Config.Backend.cuda, Config.Backend.hip}:
		multiMapsWithPadsTest()
		groupTest()

	trainTest()


def oneMapTest():
	batchsize, inmaps, h, w = 1, 1, 5, 5
	outmaps, size, postpad = 1, 2, 1

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	conv = Conv2D(inmaps, outmaps, size)
	conv(data)

	hostW, hostBias = conv.W.get(), conv.b.get()

	hostOutData = np.empty(conv.data.shape, dtype=np.float32)
	hostOutData[:, 0, :, :] = hostBias[0, 0, 0, 0]

	for y in range(hostOutData.shape[2]):
		for x in range(hostOutData.shape[3]):
			for dy in range(size):
				for dx in range(size):
					hostOutData[0, 0, y, x] += hostData[0, 0, y + dy, x + dx] * hostW[0, 0, dy, dx]

	assert np.allclose(hostOutData, conv.data.get())

	hostGrad = np.random.randn(*conv.data.shape).astype(np.float32)
	grad = gpuarray.to_gpu(hostGrad)

	conv.backward(grad)
	hostInGrad = np.zeros(data.shape).astype(np.float32)

	for y in range(hostGrad.shape[2]):
		for x in range(hostGrad.shape[3]):
			for dy in range(size):
				for dx in range(size):
					hostInGrad[0, 0, y + dy, x + dx] += hostW[0, 0, dy, dx] * hostGrad[0, 0, y, x]

	assert np.allclose(hostInGrad, conv.grad.get())


def multiOutMapsTest():
	batchsize, inmaps, h, w = 1, 1, 8, 8
	outmaps, size = 2, 4

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	conv = Conv2D(inmaps, outmaps, size)
	conv(data)

	hostW, hostBias = conv.W.get(), conv.b.get()
	hostOutData = np.empty(conv.data.shape, dtype=np.float32)

	for c in range(outmaps):
		hostOutData[:, c, :, :] = hostBias[0, c, 0, 0]

	for oc in range(outmaps):
		for y in range(conv.data.shape[2]):
			for x in range(conv.data.shape[3]):
				for dy in range(size):
					for dx in range(size):
						hostOutData[0, oc, y, x] += hostData[0, 0, y + dy, x + dx] * hostW[oc, 0, dy, dx]

	assert np.allclose(hostOutData, conv.data.get())


def multiInMapsTest():
	batchsize, inmaps, h, w = 1, 2, 10, 10
	outmaps, size = 1, 4

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	conv = Conv2D(inmaps, outmaps, size)
	conv(data)

	hostW, hostBias = conv.W.get(), conv.b.get()
	hostOutData = np.empty(conv.data.shape, dtype=np.float32)

	for c in range(outmaps):
		hostOutData[:, c, :, :] = hostBias[0, c, 0, 0]

	for ic in range(inmaps):
		for y in range(conv.data.shape[2]):
			for x in range(conv.data.shape[3]):
				for dy in range(size):
					for dx in range(size):
						hostOutData[0, 0, y, x] += hostData[0, ic, y + dy, x + dx] * hostW[0, ic, dy, dx]

	assert np.allclose(hostOutData, conv.data.get())


def multiMapsWithPadsTest():
	batchsize, inmaps, h, w = 3, 4, 3, 3
	outmaps, size, stride, pad, dilation = 4, 3, 2, 2, 2

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	conv = Conv2D(inmaps, outmaps, size=size, stride=stride, pad=pad, dilation=dilation, initscheme="gaussian")
	conv(data)

	hostW, hostBias = conv.W.get(), conv.b.get()
	dl = dilation

	hostExtData = np.zeros(shape=(batchsize, inmaps, h + 2 * pad, w + 2 * pad))

	hostExtData[:, :, pad:-pad, pad:-pad] = hostData
	hostData = hostExtData

	hostOutData = np.empty(conv.data.shape, dtype=np.float32)
	for c in range(outmaps):
		hostOutData[:, c, :, :] = hostBias[0, c, 0, 0]

	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for y in range(conv.data.shape[2]):
					for x in range(conv.data.shape[3]):
						for dy in range(size):
							for dx in range(size):
								hostOutData[b,oc,y,x] += hostData[b,ic,y*stride+dy*dl,x*stride+dx*dl]*hostW[oc,ic,dy,dx]

	assert np.allclose(hostOutData, conv.data.get())

	hostGrad = np.random.randn(*conv.data.shape).astype(np.float32)
	grad = gpuarray.to_gpu(hostGrad)

	conv.backward(grad)
	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for y in range(hostGrad.shape[2]):
					for x in range(hostGrad.shape[3]):
						for dy in range(size):
							for dx in range(size):
								hostInGrad[b,ic,y*stride+dy*dl,x*stride+dx*dl] += hostW[oc,ic,dy,dx]*hostGrad[b,oc,y,x]

	assert np.allclose(hostInGrad[:, :, pad:-pad, pad:-pad], conv.grad.get())

	hostWGrad = np.zeros(conv.getVar("W").grad.shape, dtype=np.float32)
	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for dy in range(size):
					for dx in range(size):
						for y in range(hostGrad.shape[2]):
							for x in range(hostGrad.shape[3]):
								hostWGrad[oc,ic,dy,dx]+=hostData[b,ic,y*stride+dy*dl,x*stride+dx*dl]*hostGrad[b,oc,y,x]

	assert np.allclose(hostWGrad, conv.getVar("W").grad.get())

	hostBGrad = np.empty(hostBias.shape, dtype=np.float32)
	for oc in range(outmaps):
		hostBGrad[0, oc, 0, 0] = np.sum(hostGrad[:, oc, :, :])

	assert np.allclose(hostBGrad, conv.getVar("b").grad.get())


def groupTest():
	batchsize, inmaps, inh, inw = 3, 4, 4, 5
	size, outmaps = 3, 4
	groups = 2

	hostData = np.random.randn(batchsize, inmaps, inh, inw).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	conv = Conv2D(inmaps, outmaps, size=size, initscheme="gaussian", groups=groups)
	conv(data)

	hostOutData = np.empty(conv.data.shape, dtype=np.float32)
	hostW, hostBias = conv.W.get(), conv.b.get()

	for c in range(outmaps):
		hostOutData[:, c, :, :] = hostBias[0, c, 0, 0]

	ingrpsize = inmaps // groups
	outgrpsize = outmaps // groups

	for g in range(groups):
		hostOutGroup = hostOutData[:, g * outgrpsize:(g + 1) * outgrpsize, :, :]
		hostGroup = hostData[:, g * ingrpsize:(g + 1) * ingrpsize, :, :]
		for b in range(batchsize):
			for oc in range(outgrpsize):
				for ic in range(ingrpsize):
					for y in range(conv.data.shape[2]):
						for x in range(conv.data.shape[3]):
							for dy in range(size):
								for dx in range(size):
									hostOutGroup[b, oc, y, x] += hostGroup[b, ic, y + dy, x + dx] * \
																 hostW[g * outgrpsize + oc, ic, dy, dx]

	assert np.allclose(hostOutData, conv.data.get())

	hostGrad = np.random.randn(*conv.data.shape).astype(np.float32)
	grad = gpuarray.to_gpu(hostGrad)

	conv.backward(grad)
	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for g in range(groups):
		hostGroup = hostGrad[:, g * outgrpsize:(g + 1) * outgrpsize, :, :]
		hostInGroup = hostInGrad[:, g * ingrpsize:(g + 1) * ingrpsize, :, :]
		for b in range(batchsize):
			for ic in range(ingrpsize):
				for oc in range(outgrpsize):
					for y in range(hostGrad.shape[2]):
						for x in range(hostGrad.shape[3]):
							for dy in range(size):
								for dx in range(size):
									hostInGroup[b, ic, y + dy, x + dx] += hostW[g * outgrpsize + oc, ic, dy, dx] * \
																		  hostGroup[b, oc, y, x]

	assert np.allclose(hostInGrad, conv.grad.get())

	hostWGrad = np.zeros(conv.getVar("W").grad.shape, dtype=np.float32)
	for g in range(groups):
		hostGradGroup = hostGrad[:, g * outgrpsize:(g + 1) * outgrpsize, :, :]
		hostDataGroup = hostData[:, g * ingrpsize:(g + 1) * ingrpsize, :, :]
		for b in range(batchsize):
			for oc in range(outgrpsize):
				for ic in range(ingrpsize):
					for dy in range(size):
						for dx in range(size):
							for y in range(hostGrad.shape[2]):
								for x in range(hostGrad.shape[3]):
									hostWGrad[g * outgrpsize + oc, ic, dy, dx] += hostDataGroup[b, ic, y+dy, x+dx] * \
																				  hostGradGroup[b, oc, y, x]

	assert np.allclose(hostWGrad, conv.getVar("W").grad.get())

	hostBGrad = np.empty(hostBias.shape, dtype=np.float32)
	for oc in range(outmaps):
		hostBGrad[0, oc, 0, 0] = np.sum(hostGrad[:, oc, :, :])

	assert np.allclose(hostBGrad, conv.getVar("b").grad.get())


def trainTest():
	batchsize, inmaps, h, w = 5, 1, 8, 8
	outmaps = 1
	size = 8

	data = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, inmaps, h, w)).astype(np.float32))
	conv = Conv2D(inmaps, outmaps, size)

	from PuzzleLib.Cost.MSE import MSE
	mse = MSE()

	target = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, outmaps, 1, 1)).astype(np.float32))

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
