import numpy as np

from PuzzleLib import Config
from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Modules.DeconvND import DeconvND


class Deconv2D(DeconvND):
	def __init__(self, inmaps, outmaps, size, stride=1, pad=0, dilation=1, wscale=1.0, useBias=True, name=None,
				 initscheme=None, empty=False, groups=1):
		super().__init__(
			2, inmaps, outmaps, size, stride, pad, dilation, wscale, useBias, name, initscheme, empty, groups
		)
		self.registerBlueprint(locals())


	def checkDataShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Data must be 4d tensor")

		_, inmaps, _, _ = shape
		if inmaps != self.W.shape[0]:
			raise ModuleError("Data has %d maps (expected: %d)" % (inmaps, self.W.shape[0]))


	def dataShapeFrom(self, shape):
		batchsize, inmaps, inh, inw = shape
		_, outmaps, fh, fw = self.W.shape

		hpad, wpad = self.pad
		hdilation, wdilation = self.dilation
		hstride, wstride = self.stride

		outmaps *= self.groups
		outh = (inh - 1) * hstride + hdilation * (fh - 1) - 2 * hpad + 1
		outw = (inw - 1) * wstride + wdilation * (fw - 1) - 2 * wpad + 1

		return batchsize, outmaps, outh, outw


	def checkGradShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Grad must be 4d tensor")

		_, outmaps, outh, outw = shape
		_, _, fh, fw = self.W.shape

		hpad, wpad = self.pad
		hdilation, wdilation = self.dilation

		if outmaps != self.W.shape[1] * self.groups:
			raise ModuleError("Grad has %d maps (expected: %d)" % (outmaps, self.W.shape[1] * self.groups))

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
		batchsize, outmaps, outh, outw = shape
		inmaps, _, fh, fw = self.W.shape

		hpad, wpad = self.pad
		hdilation, wdilation = self.dilation
		hstride, wstride = self.stride

		inh = (outh + 2 * hpad - hdilation * (fh - 1) - 1) // hstride + 1
		inw = (outw + 2 * wpad - wdilation * (fw - 1) - 1) // wstride + 1

		return batchsize, inmaps, inh, inw


def unittest():
	multiInMapsTest()
	multiOutMapsTest()

	if Config.backend == Config.Backend.cuda:
		multiMapsWithPadsTest()

	trainTest()


def multiInMapsTest():
	batchsize, inmaps, h, w = 1, 2, 10, 10
	outmaps = 1
	stride = 2
	size = 4

	data = gpuarray.to_gpu(np.random.randn(batchsize, inmaps, h, w).astype(np.float32))

	deconv = Deconv2D(inmaps, outmaps, size, stride=2)
	deconv(data)

	hostOutData = np.zeros(deconv.data.shape).astype(np.float32)

	for k in range(inmaps):
		for i in range(0, hostOutData.shape[2] - size + 1, stride):
			for j in range(0, hostOutData.shape[3] - size + 1, stride):
				hostOutData[0,0,i:size+i,j:size+j] += deconv.W.get()[k,0] * data.get()[0,k,int(i/stride),int(j/stride)]
		hostOutData[0, 0] += deconv.b.get()[0, 0]

	assert np.allclose(hostOutData, deconv.data.get())


def multiOutMapsTest():
	batchsize, inmaps, h, w = 1, 1, 2, 2
	outmaps = 2
	stride = 2
	size = 4

	data = gpuarray.to_gpu(np.random.randn(batchsize, inmaps, h, w).astype(np.float32))

	deconv = Deconv2D(inmaps, outmaps, size, stride=stride)
	deconv(data)

	hostOutData = np.zeros(deconv.data.shape).astype(np.float32)

	for k in range(outmaps):
		for i in range(0, hostOutData.shape[2] - size + 1, stride):
			for j in range(0, hostOutData.shape[3] - size + 1, stride):
				hostOutData[0,k,i:size+i,j:size+j] += deconv.W.get()[0,k] * data.get()[0,0,int(i/stride),int(j/stride)]
		hostOutData[0, k] += deconv.b.get()[0, k]

	assert np.allclose(hostOutData, deconv.data.get())


def multiMapsWithPadsTest():
	batchsize, inmaps, h, w = 3, 4, 2, 2
	outmaps, size, stride, pad, dilation = 4, 3, 2, 1, 2

	data = gpuarray.to_gpu(np.random.randn(batchsize, inmaps, h, w).astype(np.float32))

	deconv = Deconv2D(inmaps, outmaps, size=size, stride=stride, pad=pad, dilation=dilation, initscheme="gaussian")
	deconv(data)

	hostW, hostBias = deconv.W.get(), deconv.b.get()
	dl = dilation

	hostData, hostOutData = data.get(), np.zeros(deconv.data.shape[:2] + (deconv.data.shape[2] + 2 * pad,
												deconv.data.shape[3] + 2 * pad), dtype=np.float32)

	for c in range(outmaps):
		hostOutData[:, c, :, :] = hostBias[0, c, 0, 0]

	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for y in range(h):
					for x in range(w):
						for dy in range(size):
							for dx in range(size):
								hostOutData[b,oc,y*stride+dy*dl,x*stride+dx*dl] += hostW[ic,oc,dy,dx]*hostData[b,ic,y,x]

	assert np.allclose(hostOutData[:, :, pad:-pad, pad:-pad], deconv.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*deconv.data.shape).astype(np.float32))
	deconv.backward(grad)

	hostGrad = np.zeros(grad.shape[:2] + (grad.shape[2] + 2 * pad, grad.shape[3] + 2 * pad), dtype=np.float32)
	hostGrad[:, :, pad:-pad, pad:-pad] = grad.get()

	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for y in range(h):
					for x in range(w):
						for dy in range(size):
							for dx in range(size):
								hostInGrad[b,ic,y,x] += hostGrad[b,oc,y*stride+dy*dl, x*stride+dx*dl]*hostW[ic,oc,dy,dx]

	assert np.allclose(hostInGrad, deconv.grad.get())

	hostWGrad = np.zeros(deconv.getVar("W").grad.shape, dtype=np.float32)
	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for dy in range(size):
					for dx in range(size):
						for y in range(h):
							for x in range(w):
								hostWGrad[ic,oc,dy,dx]+=hostGrad[b,oc,y*stride+dy*dl,x*stride+dx*dl]*hostData[b,ic,y,x]

	assert np.allclose(hostWGrad, deconv.getVar("W").grad.get())

	hostBiasGrad = np.empty(hostBias.shape, dtype=np.float32)
	for oc in range(outmaps):
		hostBiasGrad[0, oc, 0, 0] = np.sum(hostGrad[:, oc, :, :])

	assert np.allclose(hostBiasGrad, deconv.getVar("b").grad.get())


def trainTest():
	batchsize, inmaps, h, w = 5, 5, 2, 2
	outmaps = 1
	size = 8

	data = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, inmaps, h, w)).astype(np.float32))
	deconv = Deconv2D(inmaps, outmaps, size)

	from PuzzleLib.Cost.MSE import MSE
	mse = MSE()

	target = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (batchsize, outmaps, 9, 9)).astype(np.float32))

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
