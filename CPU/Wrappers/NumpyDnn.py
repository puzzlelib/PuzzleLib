from enum import Enum
import numpy as np

from PuzzleLib.CPU.CPUArray import CPUArray


class PoolMode(Enum):
	max = 0
	avgWithPad = 1


def repeatValue(val, ntimes):
	if isinstance(val, int):
		return (val, ) * ntimes

	elif isinstance(val, (list, tuple)):
		return tuple(val)

	else:
		raise NotImplementedError(val.__class__.__name__)


def outshape(inshape, size, stride, pad):
	inh, inw = inshape

	hsize, wsize = size
	hstride, wstride = stride
	hpad, wpad = pad

	outh = (inh + 2 * hpad - hsize) // hstride + 1
	outw = (inw + 2 * wpad - wsize) // wstride + 1

	return outh, outw


def im2col(data, size, stride, pad):
	assert data.ndim == 4

	hsize, wsize = size
	hstride, wstride = stride
	hpad, wpad = pad

	batchsize, maps, inh, inw = data.shape
	outh, outw = outshape((inh, inw), size, stride, pad)

	data = np.pad(data, ((0, 0), (0, 0), (hpad, hpad), (wpad, wpad)), mode="constant", constant_values=0)

	strides = (
		data.strides[0], hstride * data.strides[2], wstride * data.strides[3],
		data.strides[1], data.strides[2], data.strides[3]
	)

	coldata = np.lib.stride_tricks.as_strided(data, shape=(batchsize, outh, outw, maps, hsize, wsize), strides=strides)
	coldata = coldata.reshape(batchsize * outh * outw, maps * hsize * wsize)

	return coldata


def col2im(data, maps, shape):
	assert data.ndim == 2
	h, w = shape

	data = data.reshape(-1, h, w, maps)
	data = np.moveaxis(data, 3, 1)

	return np.ascontiguousarray(data)


def linear(data, W, bias):
	outdata = np.dot(data, W)

	if bias is not None:
		outdata += bias

	return outdata


def conv2d(data, W, bias=None, stride=1, pad=0):
	assert data.ndim == 4 and W.ndim == 4

	batchsize, _, inh, inw = data.shape
	stride, pad = repeatValue(stride, 2), repeatValue(pad, 2)

	outmaps, _, hsize, wsize = W.shape
	outh, outw = outshape((inh, inw), (hsize, wsize), stride, pad)

	coldata = im2col(data.data, W.shape[2:], stride, pad)
	W = W.data.reshape(W.shape[0], -1).T

	bias = bias.data.reshape(1, bias.shape[1]) if bias is not None else None
	outdata = linear(coldata, W, bias)

	outdata = col2im(outdata, outmaps, (outh, outw))
	return CPUArray(outdata.shape, outdata.dtype, data=outdata, acquire=True)


def pool2d(data, size=2, stride=2, pad=0, mode=PoolMode.max):
	assert data.ndim == 4
	onRow = np.max if mode == PoolMode.max else np.mean

	batchsize, maps, inh, inw = data.shape
	size, stride, pad = repeatValue(size, 2), repeatValue(stride, 2), repeatValue(pad, 2)

	outh, outw = outshape((inh, inw), size, stride, pad)

	coldata = im2col(data.data.reshape(batchsize * maps, 1, inh, inw), size, stride, pad)
	outdata = onRow(coldata, axis=1, keepdims=True).reshape((batchsize, maps, outh, outw))

	return CPUArray(outdata.shape, outdata.dtype, data=outdata, acquire=True)


def batchNorm2d(data, scale, bias, mean, var, epsilon=1e-5, test=False, out=None):
	assert data.ndim == scale.ndim and scale.ndim == bias.ndim and bias.ndim == mean.ndim and mean.ndim == var.ndim
	assert test

	scale = scale.data / np.sqrt(var.data + epsilon)
	outdata = scale * (data.data - mean.data) + bias.data

	return CPUArray(outdata.shape, outdata.dtype, data=outdata, acquire=True)


def unittest():
	conv2dTest()
	maxpool2dTest()
	batchNorm2dTest()


def conv2dTest():
	batchsize, inmaps, h, w = 1, 2, 6, 6
	fsize, outmaps = 2, 4

	data = CPUArray.toDevice(np.random.randn(batchsize, inmaps, h, w).astype(np.float32))

	W = CPUArray.toDevice(np.random.randn(outmaps, inmaps, fsize, fsize).astype(np.float32))
	bias = CPUArray.toDevice(np.random.randn(1, outmaps, 1, 1).astype(np.float32))

	outdata = conv2d(data, W, bias)

	hostData, hostW, hostBias = data.get(), W.get(), bias.get()
	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for c in range(outmaps):
		hostOutData[:, c, :, :] = hostBias[0, c, 0, 0]

	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for y in range(outdata.shape[2]):
					for x in range(outdata.shape[3]):
						for dy in range(fsize):
							for dx in range(fsize):
								hostOutData[b, oc, y, x] += hostData[b, ic, y + dy, x + dx] * hostW[oc, ic, dy, dx]

	assert np.allclose(hostOutData, outdata.get())


def maxpool2dTest():
	batchsize, maps, h, w = 1, 1, 8, 8
	data = CPUArray.toDevice(np.random.randn(batchsize, maps, h, w).astype(np.float32))

	outdata = pool2d(data)

	def maxDownSample2d(dat, factor):
		trimrows = dat.shape[0] // factor * factor
		trimcols = dat.shape[1] // factor * factor

		maxSoFar = None
		first = True

		for coff in range(factor):
			for roff in range(factor):
				hopped = dat[roff:trimrows:factor, coff:trimcols:factor]
				if first:
					maxSoFar = hopped
					first = False
				else:
					maxSoFar = np.maximum(maxSoFar, hopped)

		return maxSoFar

	hostOutData = maxDownSample2d(data.get()[0, 0], 2)
	assert np.allclose(hostOutData, outdata.get())


def batchNorm2dTest():
	batchsize, maps, h, w = 4, 5, 3, 2

	data = CPUArray.toDevice(np.random.randn(batchsize, maps, h, w).astype(np.float32))
	hostData = data.get()

	scale = CPUArray.toDevice(np.random.randn(1, maps, 1, 1).astype(np.float32))
	bias = CPUArray.toDevice(np.random.randn(1, maps, 1, 1).astype(np.float32))
	mean = CPUArray.toDevice(np.random.randn(1, maps, 1, 1).astype(np.float32))
	var = CPUArray.toDevice(
		(np.ones((1, maps, 1, 1)).astype(np.float32) + np.random.randn(1, maps, 1, 1).astype(np.float32))**2
	)

	outdata = batchNorm2d(data, scale, bias, mean, var, test=True)

	hostScale, hostBias, hostMean, hostVar = scale.get(), bias.get(), mean.get(), var.get()
	hostNormData = np.empty(hostData.shape, dtype=np.float32)
	hostOutData = np.empty(hostData.shape, dtype=np.float32)

	for c in range(maps):
		hostNormData[:, c, :, :] = (hostData[:, c, :, :] - hostMean[0, c, 0, 0]) / np.sqrt(hostVar[0, c, 0, 0] + 1e-5)
		hostOutData[:, c, :, :] = hostNormData[:, c, :, :] * hostScale[0, c, 0, 0] + hostBias[0, c, 0, 0]

	assert np.allclose(hostOutData, outdata.get())


if __name__ == "__main__":
	unittest()
