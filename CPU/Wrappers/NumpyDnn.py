from enum import Enum
import itertools

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


def outshape(inshape, size, stride, pad, dilation):
	inh, inw = inshape

	fh, fw = size
	hstride, wstride = stride
	hpad, wpad = pad
	hdilation, wdilation = dilation

	outh = (inh + 2 * hpad - hdilation * (fh - 1) - 1) // hstride + 1
	outw = (inw + 2 * wpad - wdilation * (fw - 1) - 1) // wstride + 1

	return outh, outw


def im2col(data, size, stride, pad, dilation, padval=0):
	assert data.ndim == 4

	fh, fw = size
	hstride, wstride = stride
	hpad, wpad = pad
	hdilation, wdilation = dilation

	batchsize, maps, inh, inw = data.shape
	outh, outw = outshape((inh, inw), size, stride, pad, dilation)

	if hpad > 0 or wpad > 0:
		data = np.pad(data, ((0, 0), (0, 0), (hpad, hpad), (wpad, wpad)), mode="constant", constant_values=padval)

	strides = (
		data.strides[0], hstride * data.strides[2], wstride * data.strides[3],
		data.strides[1], data.strides[2] * hdilation, data.strides[3] * wdilation
	)

	coldata = np.lib.stride_tricks.as_strided(data, shape=(batchsize, outh, outw, maps, fh, fw), strides=strides)
	coldata = coldata.reshape(batchsize * outh * outw, maps * fh * fw)

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


def conv2d(data, W, bias=None, stride=1, pad=0, dilation=1):
	assert data.ndim == 4 and W.ndim == 4

	batchsize, _, inh, inw = data.shape
	stride, pad, dilation = repeatValue(stride, 2), repeatValue(pad, 2), repeatValue(dilation, 2)

	outmaps, _, hsize, wsize = W.shape
	outh, outw = outshape((inh, inw), (hsize, wsize), stride, pad, dilation)

	coldata = im2col(data.data, W.shape[2:], stride, pad, dilation)
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

	outh, outw = outshape((inh, inw), size, stride, pad, (1, 1))

	coldata = im2col(data.data.reshape(batchsize * maps, 1, inh, inw), size, stride, pad, (1, 1), padval=-np.inf)
	outdata = onRow(coldata, axis=1, keepdims=True).reshape((batchsize, maps, outh, outw))

	return CPUArray(outdata.shape, outdata.dtype, data=outdata, acquire=True)


def batchNorm2d(data, scale, bias, mean, var, epsilon=1e-5, test=False, out=None):
	assert data.ndim == scale.ndim and scale.ndim == bias.ndim and bias.ndim == mean.ndim and mean.ndim == var.ndim
	assert test

	scale = scale.data / np.sqrt(var.data + epsilon)
	outdata = scale * (data.data - mean.data) + bias.data

	if out is None:
		out = CPUArray(outdata.shape, outdata.dtype, data=outdata, acquire=True)
	else:
		out.set(outdata)

	return out


def unittest():
	conv2dTest()
	conv2dExtTest()
	maxpool2dTest()
	batchNorm2dTest()


def conv2dTest():
	batchsize, inmaps, h, w = 1, 2, 6, 6
	fsize, outmaps = 2, 4

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(np.float32)
	data = CPUArray.toDevice(hostData)

	hostW = np.random.randn(outmaps, inmaps, fsize, fsize).astype(np.float32)
	hostBias = np.random.randn(1, outmaps, 1, 1).astype(np.float32)

	W, bias = CPUArray.toDevice(hostW), CPUArray.toDevice(hostBias)
	outdata = conv2d(data, W, bias)

	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for c in range(outmaps):
		hostOutData[:, c, :, :] = hostBias[0, c, 0, 0]

	for b, oc, ic, y, x, dy, dx in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(outdata.shape[2]), range(outdata.shape[3]),
		range(fsize), range(fsize)
	):
		hostOutData[b, oc, y, x] += hostData[b, ic, y + dy, x + dx] * hostW[oc, ic, dy, dx]

	assert np.allclose(hostOutData, outdata.get())


def conv2dExtTest():
	batchsize, inmaps, h, w = 3, 4, 3, 3
	outmaps, fsize, stride, pad, dilation = 4, 3, 2, 2, 2

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(np.float32)
	data = CPUArray.toDevice(hostData)

	hostW = np.random.randn(outmaps, inmaps, fsize, fsize).astype(np.float32)
	hostBias = np.random.randn(1, outmaps, 1, 1).astype(np.float32)

	W, bias = CPUArray.toDevice(hostW), CPUArray.toDevice(hostBias)
	outdata = conv2d(data, W, bias, stride, pad, dilation)

	dl = dilation
	hostExtData = np.zeros(shape=(batchsize, inmaps, h + 2 * pad, w + 2 * pad))

	hostExtData[:, :, pad:-pad, pad:-pad] = hostData
	hostData = hostExtData

	hostOutData = np.empty(outdata.shape, dtype=np.float32)
	for c in range(outmaps):
		hostOutData[:, c, :, :] = hostBias[0, c, 0, 0]

	for b, oc, ic, y, x, dy, dx in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(outdata.shape[2]), range(outdata.shape[3]),
		range(fsize), range(fsize)
	):
		hostOutData[b, oc, y, x] += hostData[b, ic, y * stride + dy * dl, x * stride + dx * dl] * hostW[oc, ic, dy, dx]

	assert np.allclose(hostOutData, outdata.get())


def maxpool2dTest():
	batchsize, maps, h, w = 3, 2, 6, 6
	size, stride, pad = 3, 2, 1

	hostData = np.full(shape=(batchsize, maps, h + 2 * pad, w + 2 * pad), fill_value=-np.inf, dtype=np.float32)
	hostData[:, :, pad:-pad, pad:-pad] = np.random.randn(batchsize, maps, h, w).astype(np.float32)

	data = CPUArray.toDevice(hostData[:, :, pad:-pad, pad:-pad])
	outdata = pool2d(data, size=size, stride=stride, pad=pad, mode=PoolMode.max)

	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for b, c, y, x in itertools.product(
		range(batchsize), range(maps), range(hostOutData.shape[2]), range(hostOutData.shape[3])
	):
		hostOutData[b, c, y, x] = np.max(hostData[b, c, y * stride:y * stride + size, x * stride:x * stride + size])

	assert np.allclose(hostOutData, outdata.get())


def batchNorm2dTest():
	batchsize, maps, h, w = 4, 5, 3, 2

	hostData = np.random.randn(batchsize, maps, h, w).astype(np.float32)
	data = CPUArray.toDevice(hostData)

	hostScale = np.random.randn(1, maps, 1, 1).astype(np.float32)
	hostBias = np.random.randn(1, maps, 1, 1).astype(np.float32)
	hostMean = np.random.randn(1, maps, 1, 1).astype(np.float32)
	hostVar = np.ones((1, maps, 1, 1)).astype(np.float32) + np.random.randn(1, maps, 1, 1).astype(np.float32)**2

	scale, bias = CPUArray.toDevice(hostScale), CPUArray.toDevice(hostBias)
	mean, var = CPUArray.toDevice(hostMean), CPUArray.toDevice(hostVar)

	outdata = batchNorm2d(data, scale, bias, mean, var, test=True)

	hostNormData = np.empty(hostData.shape, dtype=np.float32)
	hostOutData = np.empty(hostData.shape, dtype=np.float32)

	for c in range(maps):
		hostNormData[:, c, :, :] = (hostData[:, c, :, :] - hostMean[0, c, 0, 0]) / np.sqrt(hostVar[0, c, 0, 0] + 1e-5)
		hostOutData[:, c, :, :] = hostNormData[:, c, :, :] * hostScale[0, c, 0, 0] + hostBias[0, c, 0, 0]

	assert np.allclose(hostOutData, outdata.get())


if __name__ == "__main__":
	unittest()
