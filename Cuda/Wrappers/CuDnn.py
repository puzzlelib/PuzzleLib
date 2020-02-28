import multiprocessing, itertools
from enum import Enum

import numpy as np

from PuzzleLib import Config

from PuzzleLib.Cuda.Driver import CuDnn
from PuzzleLib.Cuda.GPUArray import GPUArray

from PuzzleLib.Cuda.Utils import dtypesSupported


class ConvFwdAlgo(Enum):
	implicitGemm = CuDnn.CONV_FWD_IMPLICIT_GEMM
	implicitPrecompGemm = CuDnn.CONV_FWD_IMPLICIT_PRECOMP_GEMM
	gemm = CuDnn.CONV_FWD_GEMM
	direct = CuDnn.CONV_FWD_DIRECT
	fft = CuDnn.CONV_FWD_FFT
	fftTiling = CuDnn.CONV_FWD_FFT_TILING
	winograd = CuDnn.CONV_FWD_WINOGRAD
	winogradNonfused = CuDnn.CONV_FWD_WINOGRAD_NONFUSED


class ConvBwdDataAlgo(Enum):
	algo0 = CuDnn.CONV_BWD_DATA_ALGO_0
	algo1 = CuDnn.CONV_BWD_DATA_ALGO_1
	fft = CuDnn.CONV_BWD_DATA_FFT
	fftTiling = CuDnn.CONV_BWD_DATA_FFT_TILING
	winograd = CuDnn.CONV_BWD_DATA_WINOGRAD
	winogradNonfused = CuDnn.CONV_BWD_DATA_WINOGRAD_NONFUSED


class ConvBwdFilterAlgo(Enum):
	algo0 = CuDnn.CONV_BWD_PARAM_ALGO_0
	algo1 = CuDnn.CONV_BWD_PARAM_ALGO_1
	fft = CuDnn.CONV_BWD_PARAM_FFT
	algo3 = CuDnn.CONV_BWD_PARAM_ALGO_3
	winograd = CuDnn.CONV_BWD_PARAM_WINOGRAD
	winogradNonfused = CuDnn.CONV_BWD_PARAM_WINOGRAD_NONFUSED
	fftTiling = CuDnn.CONV_BWD_PARAM_FFT_TILING


class PoolMode(Enum):
	max = CuDnn.POOL_MODE_MAX
	avgWithPad = CuDnn.POOL_MODE_AVG_WITH_PAD
	avgNoPad = CuDnn.POOL_MODE_AVG_NO_PAD
	maxDeterminism = CuDnn.POOL_MODE_MAX_DETERMINISM


class SoftMaxMode(Enum):
	perActivation = CuDnn.SOFTMAX_MODE_PER_ACTIVATION
	spatial = CuDnn.SOFTMAX_MODE_SPATIAL


class MathType(Enum):
	default = CuDnn.MATH_DEFAULT
	tensorOp = CuDnn.MATH_TENSOR_OP
	tensorOpAllowConv = CuDnn.MATH_TENSOR_OP_ALLOW_CONVERSION


class ConvPerf:
	def __init__(self, algo, time, memory, determinism, mathType):
		self.algo = algo
		self.time = time
		self.memory = memory
		self.determinism = determinism == 1
		self.mathType = MathType(mathType)


	def toString(self):
		return "%-40s %-25s %-28s %-20s %s" % (
			"Algo %s" % self.algo, "time %.6f secs" % self.time,
			"memory %.6f mbytes" % (self.memory / 1024**2), "determinism=%s" % self.determinism,
			"mathType=%s" % self.mathType
		)


	def __str__(self):
		return self.toString()


	def __repr__(self):
		return self.toString()


def convNdbenchmark(datashape, Wshape, dtype, stride=1, pad=0, dilation=1, groups=1, algoCount=10):
	results = context.convNdbenchmark(datashape, Wshape, dtype, stride, pad, dilation, groups, algoCount)
	results = tuple(
		[ConvPerf(algotype(values[0]), *values[1:]) for values in subresults] for algotype, subresults in
		zip((ConvFwdAlgo, ConvBwdDataAlgo, ConvBwdFilterAlgo), results)
	)

	return results


context = None


def autoinit():
	global context
	context = CuDnn.DnnContext()

	if Config.systemLog:
		print("[%s]: Created CuDnn context (Using version: %s)" % (Config.libname, CuDnn.getVersion()))

	context.enableTensorOps(True)


if context is None and (multiprocessing.current_process().name == "MainProcess" or Config.allowMultiContext):
	autoinit()


def unittest():
	for dtype, atol in dtypesSupported():
		conv2dTest(dtype, atol)
		conv3dTest(dtype, atol)
		convGroupTest(dtype, atol)

		deconv2dTest(dtype, atol)
		deconv3dTest(dtype, atol)
		deconvGroupTest(dtype, atol)

		maxpool2dTest(dtype, atol)
		maxpool3dTest(dtype, atol)

		softmax2dTest(dtype, atol)


def conv2dTest(dtype, atol):
	batchsize, inmaps, h, w = 1, 2, 6, 6
	outmaps, fsize, stride = 4, 2, 2

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(dtype)
	hostW = np.random.randn(outmaps, inmaps, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = GPUArray.toGpu(hostData), GPUArray.toGpu(hostW), GPUArray.toGpu(hostBias)
	outdata = context.convNd(data, W, bias, stride=stride)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	for b, oc, ic, y, x, dy, dx in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(outdata.shape[2]), range(outdata.shape[3]),
		range(fsize), range(fsize)
	):
		hostOutData[b, oc, y, x] += hostData[b, ic, y * stride + dy, x * stride + dx] * hostW[oc, ic, dy, dx]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = context.convNdBackwardData(grad, W, data=data, stride=stride)

	hostInGrad = np.zeros(data.shape).astype(np.float32)

	for b, ic, oc, y, x, dy, dx in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(hostGrad.shape[2]), range(hostGrad.shape[3]),
		range(fsize), range(fsize)
	):
		hostInGrad[b, ic, y * stride + dy, x * stride + dx] += hostW[oc, ic, dy, dx] * hostGrad[b, oc, y, x]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = context.convNdBackwardParams(data, grad, W, stride=stride, withbias=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for b, oc, ic, dy, dx, y, x in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(fsize), range(fsize),
		range(hostGrad.shape[2]), range(hostGrad.shape[3])
	):
		hostWGrad[oc, ic, dy, dx] += hostData[b, ic, y * stride + dy, x * stride + dx] * hostGrad[b, oc, y, x]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def conv3dTest(dtype, atol):
	batchsize, inmaps, d, h, w = 1, 2, 4, 4, 4
	outmaps, fsize, s = 3, 2, 2

	hostData = np.random.randn(batchsize, inmaps, d, h, w).astype(dtype)
	hostW = np.random.randn(outmaps, inmaps, fsize, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = GPUArray.toGpu(hostData), GPUArray.toGpu(hostW), GPUArray.toGpu(hostBias)
	outdata = context.convNd(data, W, bias, stride=s)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	for b, oc, ic, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(outdata.shape[2]),
		range(outdata.shape[3]), range(outdata.shape[4]), range(fsize), range(fsize), range(fsize)
	):
		hostOutData[b, oc, z, y, x] += hostData[b, ic, z * s + dz, y * s + dy, x * s + dx] * hostW[oc, ic, dz, dy, dx]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = context.convNdBackwardData(grad, W, data=data, stride=s)

	hostInGrad = np.zeros(data.shape).astype(np.float32)

	for b, ic, oc, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(hostGrad.shape[2]),
		range(hostGrad.shape[3]), range(hostGrad.shape[4]), range(fsize), range(fsize), range(fsize)
	):
		hostInGrad[b, ic, z * s + dz, y * s + dy, x * s + dx] += hostW[oc, ic, dz, dy, dx] * hostGrad[b, oc, z, y, x]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = context.convNdBackwardParams(data, grad, W, stride=s, withbias=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for b, oc, ic, dz, dy, dx, z, y, x in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(fsize), range(fsize), range(fsize),
		range(hostGrad.shape[2]), range(hostGrad.shape[3]), range(hostGrad.shape[4])
	):
		hostWGrad[oc, ic, dz, dy, dx] += hostData[b, ic, z * s + dz, y * s + dy, x * s + dx] * hostGrad[b, oc, z, y, x]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3, 4), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def convGroupTest(dtype, atol):
	batchsize, inmaps, h, w = 5, 6, 3, 4
	outmaps, groups, fsize = 4, 2, 2

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(dtype)
	hostW = np.random.randn(outmaps, inmaps // groups, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = GPUArray.toGpu(hostData), GPUArray.toGpu(hostW), GPUArray.toGpu(hostBias)
	outdata = context.convNd(data, W, bias, groups=groups)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)
	ingroup, outgroup = inmaps // groups, outmaps // groups

	for g in range(groups):
		hostOutGroup = hostOutData[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostGroup = hostData[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, oc, ic, y, x, dy, dx in itertools.product(
			range(batchsize), range(outgroup), range(ingroup), range(outdata.shape[2]), range(outdata.shape[3]),
			range(fsize), range(fsize)
		):
			hostOutGroup[b, oc, y, x] += hostGroup[b, ic, y + dy, x + dx] * hostW[g * outgroup + oc, ic, dy, dx]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = context.convNdBackwardData(grad, W, groups=groups)

	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for g in range(groups):
		hostGroup = hostGrad[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostInGroup = hostInGrad[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, ic, oc, y, x, dy, dx in itertools.product(
			range(batchsize), range(ingroup), range(outgroup), range(hostGrad.shape[2]), range(hostGrad.shape[3]),
			range(fsize), range(fsize)
		):
			hostInGroup[b, ic, y + dy, x + dx] += hostW[g * outgroup + oc, ic, dy, dx] * hostGroup[b, oc, y, x]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = context.convNdBackwardParams(data, grad, W, groups=groups, withbias=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for g in range(groups):
		hostGrGroup = hostGrad[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostDtGroup = hostData[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, oc, ic, dy, dx, y, x in itertools.product(
			range(batchsize), range(outgroup), range(ingroup), range(fsize), range(fsize),
			range(hostGrad.shape[2]), range(hostGrad.shape[3])
		):
			hostWGrad[g * outgroup + oc, ic, dy, dx] += hostDtGroup[b, ic, y + dy, x + dx] * hostGrGroup[b, oc, y, x]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def deconv2dTest(dtype, atol):
	batchsize, inmaps, h, w = 1, 1, 2, 2
	outmaps, fsize, stride = 1, 3, 2

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(dtype)
	hostW = np.random.randn(inmaps, outmaps, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = GPUArray.toGpu(hostData), GPUArray.toGpu(hostW), GPUArray.toGpu(hostBias)
	outdata = context.convNdBackwardData(data, W, bias, stride=stride)

	hostOutData = np.zeros(outdata.shape).astype(np.float32)

	for b, oc, ic, y, x, dy, dx in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(h), range(w), range(fsize), range(fsize)
	):
		hostOutData[b, oc, y * stride + dy, x * stride + dx] += hostW[ic, oc, dy, dx] * hostData[b, ic, y, x]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = context.convNd(grad, W, stride=stride)

	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b, ic, oc, y, x, dy, dx in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(h), range(w), range(fsize), range(fsize)
	):
		hostInGrad[b, ic, y, x] += hostGrad[b, oc, y * stride + dy, x * stride + dx] * hostW[ic, oc, dy, dx]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = context.convNdBackwardParams(grad, data, W, stride=stride, withbias=True, deconv=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for b, ic, oc, dy, dx, y, x in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(fsize), range(fsize), range(h), range(w)
	):
		hostWGrad[ic, oc, dy, dx] += hostGrad[b, oc, y * stride + dy, x * stride + dx] * hostData[b, ic, y, x]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def deconv3dTest(dtype, atol):
	batchsize, inmaps, d, h, w = 1, 2, 2, 2, 2
	outmaps, fsize, s = 2, 2, 2

	hostData = np.random.randn(batchsize, inmaps, d, h, w).astype(dtype)
	hostW = np.random.randn(inmaps, outmaps, fsize, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = GPUArray.toGpu(hostData), GPUArray.toGpu(hostW), GPUArray.toGpu(hostBias)
	outdata = context.convNdBackwardData(data, W, bias, stride=s)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	for b, oc, ic, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(d), range(h), range(w),
		range(fsize), range(fsize), range(fsize)
	):
		hostOutData[b, oc, z * s + dz, y * s + dy, x * s + dx] += hostW[ic, oc, dz, dy, dx] * hostData[b, ic, z, y, x]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = context.convNd(grad, W, stride=s)

	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b, ic, oc, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(d), range(h), range(w),
		range(fsize), range(fsize), range(fsize)
	):
		hostInGrad[b, ic, z, y, x] += hostGrad[b, oc, z * s + dz, y * s + dy, x * s + dx] * hostW[ic, oc, dz, dy, dx]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = context.convNdBackwardParams(grad, data, W, stride=s, withbias=True, deconv=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for b, ic, oc, dz, dy, dx, z, y, x in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(fsize), range(fsize), range(fsize),
		range(d), range(h), range(w)
	):
		hostWGrad[ic, oc, dz, dy, dx] += hostGrad[b, oc, z * s + dz, y * s + dy, x * s + dx] * hostData[b, ic, z, y, x]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3, 4), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def deconvGroupTest(dtype, atol):
	batchsize, inmaps, h, w = 3, 4, 3, 4
	outmaps, groups, fsize = 4, 2, 2

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(dtype)
	hostW = np.random.randn(inmaps, outmaps // groups, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = GPUArray.toGpu(hostData), GPUArray.toGpu(hostW), GPUArray.toGpu(hostBias)
	outdata = context.convNdBackwardData(data, W, bias, groups=groups)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)
	ingroup, outgroup = inmaps // groups, outmaps // groups

	for g in range(groups):
		hostOutGroup = hostOutData[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostGroup = hostData[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, oc, ic, y, x, dy, dx in itertools.product(
			range(batchsize), range(outgroup), range(ingroup), range(data.shape[2]), range(data.shape[3]),
			range(fsize), range(fsize)
		):
			hostOutGroup[b, oc, y + dy, x + dx] += hostW[g * ingroup + ic, oc, dy, dx] * hostGroup[b, ic, y, x]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = context.convNd(grad, W, groups=groups)

	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for g in range(groups):
		hostGroup = hostGrad[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostInGroup = hostInGrad[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, ic, oc, y, x, dy, dx in itertools.product(
			range(batchsize), range(ingroup), range(outgroup), range(hostInGrad.shape[2]), range(hostInGrad.shape[3]),
			range(fsize), range(fsize)
		):
			hostInGroup[b, ic, y, x] += hostGroup[b, oc, y + dy, x + dx] * hostW[g * ingroup + ic, oc, dy, dx]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = context.convNdBackwardParams(grad, data, W, groups=groups, withbias=True, deconv=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for g in range(groups):
		hostGrGroup = hostGrad[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostDtGroup = hostData[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, oc, ic, dy, dx, y, x in itertools.product(
			range(batchsize), range(outgroup), range(ingroup), range(fsize), range(fsize),
			range(hostData.shape[2]), range(hostData.shape[3])
		):
			hostWGrad[g * ingroup + ic, oc, dy, dx] += hostDtGroup[b, ic, y, x] * hostGrGroup[b, oc, y + dy, x + dx]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def maxpool2dTest(dtype, atol):
	batchsize, maps, h, w = 3, 2, 6, 6
	size, stride, pad = 3, 2, 1

	hostData = np.full(shape=(batchsize, maps, h + 2 * pad, w + 2 * pad), fill_value=np.finfo(dtype).min, dtype=dtype)
	hostData[:, :, pad:-pad, pad:-pad] = np.random.randn(batchsize, maps, h, w).astype(dtype)

	data = GPUArray.toGpu(hostData[:, :, pad:-pad, pad:-pad])
	outdata = context.poolNd(data, size=size, stride=stride, pad=pad, mode=CuDnn.POOL_MODE_MAX)

	hostOutData = np.empty(outdata.shape, dtype=dtype)

	for b, c, y, x in itertools.product(
		range(batchsize), range(maps), range(hostOutData.shape[2]), range(hostOutData.shape[3])
	):
		hostOutData[b, c, y, x] = np.max(hostData[b, c, y * stride:y * stride + size, x * stride:x * stride + size])

	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = context.poolNdBackward(grad, data, outdata, size=size, stride=stride, pad=pad, mode=CuDnn.POOL_MODE_MAX)

	hostInGrad = np.zeros(hostData.shape, dtype=dtype)

	for b, c, y, x, dy, dx in itertools.product(
		range(batchsize), range(maps), range(hostOutData.shape[2]), range(hostOutData.shape[3]),
		range(size), range(size)
	):
		if hostData[b, c, y * stride + dy, x * stride + dx] == hostOutData[b, c, y, x]:
			hostInGrad[b, c, y * stride + dy, x * stride + dx] += hostGrad[b, c, y, x]

	hostInGrad = hostInGrad[:, :, pad:-pad, pad:-pad].astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)


def maxpool3dTest(dtype, atol):
	batchsize, maps, d, h, w = 1, 1, 6, 6, 6
	size, s, pad = 3, 2, 1

	hostData = np.full(
		shape=(batchsize, maps, d + 2 * pad, h + 2 * pad, w + 2 * pad), fill_value=np.finfo(dtype).min, dtype=dtype
	)
	hostData[:, :, pad:-pad, pad:-pad, pad:-pad] = np.random.randn(batchsize, maps, d, h, w).astype(dtype)

	data = GPUArray.toGpu(np.ascontiguousarray(hostData[:, :, pad:-pad, pad:-pad, pad:-pad]))
	outdata = context.poolNd(data, size=size, stride=s, pad=pad, mode=CuDnn.POOL_MODE_MAX)

	hostOutData = np.empty(outdata.shape, dtype=dtype)

	for b, c, z, y, x in itertools.product(
		range(batchsize), range(maps),
		range(hostOutData.shape[2]), range(hostOutData.shape[3]), range(hostOutData.shape[4])
	):
		hostOutData[b, c, z, y, x] = np.max(hostData[b, c, z * s:z * s + size, y * s:y * s + size, x * s:x * s + size])

	assert np.allclose(hostOutData, outdata.get())

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = context.poolNdBackward(grad, data, outdata, size=size, stride=s, pad=pad, mode=CuDnn.POOL_MODE_MAX)

	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b, c, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(maps),
		range(hostOutData.shape[2]), range(hostOutData.shape[3]), range(hostOutData.shape[4]),
		range(size), range(size), range(size)
	):
		if hostData[b, c, z * s + dz, y * s + dy, x * s + dx] == hostOutData[b, c, z, y, x]:
			hostInGrad[b, c, z * s + dz, y * s + dy, x * s + dx] += hostGrad[b, c, z, y, x]

	hostInGrad = hostInGrad[:, :, pad:-pad, pad:-pad, pad:-pad].astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)


def softmax2dTest(dtype, atol):
	batchsize, maps, h, w = 5, 8, 2, 3
	hostData = np.random.randn(batchsize, maps, h, w).astype(dtype)

	data = GPUArray.toGpu(hostData)
	outdata = context.softmaxNd(data)

	def hostSoftmax(tensor):
		e = np.exp(tensor - np.amax(tensor))
		return e / np.sum(e)

	hostOutData = np.empty(outdata.shape, dtype=dtype)

	for b, y, x in itertools.product(range(batchsize), range(h), range(w)):
		hostOutData[b, :, y, x] = hostSoftmax(hostData[b, :, y, x])

	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = context.softmaxNdBackward(grad, outdata)

	hostInGrad = np.empty(ingrad.shape, dtype=dtype)

	def hostSoftmaxBackward(d, gr):
		return d * (gr - np.dot(d, gr))

	for b, y, x in itertools.product(range(batchsize), range(h), range(w)):
		hostInGrad[b, :, y, x] = hostSoftmaxBackward(hostOutData[b, :, y, x], hostGrad[b, :, y, x])

	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)


if __name__ == "__main__":
	unittest()
