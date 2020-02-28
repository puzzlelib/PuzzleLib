from enum import Enum

from PuzzleLib import Config


ConvFwdAlgo = None
ConvBwdFilterAlgo = None
ConvBwdDataAlgo = None
convNd = None
convNdBackwardData = None
convNdBackwardParams = None

convNdbenchmark = None

deconvNd = None
deconvNdBackwardData = None
deconvNdBackwardParams = None

PoolMode = None
poolNd = None
poolNdBackward = None

BatchNormMode = None
batchNormNd = None
batchNormNdBackward = None

SoftMaxMode = None
softmaxNd = None
softmaxNdBackward = None

mapLRN = None
mapLRNBackward = None

crossMapLRN = None
crossMapLRNBackward = None


def autoinit():
	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.opencl:
		initOpenCL()
	elif Config.backend == Config.Backend.cpu:
		initCPU()
	elif Config.backend == Config.Backend.intel:
		initIntel()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	import numpy as np

	from PuzzleLib.Cuda.Wrappers import CuDnn, CuDnnNorm
	from PuzzleLib.Cuda.Wrappers.CuDnn import context
	from PuzzleLib.Cuda.Utils import memoryPool

	global ConvFwdAlgo, ConvBwdDataAlgo, ConvBwdFilterAlgo
	ConvFwdAlgo = CuDnn.ConvFwdAlgo
	ConvBwdDataAlgo = CuDnn.ConvBwdDataAlgo
	ConvBwdFilterAlgo = CuDnn.ConvBwdFilterAlgo

	def wrapConvNd(data, W, bias, stride, pad, dilation, groups, algo):
		return context.convNd(
			data, W, bias.ravel() if bias is not None else None, stride, pad, dilation, groups,
			algo.value, None, memoryPool
		)

	def wrapConvNdBackwardData(grad, W, data, stride, pad, dilation, groups, algo):
		return context.convNdBackwardData(
			grad, W, None, data, stride, pad, dilation, groups, algo.value, None, memoryPool
		)

	def wrapConvNdBackwardParams(data, grad, W, bias, stride, pad, dilation, groups,
								 wgrad, bgrad, scale, momentum, algo):
		return context.convNdBackwardParams(
			data, grad, W, stride, pad, dilation, groups, bias is not None, False,
			wgrad, bgrad.ravel() if bgrad is not None else None, scale, momentum, algo.value, memoryPool
		)

	global convNd, convNdBackwardData, convNdBackwardParams
	convNd = wrapConvNd
	convNdBackwardData = wrapConvNdBackwardData
	convNdBackwardParams = wrapConvNdBackwardParams

	def wrapConvNdbenchmark(datashape, Wshape, stride, pad, dilation, groups, transpose):
		fwdResults, bwdDataResults, bwdParamResults = CuDnn.convNdbenchmark(
			datashape, Wshape, np.float32, stride, pad, dilation, groups
		)

		return fwdResults, bwdParamResults, bwdDataResults

	global convNdbenchmark
	convNdbenchmark = wrapConvNdbenchmark

	def wrapDeconvNd(data, W, bias, stride, pad, dilation, groups, algo):
		return context.convNdBackwardData(
			data, W, bias.ravel() if bias is not None else None, None, stride, pad, dilation, groups,
			algo.value, None, memoryPool
		)

	def wrapDeconvNdBackwardData(grad, W, data, stride, pad, dilation, groups, algo):
		assert data is not None
		return context.convNd(grad, W, None, stride, pad, dilation, groups, algo.value, None, memoryPool)

	def wrapDeconvNdBackwardParams(data, grad, W, bias, stride, pad, dilation, groups,
								   wgrad, bgrad, scale, momentum, algo):
		return context.convNdBackwardParams(
			grad, data, W, stride, pad, dilation, groups, bias is not None, True,
			wgrad, bgrad.ravel() if bgrad is not None else None, scale, momentum, algo.value, memoryPool
		)

	global deconvNd, deconvNdBackwardData, deconvNdBackwardParams
	deconvNd = wrapDeconvNd
	deconvNdBackwardData = wrapDeconvNdBackwardData
	deconvNdBackwardParams = wrapDeconvNdBackwardParams

	def wrapPoolNd(data, size, stride, pad, mode, test):
		return context.poolNd(data, size, stride, pad, mode.value, None, memoryPool), None

	def wrapPoolNdBackward(indata, outdata, grad, _, size, stride, pad, mode):
		return context.poolNdBackward(grad, indata, outdata, size, stride, pad, mode.value, None, memoryPool)

	global PoolMode, poolNd, poolNdBackward
	PoolMode = CuDnn.PoolMode
	poolNd = wrapPoolNd
	poolNdBackward = wrapPoolNdBackward

	global BatchNormMode
	BatchNormMode = CuDnnNorm.BatchNormMode

	def wrapBatchNormNd(data, scale, bias, mean, var, epsilon, factor, test, mode=BatchNormMode.spatial, out=None):
		shape = scale.shape
		result = context.batchNormNd(
			data, mean.ravel(), var.ravel(), scale.ravel(), bias.ravel(), epsilon, factor, test, mode.value, out=out,
			allocator=memoryPool
		)

		if test:
			return result

		outdata, savemean, saveinvvar = result
		return outdata, savemean.reshape(shape), saveinvvar.reshape(shape)

	def wrapBatchNormNdBackward(data, grad, scale, savemean, saveinvvar, epsilon, mode=BatchNormMode.spatial):
		shape = scale.shape
		ingrad, scalegrad, bgrad = context.batchNormNdBackward(
			grad, data, scale.ravel(), savemean.ravel(), saveinvvar.ravel(), epsilon, mode.value,
			allocator=memoryPool
		)

		return ingrad, scalegrad.reshape(shape), bgrad.reshape(shape)

	global batchNormNd, batchNormNdBackward
	batchNormNd = wrapBatchNormNd
	batchNormNdBackward = wrapBatchNormNdBackward

	global SoftMaxMode
	SoftMaxMode = CuDnn.SoftMaxMode

	def wrapSoftmaxNd(data, mode=SoftMaxMode.spatial):
		return context.softmaxNd(data, mode.value, allocator=memoryPool)

	def wrapSoftmaxNdBackward(outdata, grad):
		return context.softmaxNdBackward(grad, outdata, allocator=memoryPool)

	global softmaxNd, softmaxNdBackward
	softmaxNd = wrapSoftmaxNd
	softmaxNdBackward = wrapSoftmaxNdBackward

	def wrapMapLRN(data, means, N, alpha, beta, K, test):
		return context.mapLRN(data, means, N, alpha, beta, K, allocator=memoryPool), None

	def wrapMapLRNBackward(data, _, grad, means, __, N, alpha, beta, K):
		return context.mapLRNBackward(data, grad, means, N, alpha, beta, K, allocator=memoryPool)

	global mapLRN, mapLRNBackward
	mapLRN = wrapMapLRN
	mapLRNBackward = wrapMapLRNBackward

	def wrapCrossMapLRN(data, N, alpha, beta, K, test):
		return context.crossMapLRN(data, N, alpha, beta, K, allocator=memoryPool), None

	def wrapCrossMapLRNBackward(data, outdata, grad, _, N, alpha, beta, K):
		return context.crossMapLRNBackward(data, outdata, grad, N, alpha, beta, K, allocator=memoryPool)

	global crossMapLRN, crossMapLRNBackward
	crossMapLRN = wrapCrossMapLRN
	crossMapLRNBackward = wrapCrossMapLRNBackward


def initOpenCL():
	from PuzzleLib.OpenCL.Wrappers import MIOpen

	global ConvFwdAlgo, ConvBwdDataAlgo, ConvBwdFilterAlgo
	ConvFwdAlgo = MIOpen.ConvFwdAlgo
	ConvBwdDataAlgo = MIOpen.ConvBwdDataAlgo
	ConvBwdFilterAlgo = MIOpen.ConvBwdFilterAlgo

	def wrapConvNd(data, W, bias, stride, pad, dilation, groups, algo):
		assert dilation == (1, 1) and groups == 1
		return MIOpen.conv2d(data, W, bias, stride, pad, algo=algo)

	def wrapConvNdBackwardData(grad, W, data, stride, pad, dilation, groups, algo):
		assert dilation == (1, 1) and groups == 1
		return MIOpen.conv2dBackwardData(grad, W, data, stride, pad, algo=algo)

	def wrapConvNdBackwardParams(data, grad, W, bias, stride, pad, dilation, groups, wgrad, bgrad, scale, momentum,
								 algo):
		assert dilation == (1, 1) and groups == 1
		return MIOpen.conv2dBackwardParams(data, grad, W, bias, stride, pad, wgrad, bgrad, scale, momentum, algo=algo)

	global convNd, convNdBackwardData, convNdBackwardParams
	convNd = wrapConvNd
	convNdBackwardData = wrapConvNdBackwardData
	convNdBackwardParams = wrapConvNdBackwardParams

	def wrapConvNdbenchmark(datashape, Wshape, stride, pad, dilation, groups, transpose):
		assert dilation == (1, 1) and groups == 1
		return MIOpen.conv2dbenchmark(datashape, Wshape, stride, pad,
									  mode=MIOpen.ConvMode.transpose if transpose else MIOpen.ConvMode.conv)

	global convNdbenchmark
	convNdbenchmark = wrapConvNdbenchmark

	def wrapDeconvNd(data, W, bias, stride, pad, dilation, groups, algo):
		assert dilation == (1, 1) and groups == 1
		return MIOpen.conv2d(data, W, bias, stride, pad, mode=MIOpen.ConvMode.transpose, algo=algo)

	def wrapDeconvNdBackwardData(grad, W, data, stride, pad, dilation, groups, algo):
		assert dilation == (1, 1) and groups == 1
		return MIOpen.conv2dBackwardData(grad, W, data, stride, pad, mode=MIOpen.ConvMode.transpose, algo=algo)

	def wrapDeconvNdBackwardParams(data, grad, W, bias, stride, pad, dilation, groups, wgrad, bgrad, scale, momentum,
								   algo):
		assert dilation == (1, 1) and groups == 1
		return MIOpen.conv2dBackwardParams(data, grad, W, bias, stride, pad, wgrad, bgrad, scale, momentum,
										   mode=MIOpen.ConvMode.transpose, algo=algo)

	global deconvNd, deconvNdBackwardData, deconvNdBackwardParams
	deconvNd = wrapDeconvNd
	deconvNdBackwardData = wrapDeconvNdBackwardData
	deconvNdBackwardParams = wrapDeconvNdBackwardParams

	def wrapPoolNd(data, size, stride, pad, mode, test):
		result = MIOpen.pool2d(data, size, stride, pad, mode, test)
		return result if not test else (result, None)

	global PoolMode, poolNd, poolNdBackward
	PoolMode = MIOpen.PoolMode
	poolNd = wrapPoolNd
	poolNdBackward = MIOpen.pool2dBackward

	def wrapBatchNormNd(data, scale, bias, mean, var, epsilon, factor, test, mode=None, out=None):
		return MIOpen.batchNorm2d(data, scale, bias, mean, var, epsilon, factor, test, out=out)

	def wrapBatchNormNdBackward(data, grad, scale, savemean, saveinvvar, epsilon, mode=None):
		return MIOpen.batchNorm2dBackward(data, grad, scale, savemean, saveinvvar, epsilon)

	global BatchNormMode, batchNormNd, batchNormNdBackward
	BatchNormMode = MIOpen.BatchNormMode
	batchNormNd = wrapBatchNormNd
	batchNormNdBackward = wrapBatchNormNdBackward

	global softmaxNd, softmaxNdBackward
	softmaxNd = MIOpen.softmax2d
	softmaxNdBackward = MIOpen.softmax2dBackward

	def wrapMapLRN(data, means, N, alpha, beta, K, test):
		assert means is None
		result = MIOpen.lrn(data, MIOpen.LRNMode.map, N, alpha, beta, K, test)
		return result if not test else (result, None)

	def wrapMapLRNBackward(data, outdata, grad, means, workspace, N, alpha, beta, K):
		assert means is None
		return MIOpen.lrnBackward(data, outdata, grad, workspace, MIOpen.LRNMode.map, N, alpha, beta, K)

	global mapLRN, mapLRNBackward
	mapLRN = wrapMapLRN
	mapLRNBackward = wrapMapLRNBackward

	def wrapCrossMapLRN(data, N, alpha, beta, K, test):
		result = MIOpen.lrn(data, MIOpen.LRNMode.cross, N, alpha, beta, K, test)
		return result if not test else (result, None)

	def wrapCrossMapLRNBackward(data, outdata, grad, workspace, N, alpha, beta, K):
		return MIOpen.lrnBackward(data, outdata, grad, workspace, MIOpen.LRNMode.cross, N, alpha, beta, K)

	global crossMapLRN, crossMapLRNBackward
	crossMapLRN = wrapCrossMapLRN
	crossMapLRNBackward = wrapCrossMapLRNBackward


def initCPU():
	from PuzzleLib.CPU.Wrappers import NumpyDnn

	def wrapConvNd(data, W, bias, stride, pad, dilation, groups, algo):
		assert dilation == (1, 1) and groups == 1
		return NumpyDnn.conv2d(data, W, bias, stride, pad)

	global convNd, convNdBackwardData, convNdBackwardParams
	convNd = wrapConvNd

	def wrapPoolNd(data, size, stride, pad, mode, test):
		return NumpyDnn.pool2d(data, size, stride, pad, mode), None

	global PoolMode, poolNd, poolNdBackward
	PoolMode = NumpyDnn.PoolMode
	poolNd = wrapPoolNd

	class ProxyBatchNormMode(Enum):
		perActivation = 0
		spatial = 1

	def wrapBatchNormNd(data, scale, bias, mean, var, epsilon, factor, test, mode=None, out=None):
		outdata = NumpyDnn.batchNorm2d(data, scale, bias, mean, var, epsilon, test, out=out)
		return outdata if test else (outdata, mean, var)

	global BatchNormMode, batchNormNd
	BatchNormMode = ProxyBatchNormMode
	batchNormNd = wrapBatchNormNd


def initIntel():
	from PuzzleLib.Intel.Wrappers import DNNL

	global ConvFwdAlgo, ConvBwdDataAlgo, ConvBwdFilterAlgo
	ConvFwdAlgo = DNNL.ConvAlgo
	ConvBwdDataAlgo = DNNL.ConvAlgo
	ConvBwdFilterAlgo = DNNL.ConvAlgo

	def wrapConvNd(data, W, bias, stride, pad, dilation, groups, algo):
		assert groups == 1
		return DNNL.convNd(data, W, bias, stride, pad, dilation, algo=algo)

	def wrapConvNdBackwardData(grad, W, data, stride, pad, dilation, groups, algo):
		assert groups == 1
		return DNNL.convNdBackwardData(grad, W, data, stride, pad, dilation, algo=algo)

	def wrapConvNdBackwardParams(data, grad, W, bias, stride, pad, dilation, groups, wgrad, bgrad, scale, momentum,
								 algo):
		assert groups == 1
		return DNNL.convNdBackwardParams(
			data, grad, W, bias, stride, pad, dilation, wgrad, bgrad, scale, momentum, algo=algo
		)

	global convNd, convNdBackwardData, convNdBackwardParams
	convNd = wrapConvNd
	convNdBackwardData = wrapConvNdBackwardData
	convNdBackwardParams = wrapConvNdBackwardParams

	def wrapConvNdbenchmark(datashape, Wshape, stride, pad, dilation, groups, transpose):
		assert groups == 1
		return DNNL.convNdbenchmark(datashape, Wshape, stride, pad, dilation, transpose)

	global convNdbenchmark
	convNdbenchmark = wrapConvNdbenchmark

	def wrapDeconvNd(data, W, bias, stride, pad, dilation, groups, algo):
		assert groups == 1
		return DNNL.convNd(data, W, bias, stride, pad, dilation, algo=algo, transpose=True)

	def wrapDeconvNdBackwardData(grad, W, data, stride, pad, dilation, groups, algo):
		assert groups == 1
		return DNNL.convNdBackwardData(grad, W, data, stride, pad, dilation, algo=algo, transpose=True)

	def wrapDeconvNdBackwardParams(data, grad, W, bias, stride, pad, dilation, groups, wgrad, bgrad, scale, momentum,
								   algo):
		assert groups == 1
		return DNNL.convNdBackwardParams(
			data, grad, W, bias, stride, pad, dilation, wgrad, bgrad, scale, momentum, algo=algo, transpose=True
		)

	global deconvNd, deconvNdBackwardData, deconvNdBackwardParams
	deconvNd = wrapDeconvNd
	deconvNdBackwardData = wrapDeconvNdBackwardData
	deconvNdBackwardParams = wrapDeconvNdBackwardParams

	def wrapPoolNd(data, size, stride, pad, mode, test):
		result = DNNL.poolNd(data, size, stride, pad, mode, test)
		return (result, None) if test else (result[0], result[1:])

	def wrapPoolNdBackward(indata, outdata, grad, workspace, size, stride, pad, mode):
		workspace, desc = workspace
		return DNNL.poolNdBackward(indata, grad, workspace, desc, size, stride, pad, mode)

	global PoolMode, poolNd, poolNdBackward
	PoolMode = DNNL.PoolMode
	poolNd = wrapPoolNd
	poolNdBackward = wrapPoolNdBackward

	class ProxyBatchNormMode(Enum):
		perActivation = 0
		spatial = 1

	def wrapBatchNormNd(data, scale, bias, mean, var, epsilon, factor, test, mode=None, out=None):
		outdata, mean, var, desc = DNNL.batchNormNd(data, scale, bias, mean, var, epsilon, test, out=out)
		return outdata if test else (outdata, mean, var)

	global BatchNormMode, batchNormNd
	BatchNormMode = ProxyBatchNormMode
	batchNormNd = wrapBatchNormNd

	global softmaxNd, softmaxNdBackward
	softmaxNd = DNNL.softmaxNd
	softmaxNdBackward = DNNL.softmaxNdBackward

	def wrapCrossMapLRN(data, N, alpha, beta, K, test):
		result = DNNL.lrn(data, DNNL.LRNMode.cross, N, alpha, beta, K, test)
		return (result[0], result[1:]) if not test else (result, None)

	def wrapCrossMapLRNBackward(data, outdata, grad, workspace, N, alpha, beta, K):
		workspace, desc = workspace
		return DNNL.lrnBackward(data, grad, workspace, desc, DNNL.LRNMode.cross, N, alpha, beta, K)

	global crossMapLRN, crossMapLRNBackward
	crossMapLRN = wrapCrossMapLRN
	crossMapLRNBackward = wrapCrossMapLRNBackward


autoinit()
