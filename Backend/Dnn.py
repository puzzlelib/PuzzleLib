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


instanceNorm2d = None
instanceNorm2dBackward = None


spatialTf = None
spatialTfBackward = None


RNNMode = None
DirectionMode = None

createRnn = None

acquireRnnParams = None
updateRnnParams = None

forwardRnn = None
backwardDataRnn = None
backwardParamsRnn = None

deviceSupportsBatchHint = None


def autoinit():
	if not Config.shouldInit():
		return

	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.hip:
		initHip()
	elif Config.backend == Config.Backend.cpu:
		initCPU()
	elif Config.backend == Config.Backend.intel:
		initIntel()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda import Backend

	backend = initGPU(Backend)
	memoryPool, dnn = backend.memoryPool, backend.dnn

	def wrapPoolNd(data, size, stride, pad, mode, test):
		return dnn.poolNd(data, size, stride, pad, mode.value, None, memoryPool), None

	def wrapPoolNdBackward(indata, outdata, grad, _, size, stride, pad, mode):
		return dnn.poolNdBackward(grad, indata, outdata, size, stride, pad, mode.value, None, memoryPool)

	global PoolMode, poolNd, poolNdBackward
	PoolMode = backend.PoolMode
	poolNd = wrapPoolNd
	poolNdBackward = wrapPoolNdBackward

	def wrapMapLRN(data, means, N, alpha, beta, K, test):
		return dnn.mapLRN(data, means, N, alpha, beta, K, allocator=memoryPool), None

	def wrapMapLRNBackward(data, _, grad, means, __, N, alpha, beta, K):
		return dnn.mapLRNBackward(data, grad, means, N, alpha, beta, K, allocator=memoryPool)

	global mapLRN, mapLRNBackward
	mapLRN = wrapMapLRN
	mapLRNBackward = wrapMapLRNBackward

	def wrapCrossMapLRN(data, N, alpha, beta, K, test):
		return dnn.crossMapLRN(data, N, alpha, beta, K, allocator=memoryPool), None

	def wrapCrossMapLRNBackward(data, outdata, grad, _, N, alpha, beta, K):
		return dnn.crossMapLRNBackward(data, outdata, grad, N, alpha, beta, K, allocator=memoryPool)

	global crossMapLRN, crossMapLRNBackward
	crossMapLRN = wrapCrossMapLRN
	crossMapLRNBackward = wrapCrossMapLRNBackward

	def wrapSpatialTf(data, transform, outshape, getGrid):
		return dnn.spatialTf(data, transform, outshape, getGrid, allocator=memoryPool)

	def wrapSpatialTfBackward(grad, data, grid):
		return dnn.spatialTfBackward(grad, data, grid, allocator=memoryPool)

	global spatialTf, spatialTfBackward
	spatialTf = wrapSpatialTf
	spatialTfBackward = wrapSpatialTfBackward


def initHip():
	from PuzzleLib.Hip import Backend

	backend = initGPU(Backend)
	memoryPool, dnn = backend.memoryPool, backend.dnn

	def wrapPoolNd(data, size, stride, pad, mode, test):
		result = dnn.poolNd(data, size, stride, pad, mode.value, test, None, memoryPool)
		return result if not test else (result, None)

	def wrapPoolNdBackward(indata, outdata, grad, workspace, size, stride, pad, mode):
		return dnn.poolNdBackward(grad, indata, outdata, workspace, size, stride, pad, mode.value, None, memoryPool)

	global PoolMode, poolNd, poolNdBackward
	PoolMode = backend.PoolMode
	poolNd = wrapPoolNd
	poolNdBackward = wrapPoolNdBackward

	def wrapMapLRN(data, means, N, alpha, beta, K, test):
		assert means is None

		result = dnn.lrn(data, N, alpha, beta, K, backend.LRNMode.map.value, test, allocator=memoryPool)
		return result if not test else (result, None)

	def wrapMapLRNBackward(data, outdata, grad, means, workspace, N, alpha, beta, K):
		assert means is None
		return dnn.lrnBackward(
			grad, data, outdata, workspace, N, alpha, beta, K, backend.LRNMode.map.value, allocator=memoryPool
		)

	global mapLRN, mapLRNBackward
	mapLRN = wrapMapLRN
	mapLRNBackward = wrapMapLRNBackward


def initGPU(Backend):
	backend = Backend.getBackend(Config.deviceIdx, initmode=1, logger=Config.getLogger())

	initBaseGPU(backend)
	initInstanceNormGPU(backend)
	initRnnGPU(backend)

	return backend


def initBaseGPU(backend):
	import numpy as np
	memoryPool, dnn = backend.memoryPool, backend.dnn

	global ConvFwdAlgo, ConvBwdDataAlgo, ConvBwdFilterAlgo
	ConvFwdAlgo = backend.ConvFwdAlgo
	ConvBwdDataAlgo = backend.ConvBwdDataAlgo
	ConvBwdFilterAlgo = backend.ConvBwdFilterAlgo

	def wrapConvNd(data, W, bias, stride, pad, dilation, groups, algo):
		return dnn.convNd(
			data, W, bias.ravel() if bias is not None else None, stride, pad, dilation, groups,
			algo.value, None, memoryPool
		)

	def wrapConvNdBackwardData(grad, W, data, stride, pad, dilation, groups, algo):
		return dnn.convNdBackwardData(
			grad, W, None, data, stride, pad, dilation, None, groups, algo.value, None, memoryPool
		)

	def wrapConvNdBackwardParams(data, grad, W, bias, stride, pad, dilation, groups,
								 wgrad, bgrad, scale, momentum, algo):
		return dnn.convNdBackwardParams(
			data, grad, W, stride, pad, dilation, groups, bias is not None, False,
			wgrad, bgrad.ravel() if bgrad is not None else None, scale, momentum, algo.value, memoryPool
		)

	global convNd, convNdBackwardData, convNdBackwardParams
	convNd = wrapConvNd
	convNdBackwardData = wrapConvNdBackwardData
	convNdBackwardParams = wrapConvNdBackwardParams

	def wrapConvNdbenchmark(datashape, Wshape, stride, pad, dilation, groups, transpose):
		fwdResults, bwdDataResults, bwdParamResults = backend.convNdbenchmark(
			datashape, Wshape, np.float32, stride, pad, dilation, groups
		)

		return fwdResults, bwdParamResults, bwdDataResults

	global convNdbenchmark
	convNdbenchmark = wrapConvNdbenchmark

	def wrapDeconvNd(data, W, bias, stride, pad, dilation, postpad, groups, algo):
		return dnn.convNdBackwardData(
			data, W, bias.ravel() if bias is not None else None, None, stride, pad, dilation, postpad, groups,
			algo.value, None, memoryPool
		)

	def wrapDeconvNdBackwardData(grad, W, data, stride, pad, dilation, groups, algo):
		assert data is not None
		return dnn.convNd(grad, W, None, stride, pad, dilation, groups, algo.value, None, memoryPool)

	def wrapDeconvNdBackwardParams(data, grad, W, bias, stride, pad, dilation, groups,
								   wgrad, bgrad, scale, momentum, algo):
		return dnn.convNdBackwardParams(
			grad, data, W, stride, pad, dilation, groups, bias is not None, True,
			wgrad, bgrad.ravel() if bgrad is not None else None, scale, momentum, algo.value, memoryPool
		)

	global deconvNd, deconvNdBackwardData, deconvNdBackwardParams
	deconvNd = wrapDeconvNd
	deconvNdBackwardData = wrapDeconvNdBackwardData
	deconvNdBackwardParams = wrapDeconvNdBackwardParams

	global BatchNormMode
	BatchNormMode = backend.BatchNormMode

	def wrapBatchNormNd(data, scale, bias, mean, var, epsilon, factor, test, mode=BatchNormMode.spatial, out=None):
		shape = scale.shape
		result = dnn.batchNormNd(
			data, mean.ravel(), var.ravel(), scale.ravel(), bias.ravel(), epsilon, factor, test, mode.value, out=out,
			allocator=memoryPool
		)

		if test:
			return result

		outdata, savemean, saveinvvar = result
		return outdata, savemean.reshape(shape), saveinvvar.reshape(shape)

	def wrapBatchNormNdBackward(data, grad, scale, savemean, saveinvvar, epsilon, mode=BatchNormMode.spatial):
		shape = scale.shape
		ingrad, scalegrad, bgrad = dnn.batchNormNdBackward(
			grad, data, scale.ravel(), savemean.ravel(), saveinvvar.ravel(), epsilon, mode.value, allocator=memoryPool
		)

		return ingrad, scalegrad.reshape(shape), bgrad.reshape(shape)

	global batchNormNd, batchNormNdBackward
	batchNormNd = wrapBatchNormNd
	batchNormNdBackward = wrapBatchNormNdBackward

	global SoftMaxMode
	SoftMaxMode = backend.SoftMaxMode

	def wrapSoftmaxNd(data, mode=SoftMaxMode.spatial):
		return dnn.softmaxNd(data, mode.value, allocator=memoryPool)

	def wrapSoftmaxNdBackward(outdata, grad):
		return dnn.softmaxNdBackward(grad, outdata, allocator=memoryPool)

	global softmaxNd, softmaxNdBackward
	softmaxNd = wrapSoftmaxNd
	softmaxNdBackward = wrapSoftmaxNdBackward


def initInstanceNormGPU(backend):
	memoryPool = backend.memoryPool

	def wrapInstanceNorm2d(data, scale, bias, epsilon=1e-5):
		return backend.instanceNorm2d(data, scale.ravel(), bias.ravel(), epsilon, allocator=memoryPool)

	def wrapInstanceNorm2dBackward(grad, data, extscale, savemean, saveinvvar, epsilon, affine):
		return backend.instanceNorm2dBackward(
			grad, data, extscale, savemean, saveinvvar, epsilon, affine, allocator=memoryPool
		)

	global instanceNorm2d, instanceNorm2dBackward
	instanceNorm2d = wrapInstanceNorm2d
	instanceNorm2dBackward = wrapInstanceNorm2dBackward


def initRnnGPU(backend):
	import numpy as np
	memoryPool = backend.memoryPool

	global RNNMode, DirectionMode
	RNNMode = backend.RNNMode
	DirectionMode = backend.DirectionMode

	def wrapCreateRnn(insize, hsize, layers, mode, direction, dropout, seed, batchsize):
		rnn, W, params = backend.createRnn(
			insize, hsize, np.float32, layers, mode=mode, direction=direction, dropout=dropout,
			seed=seed, batchsize=0 if batchsize is None else batchsize
		)

		return rnn, W, {i: layer for i, layer in enumerate(params)}

	def wrapAcquireRnnParams(descRnn, w):
		params = backend.acquireRnnParams(descRnn, w)
		return w, params

	def wrapUpdateRnnParams(descRnn, w, params):
		params = [params[layer] for layer in sorted(params.keys())]
		backend.updateRnnParams(descRnn, w, params)

	global createRnn, acquireRnnParams, updateRnnParams
	createRnn = wrapCreateRnn
	acquireRnnParams = wrapAcquireRnnParams
	updateRnnParams = wrapUpdateRnnParams

	def wrapForwardRnn(data, W, descRnn, test=False):
		return descRnn.forward(data, W, test=test, allocator=memoryPool)

	def wrapBackwardDataRnn(grad, outdata, W, reserve, descRnn):
		ingrad, _, _ = descRnn.backwardData(grad, outdata, W, reserve, allocator=memoryPool)
		return ingrad, reserve

	def wrapBackwardParamsRnn(data, outdata, _, reserve, descRnn):
		return descRnn.backwardParams(data, outdata, reserve, allocator=memoryPool)

	global forwardRnn, backwardDataRnn, backwardParamsRnn
	forwardRnn = wrapForwardRnn
	backwardDataRnn = wrapBackwardDataRnn
	backwardParamsRnn = wrapBackwardParamsRnn

	global deviceSupportsBatchHint
	deviceSupportsBatchHint = backend.deviceSupportsBatchHint

	return backend


def initCPU():
	from PuzzleLib.CPU.Wrappers import NumpyDnn

	def wrapConvNd(data, W, bias, stride, pad, dilation, groups, algo):
		assert groups == 1
		return NumpyDnn.conv2d(data, W, bias, stride, pad, dilation)

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

	global deviceSupportsBatchHint
	deviceSupportsBatchHint = lambda: False


def initIntel():
	from PuzzleLib.Intel.Wrappers import DNNL, DNNLInstanceNorm

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

	def wrapDeconvNd(data, W, bias, stride, pad, dilation, groups, postpad, algo):
		assert groups == 1
		return DNNL.convNd(data, W, bias, stride, pad, dilation, postpad, algo=algo, transpose=True)

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

	def wrapInstanceNorm2d(data, scale, bias, epsilon):
		result = DNNLInstanceNorm.instanceNorm2d(data, scale, bias, epsilon)

		outdata, savemean, savevar, extscale, extbias, desc = result
		return outdata, savemean, savevar, (extscale, extbias, desc)

	def wrapInstanceNorm2dBackward(grad, data, exts, savemean, savevar, epsilon, affine):
		extscale, extbias, desc = exts
		return DNNLInstanceNorm.instanceNorm2dBackward(
			grad, data, extscale, extbias, savemean, savevar, epsilon, desc, affine
		)

	global instanceNorm2d, instanceNorm2dBackward
	instanceNorm2d = wrapInstanceNorm2d
	instanceNorm2dBackward = wrapInstanceNorm2dBackward

	global deviceSupportsBatchHint
	deviceSupportsBatchHint = lambda: False


autoinit()
