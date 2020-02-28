import ctypes
import multiprocessing
from collections import namedtuple
from enum import Enum

import numpy as np

from PuzzleLib import Config
from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Kernels.ElementWise import signKer
from PuzzleLib.OpenCL.ThirdParty import libmiopen
from PuzzleLib.OpenCL.Wrappers import CLBlas
from PuzzleLib.OpenCL.Utils import queue, memoryPool as memPool, checkOffsets


context = None


def autoinit():
	global context
	context = libmiopen.miopenCreateWithStream(queue.int_ptr)

	if Config.systemLog:
		print("[%s]: Created MIOpen context" % (Config.libname, ))

	def finishUp():
		libmiopen.miopenDestroy(context)

	import atexit
	atexit.register(finishUp)


if context is None and (multiprocessing.current_process().name == "MainProcess" or Config.allowMultiContext):
	autoinit()


class DataType(Enum):
	float = libmiopen.miopenDataType["miopenFloat"]


class ConvMode(Enum):
	conv = libmiopen.miopenConvolutionMode["miopenConvolution"]
	transpose = libmiopen.miopenConvolutionMode["miopenTranspose"]


class PoolMode(Enum):
	max = libmiopen.miopenPoolingMode["miopenPoolingMax"]
	avg = libmiopen.miopenPoolingMode["miopenPoolingAverage"]


class ActMode(Enum):
	identity = libmiopen.miopenActivationMode["miopenActivationPASTHRU"]
	sigmoid = libmiopen.miopenActivationMode["miopenActivationLOGISTIC"]
	tanh = libmiopen.miopenActivationMode["miopenActivationTANH"]
	relu = libmiopen.miopenActivationMode["miopenActivationRELU"]
	softPlus = libmiopen.miopenActivationMode["miopenActivationSOFTRELU"]
	absValue = libmiopen.miopenActivationMode["miopenActivationABS"]
	power = libmiopen.miopenActivationMode["miopenActivationPOWER"]
	clippedRelu = libmiopen.miopenActivationMode["miopenActivationCLIPPEDRELU"]
	leakyRelu = libmiopen.miopenActivationMode["miopenActivationLEAKYRELU"]
	elu = libmiopen.miopenActivationMode["miopenActivationELU"]


class ConvPerf:
	def __init__(self, algo, time, memory):
		self.algo = algo
		self.time = time
		self.memory = memory


class ConvFwdAlgo(Enum):
	gemm = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoGEMM"]
	direct = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoDirect"]
	fft = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoFFT"]
	winograd = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoWinograd"]
	implicitGemm = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoImplicitGEMM"]
	staticCompiledGemm = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoStaticCompiledGEMM"]


class ConvBwdFilterAlgo(Enum):
	gemm = libmiopen.miopenConvBwdWeightsAlgorithm["miopenConvolutionBwdWeightsAlgoGEMM"]
	direct = libmiopen.miopenConvBwdWeightsAlgorithm["miopenConvolutionBwdWeightsAlgoDirect"]
	winograd = libmiopen.miopenConvBwdWeightsAlgorithm["miopenConvolutionBwdWeightsAlgoWinograd"]
	implicitGemm = libmiopen.miopenConvBwdWeightsAlgorithm["miopenConvolutionBwdWeightsAlgoImplicitGEMM"]


class ConvBwdDataAlgo(Enum):
	gemm = libmiopen.miopenConvBwdDataAlgorithm["miopenConvolutionBwdDataAlgoGEMM"]
	direct = libmiopen.miopenConvBwdDataAlgorithm["miopenConvolutionBwdDataAlgoDirect"]
	fft = libmiopen.miopenConvBwdDataAlgorithm["miopenConvolutionBwdDataAlgoFFT"]
	winograd = libmiopen.miopenConvBwdDataAlgorithm["miopenConvolutionBwdDataAlgoWinograd"]
	transposeGemm = libmiopen.miopenConvBwdDataAlgorithm["miopenTransposeBwdDataAlgoGEMM"]
	implicitGemm = libmiopen.miopenConvBwdDataAlgorithm["miopenConvolutionBwdDataAlgoImplicitGEMM"]


class LRNMode(Enum):
	map = libmiopen.miopenLRNMode["miopenLRNWithinChannel"]
	cross = libmiopen.miopenLRNMode["miopenLRNCrossChannel"]


class BatchNormMode(Enum):
	perActivation = libmiopen.miopenBatchNormMode["miopenBNPerActivation"]
	spatial = libmiopen.miopenBatchNormMode["miopenBNSpatial"]


DescTensor = namedtuple("DescTensor", "desc shape tensor ptr")

DescConvNd = namedtuple("DescConvNd", "desc mode stride pad")
DescPoolNd = namedtuple("DescPoolNd", "desc size stride pad")
DescLRN = namedtuple("DescLRN", "desc N alpha beta K")


tensorDescCache = []


def createDescribed4dTensor(tensor, allowOffset=False):
	assert tensor.ndim == 4

	tensor = checkOffsets(tensor, allowOffset=allowOffset)[0]
	n, c, h, w = tensor.shape

	if tensor.dtype == np.float32:
		dataType = DataType.float
	else:
		raise NotImplementedError()

	if len(tensorDescCache) > 0:
		desc = tensorDescCache.pop()
	else:
		desc = libmiopen.miopenCreateTensorDescriptor()

	libmiopen.miopenSet4dTensorDescriptor(desc, dataType.value, n, c, h, w)
	return DescTensor(desc=desc, shape=tensor.shape, tensor=tensor, ptr=tensor.int_ptr)


def createDescribedNdTensor(dims, strides, tensor, allowOffset=False):
	if tensor is None:
		dataType = DataType.float
	elif tensor.dtype == np.float32:
		dataType = DataType.float
	else:
		raise NotImplementedError()

	if tensor is not None:
		tensor = checkOffsets(tensor, allowOffset=allowOffset)[0]

	if dims is None:
		if tensor is None:
			raise ValueError()

		dims = tensor.shape

	if strides is None:
		if tensor is None:
			raise ValueError()

		strides = [stride // tensor.dtype.itemsize for stride in tensor.strides]

	if len(tensorDescCache) > 0:
		desc = tensorDescCache.pop()
	else:
		desc = libmiopen.miopenCreateTensorDescriptor()

	libmiopen.miopenSetTensorDescriptor(desc, dataType.value, dims, strides)

	ptr = None if tensor is None else tensor.int_ptr
	shape = None if tensor is None else tensor.shape

	return DescTensor(desc=desc, shape=shape, tensor=tensor, ptr=ptr)


def destroyDescribedTensors(*descTensors):
	for descTensor in descTensors:
		tensorDescCache.append(descTensor.desc)


def toTensorAddTensor(descTensor, descBias):
	libmiopen.miopenConvolutionForwardBias(context, 1.0, descBias.desc, descBias.ptr, 0.0, descTensor.desc,
										   descTensor.ptr)


def createDescribedConv2d(stride=1, pad=0, dilation=1, mode=ConvMode.conv):
	if isinstance(stride, int):
		strides = [stride, stride]
	else:
		strides = stride

	if isinstance(pad, int):
		pads = [pad, pad]
	else:
		pads = pad

	if isinstance(dilation, int):
		dilations = [dilation, dilation]
	else:
		dilations = dilation

	desc = libmiopen.miopenCreateConvolutionDescriptor()
	libmiopen.miopenInitConvolutionDescriptor(desc, mode.value, pads[0], pads[1], strides[0], strides[1],
											  dilations[0], dilations[1])

	return DescConvNd(desc=desc, mode=mode, stride=strides, pad=pads)


def destroyDescribedConv(descConv):
	libmiopen.miopenDestroyConvolutionDescriptor(descConv.desc)


def getConv2dOutShape(descConv, descTensor, descW):
	shape = libmiopen.miopenGetConvolutionForwardOutputDim(descConv.desc, descTensor.desc, descW.desc)
	return shape


def getConv2dInShape(descConv, descTensor, descW, mode=ConvMode.conv):
	outmaps, inmaps, fh, fw = descW.tensor.shape

	hstride, wstride = descConv.stride
	hpad, wpad = descConv.pad

	if mode == ConvMode.conv:
		inh = hstride * (descTensor.tensor.shape[2] - 1) + fh - 2 * hpad
		inw = wstride * (descTensor.tensor.shape[3] - 1) + fw - 2 * wpad
	elif mode == ConvMode.transpose:
		inh = (descTensor.tensor.shape[2] + 2 * hpad - fh) // hstride + 1
		inw = (descTensor.tensor.shape[3] + 2 * wpad - fw) // wstride + 1
	else:
		raise NotImplementedError()

	return descTensor.tensor.shape[0], inmaps, inh, inw


def conv2dWorkspace(descW, descData, descConv, descOutData):
	size = libmiopen.miopenConvolutionForwardGetWorkSpaceSize(context, descW.desc, descData.desc, descConv.desc,
															  descOutData.desc)

	workspace = Driver.empty(queue, (size, ), dtype=np.uint8, allocator=memPool) if size > 0 else None

	ptr = ctypes.c_void_p(int(workspace.int_ptr)) if workspace is not None else None
	size = workspace.nbytes if workspace is not None else 0

	return workspace, ptr, size


def conv2dBackwardDataWorkspace(descGrad, descW, descConv, descInGrad):
	size = libmiopen.miopenConvolutionBackwardDataGetWorkSpaceSize(context, descGrad.desc, descW.desc, descConv.desc,
																   descInGrad.desc)

	workspace = Driver.empty(queue, (size, ), dtype=np.uint8, allocator=memPool) if size > 0 else None

	ptr = ctypes.c_void_p(int(workspace.int_ptr)) if workspace is not None else None
	size = workspace.nbytes if workspace is not None else 0

	return workspace, ptr, size


def conv2dBackwardParamsWorkspace(descGrad, descData, descConv, descWGrad):
	size = libmiopen.miopenConvolutionBackwardWeightsGetWorkSpaceSize(context, descGrad.desc, descData.desc,
																	  descConv.desc, descWGrad.desc)

	workspace = Driver.empty(queue, (size, ), dtype=np.uint8, allocator=memPool) if size > 0 else None

	ptr = ctypes.c_void_p(int(workspace.int_ptr)) if workspace is not None else None
	size = workspace.nbytes if workspace is not None else 0

	return workspace, ptr, size


conv2dCache = {}
conv2dBackwardDataCache = {}
conv2dBackwardParamsCache = {}


def cacheConv2dAlgo(descData, descW, descConv, descOutData, workspace, algo):
	mode = "%s-%s-%s-%s-%s" % (descData.shape, descW.shape, descConv.mode, descConv.stride, descConv.pad)

	perf = conv2dCache.get(mode, None)
	if perf is None:
		ptr, size = workspace
		perf = libmiopen.miopenFindConvolutionForwardAlgorithm(context, descData.desc, descData.ptr, descW.desc,
															   descW.ptr, descConv.desc, descOutData.desc,
															   descOutData.ptr, len(ConvFwdAlgo), ptr, size, 0)

		conv2dCache[mode] = perf

	if algo is None:
		algo = ConvFwdAlgo(perf[0].algo)

	return algo


def cacheConv2dDataAlgo(descGrad, descW, descConv, descInGrad, workspace, algo):
	mode = "%s-%s-%s-%s-%s" % (descGrad.shape, descW.shape, descConv.mode, descConv.stride, descConv.pad)

	perf = conv2dBackwardDataCache.get(mode, None)
	if perf is None:
		ptr, size = workspace
		perf = libmiopen.miopenFindConvolutionBackwardDataAlgorithm(context, descGrad.desc, descGrad.ptr, descW.desc,
																	descW.ptr, descConv.desc, descInGrad.desc,
																	descInGrad.ptr, len(ConvBwdDataAlgo), ptr, size, 0)

		conv2dBackwardDataCache[mode] = perf

	if algo is None:
		algo = ConvBwdDataAlgo(perf[0].algo)

	return algo


def cacheConv2dParamsAlgo(descGrad, descData, descConv, descWGrad, workspace, algo):
	mode = "%s-%s-%s-%s-%s" % (descGrad.shape, descData.shape, descConv.mode, descConv.stride, descConv.pad)

	perf = conv2dBackwardParamsCache.get(mode, None)
	if perf is None:
		ptr, size = workspace
		perf = libmiopen.miopenFindConvolutionBackwardWeightsAlgorithm(context, descGrad.desc, descGrad.ptr,
																	   descData.desc, descData.ptr, descConv.desc,
																	   descWGrad.desc, descWGrad.ptr,
																	   len(ConvBwdFilterAlgo), ptr, size, 0)

		conv2dBackwardParamsCache[mode] = perf

	if algo is None:
		algo = ConvBwdFilterAlgo(perf[0].algo)

	return algo


def conv2d(data, W, bias=None, stride=1, pad=0, mode=ConvMode.conv, algo=None):
	assert data.ndim == W.ndim
	if mode == ConvMode.conv:
		assert data.shape[1] == W.shape[1]
	else:
		assert data.shape[1] == W.shape[0]

	descData = createDescribed4dTensor(data, allowOffset=True)
	descW = createDescribed4dTensor(W)

	descConv = createDescribedConv2d(stride, pad, 1, mode)
	outshape = getConv2dOutShape(descConv, descData, descW)

	descOutData = createDescribed4dTensor(Driver.empty(queue, outshape, dtype=data.dtype, allocator=memPool))

	_, ptr, size = conv2dWorkspace(descW, descData, descConv, descOutData)
	algo = cacheConv2dAlgo(descData, descW, descConv, descOutData, (ptr, size), algo)

	libmiopen.miopenConvolutionForward(context, 1.0, descData.desc, descData.ptr, descW.desc, descW.ptr,
									   descConv.desc, algo.value, 0.0, descOutData.desc, descOutData.ptr, ptr, size)

	if bias is not None:
		assert bias.ndim == data.ndim
		descBias = createDescribed4dTensor(bias)
		toTensorAddTensor(descOutData, descBias)
		destroyDescribedTensors(descBias)

	destroyDescribedConv(descConv)
	destroyDescribedTensors(descData, descOutData, descW)

	return descOutData.tensor


def conv2dBackwardData(grad, W, data=None, stride=1, pad=0, mode=ConvMode.conv, algo=None):
	assert grad.ndim == W.ndim
	if mode == ConvMode.conv:
		assert grad.shape[1] == W.shape[0]
	else:
		assert grad.shape[1] == W.shape[1]

	descGrad = createDescribed4dTensor(grad)
	descW = createDescribed4dTensor(W)

	descConv = createDescribedConv2d(stride, pad, 1, mode)

	if data is None:
		inshape = getConv2dInShape(descConv, descGrad, descW, mode)
	else:
		inshape = data.shape

	descInGrad = createDescribed4dTensor(Driver.empty(queue, inshape, dtype=grad.dtype, allocator=memPool))

	_, ptr, size = conv2dBackwardDataWorkspace(descGrad, descW, descConv, descInGrad)
	algo = cacheConv2dDataAlgo(descGrad, descW, descConv, descInGrad, (ptr, size), algo)

	libmiopen.miopenConvolutionBackwardData(context, 1.0, descGrad.desc, descGrad.ptr, descW.desc, descW.ptr,
											descConv.desc, algo.value, 0.0, descInGrad.desc, descInGrad.ptr, ptr, size)

	destroyDescribedConv(descConv)
	destroyDescribedTensors(descGrad, descInGrad, descW)

	return descInGrad.tensor


def conv2dBackwardParams(data, grad, W, bias=None, stride=1, pad=0, wgrad=None, bgrad=None, scale=1.0, momentum=0.0,
						 mode=ConvMode.conv, algo=None):
	assert data.ndim == grad.ndim
	if mode == ConvMode.conv:
		assert grad.shape[1] == W.shape[0] and data.shape[1] == W.shape[1]
	else:
		assert grad.shape[1] == W.shape[1] and data.shape[1] == W.shape[0]

	descData = createDescribed4dTensor(data, allowOffset=True)
	descGrad = createDescribed4dTensor(grad)

	descConv = createDescribedConv2d(stride, pad, 1, mode)

	if wgrad is not None and scale == 1.0 and momentum == 0.0:
		descWGrad = createDescribed4dTensor(wgrad)
	else:
		descWGrad = createDescribed4dTensor(Driver.zeros(queue, W.shape, dtype=W.dtype, allocator=memPool))

	_, ptr, size = conv2dBackwardParamsWorkspace(descGrad, descData, descConv, descWGrad)
	algo = cacheConv2dParamsAlgo(descGrad, descData, descConv, descWGrad, (ptr, size), algo)

	libmiopen.miopenConvolutionBackwardWeights(context, 1.0, descGrad.desc, descGrad.ptr, descData.desc, descData.ptr,
											   descConv.desc, algo.value, 0.0, descWGrad.desc, descWGrad.ptr,
											   ptr, size)

	if wgrad is not None and scale != 1.0 or momentum != 0.0:
		CLBlas.addVectorToVector(descWGrad.tensor.ravel(), wgrad.ravel(), out=wgrad.ravel(), alpha=scale, beta=momentum)

	tup = (descWGrad.tensor, )
	if bias is not None:
		assert bias.ndim == data.ndim

		if bgrad is not None and scale == 1.0 and momentum == 0.0:
			descBGrad = createDescribed4dTensor(bgrad)
		else:
			descBGrad = createDescribed4dTensor(Driver.empty(queue, bias.shape, dtype=bias.dtype, allocator=memPool))

		libmiopen.miopenConvolutionBackwardBias(context, 1.0, descGrad.desc, descGrad.ptr, 0.0, descBGrad.desc,
												descBGrad.ptr)

		if bgrad is not None and scale != 1.0 or momentum != 0.0:
			CLBlas.addVectorToVector(descBGrad.tensor.ravel(), bgrad.ravel(), out=bgrad.ravel(),
									 alpha=scale, beta=momentum)

		tup = (descWGrad.tensor, descBGrad.tensor)
		destroyDescribedTensors(descBGrad)

	destroyDescribedConv(descConv)
	destroyDescribedTensors(descData, descGrad, descWGrad)

	return tup


def conv2dbenchmark(datashape, Wshape, stride=1, pad=0, mode=ConvMode.conv, algoCount=10, exhaustive=False):
	assert len(datashape) == len(Wshape)
	if mode == ConvMode.conv:
		assert datashape[1] == Wshape[1]
	else:
		assert datashape[1] == Wshape[0]

	descData = createDescribed4dTensor(Driver.zeros(queue, datashape, dtype=np.float32))
	descW = createDescribed4dTensor(Driver.zeros(queue, Wshape, dtype=np.float32))

	descConv = createDescribedConv2d(stride, pad, 1, mode)
	outshape = getConv2dOutShape(descConv, descData, descW)

	descGrad = createDescribed4dTensor(Driver.zeros(queue, outshape, dtype=np.float32))
	descInGrad = createDescribed4dTensor(Driver.zeros(queue, datashape, dtype=np.float32))

	descWGrad = createDescribed4dTensor(Driver.zeros(queue, Wshape, dtype=np.float32))
	descOutData = createDescribed4dTensor(Driver.zeros(queue, outshape, dtype=np.float32))

	_, ptr, size = conv2dWorkspace(descW, descData, descConv, descOutData)

	perfResults = libmiopen.miopenFindConvolutionForwardAlgorithm(context, descData.desc, descData.ptr, descW.desc,
																  descW.ptr, descConv.desc, descOutData.desc,
																  descOutData.ptr, algoCount, ptr, size, exhaustive)

	millisInSec = 1e-3

	fwdResults = []
	for res in perfResults:
		fwdResults.append(ConvPerf(res.algo, res.time * millisInSec, res.memory))

	_, ptr, size = conv2dBackwardParamsWorkspace(descGrad, descData, descConv, descWGrad)

	perfResults = libmiopen.miopenFindConvolutionBackwardWeightsAlgorithm(context, descGrad.desc, descGrad.ptr,
																		  descData.desc, descData.ptr, descConv.desc,
																		  descWGrad.desc, descWGrad.ptr, algoCount,
																		  ptr, size, exhaustive)

	bwdParamResults = []
	for res in perfResults:
		bwdParamResults.append(ConvPerf(res.algo, res.time * millisInSec, res.memory))

	_, ptr, size = conv2dBackwardDataWorkspace(descGrad, descW, descConv, descInGrad)

	perfResults = libmiopen.miopenFindConvolutionBackwardDataAlgorithm(context, descGrad.desc, descGrad.ptr, descW.desc,
																	   descW.ptr, descConv.desc, descInGrad.desc,
																	   descInGrad.ptr, algoCount, ptr, size, exhaustive)

	bwdDataResults = []
	for res in perfResults:
		bwdDataResults.append(ConvPerf(res.algo, res.time * millisInSec, res.memory))

	destroyDescribedTensors(descData, descGrad, descInGrad, descOutData, descW, descWGrad)
	destroyDescribedConv(descConv)

	return fwdResults, bwdParamResults, bwdDataResults


def createDescribedPool2d(size=2, stride=2, pad=0, mode=PoolMode.max):
	if isinstance(stride, int):
		hstride = wstride = stride
		stride = [hstride, wstride]
	else:
		hstride, wstride = stride

	if isinstance(size, int):
		hsize = wsize = size
		size = [hsize, wsize]
	else:
		hsize, wsize = size

	if isinstance(pad, int):
		hpad = wpad = pad
		pad = [hpad, wpad]
	else:
		hpad, wpad = pad

	desc = libmiopen.miopenCreatePoolingDescriptor()
	libmiopen.miopenSet2dPoolingDescriptor(desc, mode.value, hsize, wsize, hpad, wpad, hstride, wstride)

	return DescPoolNd(desc=desc, size=size, stride=stride, pad=pad)


def destroyDescribedPool(descPool):
	libmiopen.miopenDestroyPoolingDescriptor(descPool.desc)


def getPool2dOutShape(descPool, descTensor):
	hsize, wsize = descPool.size
	hpad, wpad = descPool.pad
	hstride, wstride = descPool.stride

	outh = (descTensor.shape[2] + 2 * hpad - hsize) // hstride + 1
	outw = (descTensor.shape[3] + 2 * wpad - wsize) // wstride + 1

	return descTensor.shape[:2] + (outh, outw)


def pool2d(data, size=2, stride=2, pad=0, mode=PoolMode.max, test=False):
	descData = createDescribed4dTensor(data, allowOffset=True)

	descPool = createDescribedPool2d(size, stride, pad, mode)
	outshape = getPool2dOutShape(descPool, descData)

	descOutData = createDescribed4dTensor(Driver.empty(queue, outshape, dtype=data.dtype, allocator=memPool))

	workspace, ptr, size = None, None, 0
	if not test:
		size = libmiopen.miopenPoolingGetWorkSpaceSize(descOutData.desc)
		workspace = Driver.empty(queue, (size, ), dtype=np.uint8, allocator=memPool)

		ptr = workspace.int_ptr

	libmiopen.miopenPoolingForward(context, descPool.desc, 1.0, descData.desc, descData.ptr, 0.0, descOutData.desc,
								   descOutData.ptr, not test, ptr, size)

	destroyDescribedTensors(descData, descOutData)
	destroyDescribedPool(descPool)

	if test:
		return descOutData.tensor
	else:
		return descOutData.tensor, workspace


def pool2dBackward(indata, outdata, grad, workspace, size=2, stride=2, pad=0, mode=PoolMode.max):
	descInData = createDescribed4dTensor(indata, allowOffset=True)
	descOutData = createDescribed4dTensor(outdata)
	descGrad = createDescribed4dTensor(grad)

	descPool = createDescribedPool2d(size, stride, pad, mode)

	descInGrad = createDescribed4dTensor(Driver.empty(queue, indata.shape, dtype=grad.dtype, allocator=memPool))

	libmiopen.miopenPoolingBackward(context, descPool.desc, 1.0, descOutData.desc, descOutData.ptr, descGrad.desc,
									descGrad.ptr, descInData.desc, descInData.ptr, 0.0, descInGrad.desc, descInGrad.ptr,
									workspace.int_ptr)

	destroyDescribedTensors(descInData, descOutData, descGrad, descInGrad)
	destroyDescribedPool(descPool)

	return descInGrad.tensor


def softmax2d(data):
	descData = createDescribed4dTensor(data, allowOffset=True)
	descOutData = createDescribed4dTensor(Driver.empty(queue, data.shape, dtype=data.dtype, allocator=memPool))

	libmiopen.miopenSoftmaxForward(context, 1.0, descData.desc, descData.ptr, 0.0, descOutData.desc, descOutData.ptr)

	destroyDescribedTensors(descData, descOutData)

	return descOutData.tensor


def softmax2dBackward(outdata, grad):
	descOutData = createDescribed4dTensor(outdata)
	descGrad = createDescribed4dTensor(grad)

	descInGrad = createDescribed4dTensor(Driver.empty(queue, grad.shape, dtype=grad.dtype, allocator=memPool))

	libmiopen.miopenSoftmaxBackward(context, 1.0, descOutData.desc, descOutData.ptr, descGrad.desc, descGrad.ptr,
									0.0, descInGrad.desc, descInGrad.ptr)

	destroyDescribedTensors(descOutData, descGrad, descInGrad)

	return descInGrad.tensor


def createDescribedLRN(mode, N=5, alpha=1e-4, beta=0.75, K=2.0):
	desc = libmiopen.miopenCreateLRNDescriptor()
	libmiopen.miopenSetLRNDescriptor(desc, mode.value, N, alpha, beta, K)

	return DescLRN(desc=desc, N=N, alpha=alpha, beta=beta, K=K)


def destroyDescribedLRN(descLRN):
	libmiopen.miopenDestroyLRNDescriptor(descLRN.desc)


def lrn(data, mode=LRNMode.map, N=5, alpha=1e-4, beta=0.75, K=2.0, test=False):
	descData = createDescribed4dTensor(data, allowOffset=True)
	descOutData = createDescribed4dTensor(Driver.empty(queue, data.shape, dtype=data.dtype, allocator=memPool))

	descLRN = createDescribedLRN(mode, N, alpha, beta, K)

	workspace, ptr = None, None
	if not test:
		size = libmiopen.miopenLRNGetWorkSpaceSize(descOutData.desc)
		workspace = Driver.empty(queue, (size, ), dtype=np.uint8, allocator=memPool)

		ptr = workspace.int_ptr

	libmiopen.miopenLRNForward(context, descLRN.desc, 1.0, descData.desc, descData.ptr, 0.0,
							   descOutData.desc, descOutData.ptr, not test, ptr)

	if mode == LRNMode.cross:
		signKer(descOutData.tensor, descOutData.tensor, descData.tensor)

	destroyDescribedTensors(descData, descOutData)
	destroyDescribedLRN(descLRN)

	if test:
		return descOutData.tensor
	else:
		return descOutData.tensor, workspace


def lrnBackward(data, outdata, grad, workspace, mode=LRNMode.map, N=5, alpha=1e-4, beta=0.75, K=2.0):
	descData = createDescribed4dTensor(data, allowOffset=True)
	descOutData = createDescribed4dTensor(outdata)
	descGrad = createDescribed4dTensor(grad)

	descInGrad = createDescribed4dTensor(Driver.empty(queue, data.shape, dtype=grad.dtype, allocator=memPool))

	descLRN = createDescribedLRN(mode, N, alpha, beta, K)

	libmiopen.miopenLRNBackward(context, descLRN.desc, 1.0, descOutData.desc, descOutData.ptr, descGrad.desc,
								descGrad.ptr, descData.desc, descData.ptr, 0.0, descInGrad.desc, descInGrad.ptr,
								workspace.int_ptr)

	destroyDescribedTensors(descData, descOutData, descGrad, descInGrad)
	destroyDescribedLRN(descLRN)

	return descInGrad.tensor


def deriveBatchNormShape(data):
	descData = createDescribed4dTensor(data)

	desc = libmiopen.miopenCreateTensorDescriptor()
	libmiopen.miopenDeriveBNTensorDescriptor(desc, descData.desc, BatchNormMode.spatial.value)
	_, n, c, h, w, _, _, _, _ = libmiopen.miopenGet4dTensorDescriptor(desc)

	libmiopen.miopenDestroyTensorDescriptor(desc)
	destroyDescribedTensors(descData)

	return n, c, h, w


def batchNorm2d(data, scale, bias, mean, var, epsilon=1e-5, factor=1.0, test=False, mode=BatchNormMode.spatial,
				out=None):
	assert data.ndim == scale.ndim and scale.ndim == bias.ndim and bias.ndim == mean.ndim and mean.ndim == var.ndim
	checkOffsets(bias, mean, var)

	descData = createDescribed4dTensor(data, allowOffset=True)
	descScale = createDescribed4dTensor(scale)

	if out is None:
		descOutData = createDescribed4dTensor(Driver.empty(queue, data.shape, dtype=data.dtype, allocator=memPool))
	else:
		descOutData = createDescribed4dTensor(out)

	if test:
		savemean, saveinvvar = None, None
		libmiopen.miopenBatchNormalizationForwardInference(context, mode.value, 1.0, 0.0, descData.desc, descData.ptr,
														   descOutData.desc, descOutData.ptr, descScale.desc,
														   descScale.ptr, bias.int_ptr, mean.int_ptr, var.int_ptr,
														   epsilon)
	else:
		savemean = Driver.empty(queue, mean.shape, dtype=data.dtype, allocator=memPool)
		saveinvvar = Driver.empty(queue, var.shape, dtype=data.dtype, allocator=memPool)

		libmiopen.miopenBatchNormalizationForwardTraining(context, mode.value, 1.0, 0.0, descData.desc, descData.ptr,
														  descOutData.desc, descOutData.ptr, descScale.desc,
														  descScale.ptr, bias.int_ptr, factor,
														  mean.int_ptr, var.int_ptr, epsilon,
														  savemean.int_ptr, saveinvvar.int_ptr)

	destroyDescribedTensors(descData, descScale, descOutData)

	if test:
		return descOutData.tensor
	else:
		return descOutData.tensor, savemean, saveinvvar


def batchNorm2dBackward(data, grad, scale, savemean=None, saveinvvar=None, epsilon=1e-5, mode=BatchNormMode.spatial):
	assert data.ndim == grad.ndim and grad.ndim == scale.ndim

	if savemean is not None:
		assert scale.ndim == savemean.ndim
		checkOffsets(savemean)
		savemean = savemean.int_ptr

	if saveinvvar is not None:
		assert scale.ndim == saveinvvar.ndim
		checkOffsets(saveinvvar)
		saveinvvar = saveinvvar.int_ptr

	descData = createDescribed4dTensor(data, allowOffset=True)
	descGrad = createDescribed4dTensor(grad)
	descScale = createDescribed4dTensor(scale)

	descInGrad = createDescribed4dTensor(Driver.empty(queue, grad.shape, dtype=grad.dtype, allocator=memPool))
	scalegrad = Driver.empty(queue, scale.shape, dtype=scale.dtype, allocator=memPool)
	bgrad = Driver.empty(queue, scale.shape, dtype=scale.dtype, allocator=memPool)

	libmiopen.miopenBatchNormalizationBackward(context, mode.value, 1.0, 0.0, 1.0, 0.0, descData.desc, descData.ptr,
											   descGrad.desc, descGrad.ptr, descInGrad.desc, descInGrad.ptr,
											   descScale.desc, descScale.ptr, scalegrad.int_ptr, bgrad.int_ptr,
											   epsilon, savemean, saveinvvar)

	destroyDescribedTensors(descData, descGrad, descInGrad, descScale)
	return descInGrad.tensor, scalegrad, bgrad


def unittest():
	conv2dTest()
	deconv2dTest()
	maxpool2dTest()
	softmaxTest()
	mapLRNTest()
	crossMapLRNTest()
	batchNorm2dTest()
	batchNormTest()


def conv2dTest():
	batchsize, inmaps, h, w = 1, 2, 6, 6
	fsize, outmaps = 2, 4

	data = Driver.to_device(queue, np.random.randn(batchsize, inmaps, h, w).astype(np.float32))

	W = Driver.to_device(queue, np.random.randn(outmaps, inmaps, fsize, fsize).astype(np.float32))
	bias = Driver.to_device(queue, np.random.randn(1, outmaps, 1, 1).astype(np.float32))

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

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))
	ingrad = conv2dBackwardData(grad, W)

	hostInGrad, hostGrad = np.zeros(data.shape).astype(np.float32), grad.get()

	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for y in range(hostGrad.shape[2]):
					for x in range(hostGrad.shape[3]):
						for dy in range(fsize):
							for dx in range(fsize):
								hostInGrad[b, ic, y + dy, x + dx] += hostW[oc, ic, dy, dx] * hostGrad[b, oc, y, x]

	assert np.allclose(hostInGrad, ingrad.get())

	wgrad, bgrad = conv2dBackwardParams(data, grad, W, bias)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for dy in range(fsize):
					for dx in range(fsize):
						for y in range(hostGrad.shape[2]):
							for x in range(hostGrad.shape[3]):
								hostWGrad[oc, ic, dy, dx] += hostData[b, ic, y + dy, x + dx] * hostGrad[b, oc, y, x]

	assert np.allclose(hostWGrad, wgrad.get())

	hostBGrad = np.empty(hostBias.shape, dtype=np.float32)
	for oc in range(outmaps):
		hostBGrad[0, oc, 0, 0] = np.sum(hostGrad[:, oc, :, :])

	assert np.allclose(hostBGrad, bgrad.get())


def deconv2dTest():
	batchsize, inmaps, h, w = 1, 1, 2, 2
	fsize, stride, outmaps = 3, 2, 1

	data = Driver.to_device(queue, np.random.randn(batchsize, inmaps, h, w).astype(np.float32))

	W = Driver.to_device(queue, np.random.randn(inmaps, outmaps, fsize, fsize).astype(np.float32))
	bias = Driver.to_device(queue, np.random.randn(1, outmaps, 1, 1).astype(np.float32))

	outdata = conv2d(data, W, bias, stride=stride, mode=ConvMode.transpose)

	hostOutData = np.zeros(outdata.shape).astype(np.float32)
	for i in range(0, hostOutData.shape[2] - fsize + 1, stride):
		for j in range(0, hostOutData.shape[3] - fsize + 1, stride):
			hostOutData[0, 0, i:fsize + i, j:fsize + j] += W.get()[0, 0] * data.get()[0, 0, i // stride, j // stride]

	hostOutData += bias.get()
	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))

	ingrad = conv2dBackwardData(grad, W, stride=stride, mode=ConvMode.transpose)

	hostInGrad = np.zeros(data.shape, dtype=np.float32)
	for i in range(0, hostInGrad.shape[2]):
		for j in range(0, hostInGrad.shape[3]):
			y, x = i * stride, j * stride
			hostInGrad[0, 0, i, j] += np.dot(W.get()[0, 0].ravel(), grad.get()[0, 0, y:y + fsize, x:x + fsize].ravel())

	assert np.allclose(hostInGrad, ingrad.get())

	wgrad, bgrad = conv2dBackwardParams(data, grad, W, bias, stride=stride, mode=ConvMode.transpose)

	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)
	for i in range(0, hostOutData.shape[2] - fsize + 1, stride):
		for j in range(0, hostOutData.shape[3] - fsize + 1, stride):
			hostWGrad[0, 0] += grad.get()[0, 0, i:i + fsize, j:j + fsize] * data.get()[0, 0, i // stride, j // stride]

	assert np.allclose(hostWGrad, wgrad.get())

	hostBGrad = np.sum(grad.get())
	assert np.allclose(hostBGrad, bgrad.get())


def maxpool2dTest():
	batchsize, maps, h, w = 1, 1, 8, 8
	data = Driver.to_device(queue, np.random.randn(batchsize, maps, h, w).astype(np.float32))

	outdata, workspace = pool2d(data, test=False)

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

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))
	pool2dBackward(data, outdata, grad, workspace)


def softmaxTest():
	batchsize, maps = 5, 8
	data = Driver.to_device(queue, np.random.randn(batchsize, maps, 1, 1).astype(np.float32))

	outdata = softmax2d(data)

	def hostSoftmax(w):
		e = np.exp(w - np.amax(w))
		p = e / np.sum(e)
		return p

	hostData = data.get().reshape(batchsize, maps)
	hostOutData = np.vstack([hostSoftmax(hostData[i]) for i in range(batchsize)])
	assert np.allclose(hostOutData, outdata.get().reshape(batchsize, maps))

	grad = Driver.to_device(queue, np.random.randn(batchsize, maps, 1, 1).astype(np.float32))
	ingrad = softmax2dBackward(outdata, grad)

	def hostSoftmaxBackward(outdat, gr):
		ingr = np.zeros(outdat.shape, dtype=np.float32)
		for i in range(ingr.shape[0]):
			ingr[i] += outdat[i] * gr[i]

			for j in range(outdat.shape[0]):
				ingr[i] -= outdat[i] * outdat[j] * gr[j]
		return ingr

	hostGrad = grad.get().reshape(batchsize, maps)
	hostInGrad = np.vstack([hostSoftmaxBackward(hostOutData[i], hostGrad[i]) for i in range(batchsize)])
	assert np.allclose(hostInGrad, ingrad.get().reshape(batchsize, maps))


def mapLRNTest():
	h, w = 10, 10
	N, alpha, beta, K = 5, 1.0, 0.5, 2.0

	lookBehind = int((N - 1) / 2)
	lookAhead = N - lookBehind

	data = Driver.to_device(queue, np.random.randn(1, 1, h, w).astype(np.float32))
	outdata, workspace = lrn(data, mode=LRNMode.map, N=N, alpha=alpha, beta=beta, K=K)

	hostData = data.get().reshape(h, w).astype(np.float32)
	norms = np.empty((h, w), dtype=np.float32)
	for i in range(h):
		for j in range(w):
			norm = 0.0
			for m in range(max(0, i - lookBehind), min(h, i + lookAhead)):
				for n in range(max(0, j - lookBehind), min(w, j + lookAhead)):
					norm += hostData[m, n]**2
			norms[i, j] = K + norm * alpha / N / N

	hostOutData = hostData / norms**beta
	assert np.allclose(hostOutData, outdata.reshape(h, w).get())

	grad = Driver.to_device(queue, np.random.randn(1, 1, h, w).astype(np.float32))
	ingrad = lrnBackward(data, outdata, grad, workspace, mode=LRNMode.map, N=N, alpha=alpha, beta=beta, K=K)

	hostGrad = grad.get().reshape(h, w).astype(np.float32)
	hostInGrad = np.zeros((h, w), dtype=np.float32)
	c = 2.0 * alpha * beta / N / N
	for i in range(h):
		for j in range(w):
			hostInGrad[i, j] += hostGrad[i, j] / norms[i, j]**beta

			for m in range(max(0, i - lookBehind), min(h, i + lookAhead)):
				for n in range(max(0, j - lookBehind), min(w, j + lookAhead)):
					hostInGrad[i, j] -= hostGrad[m, n] * c * hostData[i, j] * hostData[m, n] / norms[m, n]**(beta+1)

	assert np.allclose(hostInGrad, ingrad.reshape(h, w).get())


def crossMapLRNTest():
	maps = 10
	N, alpha, beta, K = 5, 1.0, 0.5, 2.0

	lookBehind = int((N - 1) / 2)
	lookAhead = N - lookBehind

	data = Driver.to_device(queue, np.random.randn(1, maps, 1, 1).astype(np.float32))
	outdata, workspace = lrn(data, mode=LRNMode.cross, N=N, alpha=alpha, beta=beta, K=K)

	hostData = data.get().reshape(maps, ).astype(np.float32)
	norms = np.empty((maps, ), dtype=np.float32)
	for i in range(maps):
		norm = 0.0
		for j in range(max(0, i - lookBehind), min(maps, i + lookAhead)):
			norm += hostData[j]**2
		norms[i] = K + norm * alpha / N

	hostOutData = hostData / norms**beta
	assert np.allclose(hostOutData, outdata.reshape(maps, ).get())

	grad = Driver.to_device(queue, np.random.randn(1, maps, 1, 1).astype(np.float32))
	ingrad = lrnBackward(data, outdata, grad, workspace, mode=LRNMode.cross, N=N, alpha=alpha, beta=beta, K=K)

	hostGrad = grad.get().reshape(maps, ).astype(np.float32)
	hostInGrad = np.zeros((maps, ), dtype=np.float32)
	k = 2.0 * alpha * beta / N
	for i in range(maps):
		hostInGrad[i] += hostGrad[i] / norms[i]**beta

		for j in range(max(0, i - lookBehind), min(maps, i + lookAhead)):
			hostInGrad[j] -= hostGrad[i] * k * hostData[i] * hostData[j] / norms[i]**(beta + 1)

	assert np.allclose(hostInGrad, ingrad.reshape(maps, ).get())


def batchNorm2dTest():
	batchsize, maps, h, w = 4, 5, 1, 1
	data = Driver.to_device(queue, np.random.randn(batchsize, maps, h, w).astype(np.float32))
	hostData = data.get()

	scale = Driver.to_device(queue, np.random.randn(*deriveBatchNormShape(data)).astype(np.float32))
	bias = Driver.to_device(queue, np.random.randn(*deriveBatchNormShape(data)).astype(np.float32))
	mean = Driver.zeros(queue, deriveBatchNormShape(data), dtype=np.float32)
	var = Driver.to_device(queue, np.ones(deriveBatchNormShape(data), dtype=np.float32))

	outdata, savemean, saveinvvar = batchNorm2d(data, scale, bias, mean, var, out=data)

	hostScale, hostBias = scale.get(), bias.get()
	hostNormData = np.empty(hostData.shape, dtype=np.float32)
	hostOutData = np.empty(hostData.shape, dtype=np.float32)
	hostMean = np.zeros(scale.shape, dtype=np.float32)
	hostInvVar = np.zeros(scale.shape, dtype=np.float32)
	for c in range(maps):
		for b in range(batchsize):
			hostMean[0, c, 0, 0] += np.sum(hostData[b, c])
		hostMean[0, c, 0, 0] /= (batchsize * w * h)

		for b in range(batchsize):
			hostInvVar[0, c, 0, 0] += np.sum((hostData[b, c] - hostMean[0, c, 0, 0])**2)
		hostInvVar[0, c, 0, 0] /= (batchsize * w * h)

		hostInvVar[0, c, 0, 0] = 1.0 / np.sqrt(hostInvVar[0, c, 0, 0] + 1e-5)
		hostNormData[:, c, :, :] = (hostData[:, c, :, :] - hostMean[0, c, 0, 0]) * hostInvVar[0, c, 0, 0]
		hostOutData[:, c, :, :] = hostNormData[:, c, :, :] * hostScale[0, c, 0, 0] + hostBias[0, c, 0, 0]

	assert np.allclose(hostMean, mean.get())
	assert np.allclose(hostInvVar, saveinvvar.get())
	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(batchsize, maps, h, w).astype(np.float32))

	data = Driver.to_device(queue, hostData)
	ingrad, scalegrad, bgrad = batchNorm2dBackward(data, grad, scale, savemean, saveinvvar)

	hostGrad = grad.get()
	hostInGrad, hostScaleGrad = np.empty(hostGrad.shape, dtype=np.float32), np.empty(hostScale.shape, dtype=np.float32)
	hostBiasGrad, hostMeanGrad = np.empty(hostBias.shape, dtype=np.float32), np.empty(hostMean.shape, dtype=np.float32)
	hostVarGrad = np.empty(hostInvVar.shape, dtype=np.float32)
	for c in range(maps):
		hostBiasGrad[0, c, 0, 0] = np.sum(hostGrad[:, c, :, :])
		hostScaleGrad[0, c, 0, 0] = np.sum(hostGrad[:, c, :, :] * hostNormData[:, c, :, :])

		hostMeanGrad[0, c, 0, 0] = np.sum(hostGrad[:, c, :, :]) * hostScale[0, c, 0, 0] * -hostInvVar[0, c, 0, 0]
		hostVarGrad[0, c, 0, 0] = np.sum(hostGrad[:, c, :, :] * (hostData[:, c, :, :] - hostMean[0, c, 0, 0])) * \
								  hostScale[0, c, 0, 0] * -0.5 * hostInvVar[0, c, 0, 0]**3

		hostInGrad[:, c, :, :] = hostGrad[:, c, :, :] * hostScale[0, c, 0, 0] * hostInvVar[0, c, 0, 0] + \
								 hostVarGrad[0,c,0,0] * 2/(batchsize*w*h) * (hostData[:,c,:,:] - hostMean[0,c,0,0]) + \
								 hostMeanGrad[0, c, 0, 0] / (batchsize * w * h)
	assert np.allclose(hostInGrad, ingrad.get())
	assert np.allclose(hostScaleGrad, scalegrad.get())
	assert np.allclose(hostBiasGrad, bgrad.get())

	batchNorm2d(data, scale, bias, mean, var, test=True)


def batchNormTest():
	batchsize, size = 4, 5

	data = Driver.to_device(queue, np.random.randn(batchsize, size, 1, 1).astype(np.float32))
	hostData = data.get().squeeze()

	scale = Driver.to_device(queue, np.random.randn(1, size, 1, 1).astype(np.float32))
	bias = Driver.to_device(queue, np.random.randn(1, size, 1, 1).astype(np.float32))
	mean = Driver.zeros(queue, (1, size, 1, 1), dtype=np.float32)
	var = Driver.to_device(queue, np.ones((1, size, 1, 1), dtype=np.float32))

	outdata, savemean, saveinvvar = batchNorm2d(data, scale, bias, mean, var, mode=BatchNormMode.perActivation,
												out=data)

	hostMean = np.mean(hostData, axis=0, keepdims=False)
	hostInvVar = 1.0 / np.sqrt(np.sum((hostData - hostMean[np.newaxis, :])**2, axis=0) / batchsize + 1e-5)

	hostNormData = (hostData - hostMean) * hostInvVar
	hostScale = scale.get().squeeze()
	hostBias = bias.get().squeeze()
	hostOutData = hostNormData * hostScale + hostBias

	assert np.allclose(hostMean, savemean.get().squeeze())
	assert np.allclose(hostInvVar, saveinvvar.get().squeeze())
	assert np.allclose(hostOutData, outdata.get().squeeze())

	grad = Driver.to_device(queue, np.random.randn(batchsize, size, 1, 1).astype(np.float32))

	data = Driver.to_device(queue, hostData).reshape(batchsize, size, 1, 1)
	ingrad, scalegrad, bgrad = batchNorm2dBackward(data, grad, scale, savemean, saveinvvar, mode=BatchNormMode.spatial)

	hostGrad = grad.get().squeeze()

	hostBGrad = np.sum(hostGrad, axis=0)
	hostScaleGrad = np.sum(hostGrad * hostNormData, axis=0)
	hostMeanGrad = np.sum(hostGrad, axis=0) * hostScale * -hostInvVar
	hostVarGrad = np.sum(hostGrad * (hostData - hostMean[np.newaxis, :]), axis=0) * \
				  hostScale[np.newaxis, :] * -0.5 * hostInvVar[np.newaxis, :]**3

	hostInGrad = hostGrad * hostScale[np.newaxis, :] * hostInvVar[np.newaxis, :] + \
				 hostVarGrad * 2 / batchsize * (hostData - hostMean) + hostMeanGrad / batchsize

	assert np.allclose(hostBGrad, bgrad.get().squeeze())
	assert np.allclose(hostScaleGrad, scalegrad.get().squeeze())
	assert np.allclose(hostInGrad, ingrad.get().squeeze())


if __name__ == "__main__":
	unittest()
