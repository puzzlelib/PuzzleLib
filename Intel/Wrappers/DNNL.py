import math, multiprocessing
from collections import namedtuple
from enum import Enum

import numpy as np

from PuzzleLib import Config

from PuzzleLib.CPU.CPUArray import CPUArray
from PuzzleLib.CPU.Wrappers import NumpyBlas
from PuzzleLib.CPU.Benchmarks.Utils import timeKernel

from PuzzleLib.Intel.ThirdParty import libdnnl


class EngineKind(Enum):
	any = libdnnl.dnnl_engine_kind_t["dnnl_any_engine"]
	cpu = libdnnl.dnnl_engine_kind_t["dnnl_cpu"]


class StreamFlags(Enum):
	default = libdnnl.dnnl_stream_flags_t["dnnl_stream_default_order"]


engine = None
stream = None


def autoinit():
	global engine
	engine = libdnnl.dnnl_engine_create(EngineKind.cpu.value, Config.deviceIdx)

	if Config.systemLog:
		version = libdnnl.dnnl_version()

		print("[%s]: Created dnnl engine (Using version: %s.%s.%s)" % (
			Config.libname, version.major, version.minor, version.patch
		))

	global stream
	stream = libdnnl.dnnl_stream_create(engine, StreamFlags.default.value)

	def finishUp():
		libdnnl.dnnl_stream_destroy(stream)
		libdnnl.dnnl_engine_destroy(engine)

	import atexit
	atexit.register(finishUp)


if engine is None and (multiprocessing.current_process().name == "MainProcess" or Config.allowMultiContext):
	autoinit()


class TensorFormat(Enum):
	any = libdnnl.dnnl_format_tag_t["dnnl_format_tag_any"]
	a = libdnnl.dnnl_format_tag_t["dnnl_a"]
	abcd = libdnnl.dnnl_format_tag_t["dnnl_abcd"]
	abcde = libdnnl.dnnl_format_tag_t["dnnl_abcde"]


class DataType(Enum):
	s8 = libdnnl.dnnl_data_type_t["dnnl_s8"]
	u8 = libdnnl.dnnl_data_type_t["dnnl_u8"]
	s32 = libdnnl.dnnl_data_type_t["dnnl_s32"]
	f32 = libdnnl.dnnl_data_type_t["dnnl_f32"]


class ConvPerf:
	def __init__(self, algo, time, memory=0):
		self.algo = algo
		self.time = time
		self.memory = memory


class ConvAlgo(Enum):
	direct = libdnnl.dnnl_alg_kind_t["dnnl_convolution_direct"]
	winograd = libdnnl.dnnl_alg_kind_t["dnnl_convolution_winograd"]
	auto = libdnnl.dnnl_alg_kind_t["dnnl_convolution_auto"]


class DeconvAlgo(Enum):
	direct = libdnnl.dnnl_alg_kind_t["dnnl_deconvolution_direct"]
	winograd = libdnnl.dnnl_alg_kind_t["dnnl_deconvolution_winograd"]


class PoolMode(Enum):
	max = libdnnl.dnnl_alg_kind_t["dnnl_pooling_max"]
	avgWithPad = libdnnl.dnnl_alg_kind_t["dnnl_pooling_avg_include_padding"]
	avgNoPad = libdnnl.dnnl_alg_kind_t["dnnl_pooling_avg_exclude_padding"]


class LRNMode(Enum):
	map = libdnnl.dnnl_alg_kind_t["dnnl_lrn_within_channel"]
	cross = libdnnl.dnnl_alg_kind_t["dnnl_lrn_across_channels"]


class BatchNormFlags(Enum):
	useGlobalStats = libdnnl.dnnl_normalization_flags_t["dnnl_use_global_stats"]
	scaleShift = libdnnl.dnnl_normalization_flags_t["dnnl_use_scaleshift"]


class PropKind(Enum):
	fwdTrain = libdnnl.dnnl_prop_kind_t["dnnl_forward_training"]
	fwdInfer = libdnnl.dnnl_prop_kind_t["dnnl_forward_inference"]
	bwdData = libdnnl.dnnl_prop_kind_t["dnnl_backward_data"]
	bwdWeights = libdnnl.dnnl_prop_kind_t["dnnl_backward_weights"]
	bwdBias = libdnnl.dnnl_prop_kind_t["dnnl_backward_bias"]
	backward = libdnnl.dnnl_prop_kind_t["dnnl_backward"]


class Query(Enum):
	src = libdnnl.dnnl_query_t["dnnl_query_src_md"]
	diffSrc = libdnnl.dnnl_query_t["dnnl_query_diff_src_md"]
	weights = libdnnl.dnnl_query_t["dnnl_query_weights_md"]
	diffWeights = libdnnl.dnnl_query_t["dnnl_query_diff_weights_md"]
	dst = libdnnl.dnnl_query_t["dnnl_query_dst_md"]
	diffDst = libdnnl.dnnl_query_t["dnnl_query_diff_dst_md"]
	workspace = libdnnl.dnnl_query_t["dnnl_query_workspace_md"]


class ArgIndex(Enum):
	src = libdnnl.dnnl_ARG["dnnl_ARG_SRC_0"]
	weights = libdnnl.dnnl_ARG["dnnl_ARG_WEIGHTS_0"]
	bias = libdnnl.dnnl_ARG["dnnl_ARG_BIAS"]
	dst = libdnnl.dnnl_ARG["dnnl_ARG_DST_0"]

	diffSrc = libdnnl.dnnl_ARG["dnnl_ARG_DIFF_SRC_0"]
	diffWeights = libdnnl.dnnl_ARG["dnnl_ARG_DIFF_WEIGHTS_0"]
	diffBias = libdnnl.dnnl_ARG["dnnl_ARG_DIFF_BIAS"]
	diffDst = libdnnl.dnnl_ARG["dnnl_ARG_DIFF_DST_0"]

	workspace = libdnnl.dnnl_ARG["dnnl_ARG_WORKSPACE"]

	mean = libdnnl.dnnl_ARG["dnnl_ARG_MEAN"]
	variance = libdnnl.dnnl_ARG["dnnl_ARG_VARIANCE"]


DescTensor = namedtuple("DescTensor", "memory desc shape tensor")


dataTypeDct = {
	None: DataType.f32,
	np.float32: DataType.f32,
	np.int8: DataType.s8,
	np.uint8: DataType.u8,
	np.int32: DataType.s32
}


dataFormatDct = {
	1: TensorFormat.a,
	4: TensorFormat.abcd,
	5: TensorFormat.abcde
}


convPrimitiveCache, convBwdDataPrimitiveCache, convBwdParamPrimitiveCache = {}, {}, {}
poolPrimitiveCache, poolBwdPrimitiveCache = {}, {}
lrnPrimitiveCache, lrnBwdPrimitiveCache = {}, {}
bnPrimitiveCache, bnBwdPrimitiveCache = {}, {}


def createMemoryDescriptor(shape, tensor=None, dtype=None, fmt=None):
	shape = tensor.shape if shape is None else shape

	dataType = dataTypeDct[tensor.dtype.type if tensor is not None else dtype.type]
	dataFormat = dataFormatDct[len(shape)] if fmt is None else fmt

	memoryDesc = libdnnl.dnnl_memory_desc_init_by_tag(shape, dataType.value, dataFormat.value)
	return memoryDesc


def queryDescribedNdTensor(desc, query, tensor, index=0):
	desc = libdnnl.dnnl_primitive_desc_query_md(desc, query.value, index)

	memory = libdnnl.dnnl_memory_create(desc, engine)
	libdnnl.dnnl_memory_set_data_handle(memory, tensor.ptr)

	return DescTensor(memory=memory, desc=desc, shape=tensor.shape, tensor=tensor)


def createDescribedNdTensor(tensor, desc=None, fmt=None):
	if desc is None:
		desc = createMemoryDescriptor(None, tensor, fmt=fmt)

	memory = libdnnl.dnnl_memory_create(desc, engine)
	libdnnl.dnnl_memory_set_data_handle(memory, tensor.ptr)

	return DescTensor(memory=memory, desc=desc, shape=tensor.shape, tensor=tensor)


def destroyDescribedTensors(*descTensors):
	for descTensor in descTensors:
		libdnnl.dnnl_memory_destroy(descTensor.memory)


def executePrimitive(primitive, args):
	libdnnl.dnnl_primitive_execute(primitive, stream, args)
	libdnnl.dnnl_stream_wait(stream)


def prepareConvNdParams(ndim, stride=1, pad=0, dilation=1):
	stride = tuple(stride for _ in range(ndim - 2)) if isinstance(stride, int) else stride
	pad = tuple(pad for _ in range(ndim - 2)) if isinstance(pad, int) else pad

	if isinstance(dilation, int):
		dilation = tuple(dilation - 1 for _ in range(ndim - 2))
	else:
		dilation = tuple(dil - 1 for dil in dilation)

	return stride, pad, dilation


def getConvNdOutShape(datashape, Wshape, stride, pad, dilation):
	fsize = Wshape[2:]
	shape = tuple(
		(datashape[d + 2] + 2 * pad[d] - (dilation[d] + 1) * (fsize[d] - 1) - 1) // stride[d] + 1
		for d in range(len(stride))
	)

	return (datashape[0], Wshape[0]) + shape


def getConvNdInShape(gradshape, Wshape, stride, pad, dilation):
	fsize = Wshape[2:]
	shape = tuple(
		stride[d] * (gradshape[d + 2] - 1) - 2 * pad[d] + (dilation[d] + 1) * (fsize[d] - 1) + 1
		for d in range(len(stride))
	)

	return (gradshape[0], Wshape[1]) + shape


def dilationIsNotTrivial(dilation):
	return any(dil > 0 for dil in dilation)


def convNd(data, W, bias=None, stride=1, pad=0, dilation=1, algo=ConvAlgo.auto, transpose=False):
	assert data.ndim == W.ndim
	assert data.shape[1] == W.shape[1] if not transpose else data.shape[1] == W.shape[0]

	descData = createDescribedNdTensor(data)
	descW = createDescribedNdTensor(CPUArray.swapaxes(W, 0, 1) if transpose else W)

	descBias = createDescribedNdTensor(bias.reshape(bias.size)) if bias is not None else None
	biasDesc = None if descBias is None else descBias.desc

	stride, pad, dilation = prepareConvNdParams(data.ndim, stride, pad, dilation)
	dilated = dilationIsNotTrivial(dilation)

	if transpose:
		getOutShape = getConvNdInShape
		descInit = libdnnl.dnnl_dilated_deconvolution_forward_desc_init if dilated else \
			libdnnl.dnnl_deconvolution_forward_desc_init
		algo = DeconvAlgo.winograd if algo == ConvAlgo.winograd else DeconvAlgo.direct
	else:
		getOutShape = getConvNdOutShape
		descInit = libdnnl.dnnl_dilated_convolution_forward_desc_init if dilated else \
			libdnnl.dnnl_convolution_forward_desc_init

	outshape = getOutShape(data.shape, W.shape, stride, pad, dilation)
	dilation = (dilation, ) if dilated else ()

	key = (
		data.shape, data.dtype, W.shape, W.dtype, bias.shape if bias is not None else None,
		stride, pad, *dilation, algo, transpose
	)
	cache = convPrimitiveCache.get(key, None)

	if cache is None:
		outDesc = createMemoryDescriptor(outshape, dtype=data.dtype)
		convDesc = descInit(
			PropKind.fwdTrain.value, algo.value, descData.desc, descW.desc, biasDesc, outDesc, stride, *dilation, pad
		)

		convDesc = libdnnl.dnnl_primitive_desc_create(convDesc, None, engine, None)
		convPrimitive = libdnnl.dnnl_primitive_create(convDesc)

		convPrimitiveCache[key] = (convDesc, convPrimitive)

	else:
		convDesc, convPrimitive = cache

	outdata = CPUArray.empty(outshape, dtype=data.dtype)
	descOutData = queryDescribedNdTensor(convDesc, Query.dst, tensor=outdata)

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.src.value, descData.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.weights.value, descW.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.dst.value, descOutData.memory)
	]

	if descBias is not None:
		args.append(libdnnl.dnnl_exec_arg_t(ArgIndex.bias.value, descBias.memory))

	executePrimitive(convPrimitive, args)
	destroyDescribedTensors(descData, descW, descOutData)

	if descBias is not None:
		destroyDescribedTensors(descBias)

	return descOutData.tensor


def convNdBackwardData(grad, W, data=None, stride=1, pad=0, dilation=1, algo=ConvAlgo.auto, transpose=False):
	assert grad.ndim == W.ndim
	assert grad.shape[1] == W.shape[0] if not transpose else grad.shape[1] == W.shape[1]

	descGrad = createDescribedNdTensor(grad)
	descW = createDescribedNdTensor(CPUArray.swapaxes(W, 0, 1) if transpose else W)

	stride, pad, dilation = prepareConvNdParams(grad.ndim, stride, pad, dilation)
	dilated = dilationIsNotTrivial(dilation)

	if transpose:
		getInShape = getConvNdOutShape
		descInit = libdnnl.dnnl_dilated_deconvolution_backward_data_desc_init if dilated else \
			libdnnl.dnnl_deconvolution_backward_data_desc_init
		algo = DeconvAlgo.winograd if algo == ConvAlgo.winograd else DeconvAlgo.direct
	else:
		getInShape = getConvNdInShape
		descInit = libdnnl.dnnl_dilated_convolution_backward_data_desc_init if dilated else \
			libdnnl.dnnl_convolution_backward_data_desc_init

	inshape = getInShape(grad.shape, W.shape, stride, pad, dilation) if data is None else data.shape
	dilation = (dilation, ) if dilated else ()

	key = (grad.shape, grad.dtype, W.shape, W.dtype, stride, pad, *dilation, algo, transpose)
	cache = convBwdDataPrimitiveCache.get(key, None)

	if cache is None:
		inDesc = createMemoryDescriptor(inshape, dtype=grad.dtype)
		convDesc = descInit(algo.value, inDesc, descW.desc, descGrad.desc, stride, *dilation, pad)

		convDesc = libdnnl.dnnl_primitive_desc_create(convDesc, None, engine, None)
		convPrimitive = libdnnl.dnnl_primitive_create(convDesc)

		convBwdDataPrimitiveCache[key] = (convDesc, convPrimitive)

	else:
		convDesc, convPrimitive = cache

	ingrad = CPUArray.empty(inshape, dtype=grad.dtype)
	descInGrad = queryDescribedNdTensor(convDesc, Query.diffDst, tensor=ingrad)

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffDst.value, descGrad.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.weights.value, descW.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffSrc.value, descInGrad.memory)
	]

	executePrimitive(convPrimitive, args)
	destroyDescribedTensors(descGrad, descW, descInGrad)

	return descInGrad.tensor


def convNdBackwardParams(data, grad, W, bias=None, stride=1, pad=0, dilation=1, wgrad=None, bgrad=None,
						 scale=1.0, momentum=0.0, algo=ConvAlgo.auto, transpose=False):
	assert data.ndim == grad.ndim
	if not transpose:
		assert grad.shape[1] == W.shape[0] and data.shape[1] == W.shape[1]
	else:
		assert grad.shape[1] == W.shape[1] and data.shape[1] == W.shape[0]

	descData = createDescribedNdTensor(data)
	descGrad = createDescribedNdTensor(grad)

	stride, pad, dilation = prepareConvNdParams(grad.ndim, stride, pad, dilation)
	dilated = dilationIsNotTrivial(dilation)

	if transpose:
		descInit = libdnnl.dnnl_dilated_deconvolution_backward_weights_desc_init if dilated else \
			libdnnl.dnnl_deconvolution_backward_weights_desc_init
		algo = DeconvAlgo.winograd if algo == ConvAlgo.winograd else DeconvAlgo.direct
	else:
		descInit = libdnnl.dnnl_dilated_convolution_backward_weights_desc_init if dilated else \
			libdnnl.dnnl_convolution_backward_weights_desc_init

	if wgrad is not None and scale == 1.0 and momentum == 0.0:
		descWGrad = createDescribedNdTensor(CPUArray.swapaxes(wgrad, 0, 1) if transpose else wgrad)
	else:
		Wshape = (W.shape[1], W.shape[0]) + W.shape[2:] if transpose else W.shape
		descWGrad = createDescribedNdTensor(CPUArray.empty(Wshape, dtype=W.dtype))

	descBGrad, bgradDesc = None, None
	if bias is not None:
		if bgrad is not None and scale == 1.0 and momentum == 0.0:
			descBGrad = createDescribedNdTensor(bgrad.reshape(bgrad.size))
		else:
			descBGrad = createDescribedNdTensor(CPUArray.empty((bias.size, ), dtype=bias.dtype))

		bgradDesc = descBGrad.desc

	dilation = (dilation, ) if dilated else ()

	key = (
		data.shape, data.dtype, grad.shape, grad.dtype, W.shape, W.dtype, bias.shape if bias is not None else None,
		stride, pad, *dilation, algo, transpose
	)
	cache = convBwdParamPrimitiveCache.get(key, None)

	if cache is None:
		convDesc = descInit(algo.value, descData.desc, descWGrad.desc, bgradDesc, descGrad.desc, stride, *dilation, pad)

		convDesc = libdnnl.dnnl_primitive_desc_create(convDesc, None, engine, None)
		convPrimitive = libdnnl.dnnl_primitive_create(convDesc)

		convBwdParamPrimitiveCache[key] = (convDesc, convPrimitive)

	else:
		convDesc, convPrimitive = cache

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.src.value, descData.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffDst.value, descGrad.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffWeights.value, descWGrad.memory)
	]

	if bias is not None:
		args.append(libdnnl.dnnl_exec_arg_t(ArgIndex.diffBias.value, descBGrad.memory))

	executePrimitive(convPrimitive, args)
	currWgrad = CPUArray.swapaxes(descWGrad.tensor, 0, 1) if transpose else descWGrad.tensor

	if scale != 1.0 or momentum != 0.0:
		if wgrad is not None:
			NumpyBlas.addVectorToVector(currWgrad.ravel(), wgrad.ravel(), out=wgrad.ravel(), alpha=scale, beta=momentum)

		if bgrad is not None:
			NumpyBlas.addVectorToVector(
				descBGrad.tensor.ravel(), bgrad.ravel(), out=bgrad.ravel(), alpha=scale, beta=momentum
			)

	destroyDescribedTensors(descData, descGrad, descWGrad)
	if bias is not None:
		destroyDescribedTensors(descBGrad)

	return (currWgrad, descBGrad.tensor.reshape(bias.shape)) if bias is not None else currWgrad


def convNdbenchmark(datashape, Wshape, stride=1, pad=0, dilation=1, transpose=False):
	startStride, startPad, startDilation = stride, pad, dilation
	stride, pad, dilation = prepareConvNdParams(len(Wshape), stride, pad, dilation)

	if transpose:
		outshape = getConvNdInShape(datashape, Wshape, stride, pad, dilation)
	else:
		outshape = getConvNdOutShape(datashape, Wshape, stride, pad, dilation)

	data, grad = CPUArray.empty(datashape, dtype=np.float32), CPUArray.empty(outshape, dtype=np.float32)
	W, bias = CPUArray.empty(Wshape, dtype=np.float32), CPUArray.empty((outshape[1] ), dtype=np.float32)

	fwdResults, bwdParamResults, bwdDataResults = [], [], []
	looplength = 1

	for algo in ConvAlgo:
		kwargs = {"algo": algo, "transpose": transpose}

		try:
			secs = timeKernel(
				convNd, args=(data, W, bias, startStride, startPad, startDilation), kwargs=kwargs,
				looplength=looplength, log=False, normalize=True
			)

		except libdnnl.dnnlUnimplemented:
			secs = -1.0

		fwdResults.append(ConvPerf(algo, secs))

		try:
			secs = timeKernel(
				convNdBackwardParams, args=(data, grad, W, bias, startStride, startPad, startDilation), kwargs=kwargs,
				looplength=looplength, log=False, normalize=True
			)

		except libdnnl.dnnlUnimplemented:
			secs = -1.0

		bwdParamResults.append(ConvPerf(algo, secs))

		try:
			secs = timeKernel(
				convNdBackwardData, args=(grad, W, data, startStride, startPad, startDilation), kwargs=kwargs,
				looplength=looplength, log=False, normalize=True
			)

		except libdnnl.dnnlUnimplemented:
			secs = -1.0

		bwdDataResults.append(ConvPerf(algo, secs))

	key = lambda res: res.time if res.time >= 0.0 else math.inf
	return sorted(fwdResults, key=key), sorted(bwdParamResults, key=key), sorted(bwdDataResults, key=key)


def preparePoolNdParams(ndim, size=2, stride=2, pad=0):
	stride = tuple(stride for _ in range(ndim - 2)) if isinstance(stride, int) else stride
	size = tuple(size for _ in range(ndim - 2)) if isinstance(size, int) else size
	pad = tuple(pad for _ in range(ndim - 2)) if isinstance(pad, int) else pad

	return size, stride, pad


def getPoolNdOutShape(datashape, sizes, stride, pad):
	shape = tuple((datashape[d + 2] - sizes[d] + 2 * pad[d]) // stride[d] + 1 for d in range(len(datashape) - 2))
	return datashape[:2] + shape


def poolNd(data, size=2, stride=2, pad=0, mode=PoolMode.max, test=False):
	descData = createDescribedNdTensor(data)

	size, stride, pad = preparePoolNdParams(data.ndim, size, stride, pad)
	outshape = getPoolNdOutShape(data.shape, size, stride, pad)

	descOutData = createDescribedNdTensor(CPUArray.empty(outshape, dtype=data.dtype))

	key = (data.shape, data.dtype, size, stride, pad, mode, test)
	cache = poolPrimitiveCache.get(key, None)

	if cache is None:
		prop = PropKind.fwdInfer if test else PropKind.fwdTrain
		poolDesc = libdnnl.dnnl_pooling_forward_desc_init(
			prop.value, mode.value, descData.desc, descOutData.desc, stride, size, pad
		)

		poolDesc = libdnnl.dnnl_primitive_desc_create(poolDesc, None, engine, None)
		poolPrimitive = libdnnl.dnnl_primitive_create(poolDesc)

		poolPrimitiveCache[key] = (poolDesc, poolPrimitive)

	else:
		poolDesc, poolPrimitive = cache

	workspaceDesc, descWorkspace = None, None
	if not test:
		workspaceDesc = libdnnl.dnnl_primitive_desc_query_md(poolDesc, Query.workspace.value, 0)

		if workspaceDesc is not None:
			descWorkspace = createDescribedNdTensor(CPUArray.empty(
				libdnnl.dnnl_memory_desc_get_size(workspaceDesc), dtype=np.int8), desc=workspaceDesc
			)

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.src.value, descData.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.dst.value, descOutData.memory)
	]

	if descWorkspace is not None:
		args.append(libdnnl.dnnl_exec_arg_t(ArgIndex.workspace.value, descWorkspace.memory))

	executePrimitive(poolPrimitive, args)
	destroyDescribedTensors(descData, descOutData)

	workspace = None
	if workspaceDesc is not None:
		destroyDescribedTensors(descWorkspace)
		workspace = descWorkspace.tensor

	return descOutData.tensor if test else (descOutData.tensor, workspace, poolDesc)


def poolNdBackward(indata, grad, workspace, desc, size=2, stride=2, pad=0, mode=PoolMode.max):
	size, stride, pad = preparePoolNdParams(grad.ndim, size, stride, pad)

	descGrad = createDescribedNdTensor(grad)
	descInGrad = createDescribedNdTensor(CPUArray.empty(indata.shape, dtype=indata.dtype))

	key = (indata.shape, indata.dtype, grad.shape, grad.dtype, size, stride, pad, mode)
	cache = poolBwdPrimitiveCache.get(key, None)

	if cache is None:
		poolDesc = libdnnl.dnnl_pooling_backward_desc_init(
			mode.value, descInGrad.desc, descGrad.desc, stride, size, pad
		)

		poolDesc = libdnnl.dnnl_primitive_desc_create(poolDesc, None, engine, desc)
		poolPrimitive = libdnnl.dnnl_primitive_create(poolDesc)

		poolBwdPrimitiveCache[key] = (poolDesc, poolPrimitive)

	else:
		poolDesc, poolPrimitive = cache

	descWorkspace = None
	if workspace is not None:
		workspaceDesc = libdnnl.dnnl_primitive_desc_query_md(poolDesc, Query.workspace.value, 0)
		descWorkspace = createDescribedNdTensor(workspace, desc=workspaceDesc)

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffDst.value, descGrad.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffSrc.value, descInGrad.memory)
	]

	if descWorkspace is not None:
		args.append(libdnnl.dnnl_exec_arg_t(ArgIndex.workspace.value, descWorkspace.memory))

	executePrimitive(poolPrimitive, args)
	destroyDescribedTensors(descGrad, descInGrad)

	if workspace is not None:
		destroyDescribedTensors(descWorkspace)

	return descInGrad.tensor


def softmaxNd(data):
	descData = createDescribedNdTensor(data)
	descOutData = createDescribedNdTensor(CPUArray.empty(data.shape, dtype=data.dtype))

	softmaxDesc = libdnnl.dnnl_softmax_forward_desc_init(PropKind.fwdInfer.value, descData.desc, 1)
	softmaxDesc = libdnnl.dnnl_primitive_desc_create(softmaxDesc, None, engine, None)

	descSoftmax = libdnnl.dnnl_primitive_create(softmaxDesc)
	libdnnl.dnnl_primitive_desc_destroy(softmaxDesc)

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.src.value, descData.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.dst.value, descOutData.memory)
	]

	executePrimitive(descSoftmax, args)
	libdnnl.dnnl_primitive_destroy(descSoftmax)

	destroyDescribedTensors(descData, descOutData)
	return descOutData.tensor


def softmaxNdBackward(outdata, grad):
	descOutData = createDescribedNdTensor(outdata)

	descGrad = createDescribedNdTensor(grad)
	descInGrad = createDescribedNdTensor(CPUArray.empty(grad.shape, dtype=grad.dtype))

	softmaxDesc = libdnnl.dnnl_softmax_backward_desc_init(descGrad.desc, descOutData.desc, 1)
	softmaxDesc = libdnnl.dnnl_primitive_desc_create(softmaxDesc, None, engine, None)

	descSoftmax = libdnnl.dnnl_primitive_create(softmaxDesc)
	libdnnl.dnnl_primitive_desc_destroy(softmaxDesc)

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.dst.value, descOutData.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffDst.value, descGrad.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffSrc.value, descInGrad.memory)
	]

	executePrimitive(descSoftmax, args)
	libdnnl.dnnl_primitive_destroy(descSoftmax)

	destroyDescribedTensors(descOutData, descGrad, descInGrad)
	return descInGrad.tensor


def lrn(data, mode=LRNMode.map, N=5, alpha=1e-4, beta=0.75, K=2.0, test=False):
	descData = createDescribedNdTensor(data)
	descOutData = createDescribedNdTensor(CPUArray.empty(data.shape, dtype=data.dtype))

	key = (data.shape, data.dtype, mode, N, alpha, beta, K, test)
	cache = lrnPrimitiveCache.get(key, None)

	if cache is None:
		prop = PropKind.fwdInfer if test else PropKind.fwdTrain
		lrnDesc = libdnnl.dnnl_lrn_forward_desc_init(prop.value, mode.value, descData.desc, N, alpha, beta, K)

		lrnDesc = libdnnl.dnnl_primitive_desc_create(lrnDesc, None, engine, None)
		lrnPrimitive = libdnnl.dnnl_primitive_create(lrnDesc)

		lrnPrimitiveCache[key] = (lrnDesc, lrnPrimitive)

	else:
		lrnDesc, lrnPrimitive = cache

	descWorkspace = None

	if not test:
		workspaceDesc = libdnnl.dnnl_primitive_desc_query_md(lrnDesc, Query.workspace.value, 0)
		descWorkspace = createDescribedNdTensor(CPUArray.empty(
			libdnnl.dnnl_memory_desc_get_size(workspaceDesc), dtype=np.int8), desc=workspaceDesc
		)

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.src.value, descData.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.dst.value, descOutData.memory)
	]

	if descWorkspace is not None:
		args.append(libdnnl.dnnl_exec_arg_t(ArgIndex.workspace.value, descWorkspace.memory))

	executePrimitive(lrnPrimitive, args)
	destroyDescribedTensors(descData, descOutData)

	return descOutData.tensor if test else (descOutData.tensor, descWorkspace, lrnDesc)


def lrnBackward(data, grad, descWorkspace, desc, mode=LRNMode.map, N=5, alpha=1e-4, beta=0.75, K=2.0):
	assert mode != LRNMode.map

	descGrad = createDescribedNdTensor(grad)
	descInGrad = createDescribedNdTensor(CPUArray.empty(grad.shape, dtype=grad.dtype))
	descData = createDescribedNdTensor(data)

	key = (data.shape, data.dtype, grad.shape, grad.dtype, mode, N, alpha, beta, K)
	cache = lrnBwdPrimitiveCache.get(key, None)

	if cache is None:
		lrnDesc = libdnnl.dnnl_lrn_backward_desc_init(mode.value, descData.desc, descGrad.desc, N, alpha, beta, K)

		lrnDesc = libdnnl.dnnl_primitive_desc_create(lrnDesc, None, engine, desc)
		lrnPrimitive = libdnnl.dnnl_primitive_create(lrnDesc)

		lrnBwdPrimitiveCache[key] = (lrnDesc, lrnPrimitive)

	else:
		lrnDesc, lrnPrimitive = cache

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.src.value, descData.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffDst.value, descGrad.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.workspace.value, descWorkspace.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffSrc.value, descInGrad.memory)
	]

	executePrimitive(lrnPrimitive, args)
	destroyDescribedTensors(descGrad, descInGrad, descData, descWorkspace)

	return descInGrad.tensor


def batchNormNd(data, scale, bias, mean, var, epsilon=1e-5, test=False, out=None):
	assert data.ndim == scale.ndim and scale.ndim == bias.ndim and bias.ndim == mean.ndim and mean.ndim == var.ndim

	descData = createDescribedNdTensor(data)
	descOutData = createDescribedNdTensor(CPUArray.empty(data.shape, dtype=data.dtype) if out is None else out)

	descWeights = createDescribedNdTensor(CPUArray.toDevice(np.concatenate(
		(scale.data.reshape(scale.size), bias.data.reshape(bias.size))
	)))

	descMean = createDescribedNdTensor(mean.reshape(mean.size))
	descVar = createDescribedNdTensor(var.reshape(var.size))

	key = (data.shape, data.dtype, epsilon, test)
	cache = bnPrimitiveCache.get(key, None)

	if cache is None:
		flags = BatchNormFlags.useGlobalStats.value if test else 0
		prop = PropKind.fwdInfer if test else PropKind.fwdTrain

		bnDesc = libdnnl.dnnl_batch_normalization_forward_desc_init(
			prop.value, descData.desc, epsilon, BatchNormFlags.scaleShift.value | flags
		)

		bnDesc = libdnnl.dnnl_primitive_desc_create(bnDesc, None, engine, None)
		bnPrimitive = libdnnl.dnnl_primitive_create(bnDesc)

		bnPrimitiveCache[key] = (bnDesc, bnPrimitive)

	else:
		bnDesc, bnPrimitive = cache

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.src.value, descData.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.weights.value, descWeights.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.dst.value, descOutData.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.mean.value, descMean.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.variance.value, descVar.memory)
	]

	executePrimitive(bnPrimitive, args)
	destroyDescribedTensors(descData, descOutData, descWeights, descMean, descVar)

	bnDesc = None if test else bnDesc
	return descOutData.tensor, descMean.tensor.reshape(scale.shape), descVar.tensor.reshape(scale.shape), bnDesc


def batchNormNdBackward(data, grad, scale, bias, savemean, savevar, desc, epsilon=1e-5):
	assert scale.ndim == savemean.ndim and savemean.ndim == savevar.ndim

	descData = createDescribedNdTensor(data)
	descGrad = createDescribedNdTensor(grad)
	descScale = createDescribedNdTensor(scale.reshape(scale.size))

	descWeights = createDescribedNdTensor(CPUArray.toDevice(np.concatenate(
		(scale.data.reshape(scale.size), bias.data.reshape(bias.size))
	)))

	descMean = createDescribedNdTensor(savemean.reshape(savemean.size))
	descVar = createDescribedNdTensor(savevar.reshape(savevar.size))

	descInGrad = createDescribedNdTensor(CPUArray.empty(grad.shape, dtype=grad.dtype))
	descWGrad = createDescribedNdTensor(CPUArray.empty((2 * scale.size, ), dtype=scale.dtype))

	key = (data.shape, data.dtype, grad.shape, grad.dtype, epsilon)
	cache = bnBwdPrimitiveCache.get(key, None)

	if cache is None:
		bnDesc = libdnnl.dnnl_batch_normalization_backward_desc_init(
			PropKind.backward.value, descGrad.desc, descData.desc, epsilon, BatchNormFlags.scaleShift.value
		)

		bnDesc = libdnnl.dnnl_primitive_desc_create(bnDesc, None, engine, desc)
		bnPrimitive = libdnnl.dnnl_primitive_create(bnDesc)

		bnBwdPrimitiveCache[key] = (bnDesc, bnPrimitive)

	else:
		bnDesc, bnPrimitive = cache

	args = [
		libdnnl.dnnl_exec_arg_t(ArgIndex.src.value, descData.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.mean.value, descMean.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.variance.value, descVar.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffDst.value, descGrad.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.weights.value, descWeights.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffSrc.value, descInGrad.memory),
		libdnnl.dnnl_exec_arg_t(ArgIndex.diffWeights.value, descWGrad.memory)
	]

	executePrimitive(bnPrimitive, args)
	destroyDescribedTensors(descData, descGrad, descScale, descWeights, descMean, descVar, descInGrad, descWGrad)

	scalegrad = descWGrad.tensor[:scale.size].reshape(*scale.shape)
	biasgrad = descWGrad.tensor[scale.size:].reshape(*scale.shape)

	return descInGrad.tensor, scalegrad, biasgrad


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

	data = CPUArray.toDevice(np.random.randn(batchsize, inmaps, h, w).astype(np.float32))

	W = CPUArray.toDevice(np.random.randn(outmaps, inmaps, fsize, fsize).astype(np.float32))
	bias = CPUArray.toDevice(np.random.randn(1, outmaps, 1, 1).astype(np.float32))

	outdata = convNd(data, W, bias)

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

	grad = CPUArray.toDevice(np.random.randn(*outdata.shape).astype(np.float32))
	ingrad = convNdBackwardData(grad, W)

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

	wgrad, bgrad = convNdBackwardParams(data, grad, W, bias)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for dy in range(fsize):
					for dx in range(fsize):
						for y in range(hostGrad.shape[2]):
							for x in range(hostGrad.shape[3]):
								hostWGrad[oc, ic, dy, dx] += hostData[b,ic,y + dy, x + dx] * hostGrad[b, oc, y, x]

	assert np.allclose(hostWGrad, wgrad.get())

	hostBGrad = np.empty(hostBias.shape, dtype=np.float32)
	for oc in range(outmaps):
		hostBGrad[0, oc, 0, 0] = np.sum(hostGrad[:, oc, :, :])

	assert np.allclose(hostBGrad, bgrad.get())


def deconv2dTest():
	batchsize, inmaps, h, w = 1, 1, 2, 2
	fsize, stride, outmaps = 3, 2, 1

	data = CPUArray.toDevice(np.random.randn(batchsize, inmaps, h, w).astype(np.float32))

	W = CPUArray.toDevice(np.random.randn(inmaps, outmaps, fsize, fsize).astype(np.float32))
	bias = CPUArray.toDevice(np.random.randn(1, outmaps, 1, 1).astype(np.float32))

	outdata = convNd(data, W, bias, stride=stride, transpose=True)

	hostOutData = np.zeros(outdata.shape).astype(np.float32)
	for i in range(0, hostOutData.shape[2] - fsize + 1, stride):
		for j in range(0, hostOutData.shape[3] - fsize + 1, stride):
			hostOutData[0, 0, i:fsize+i, j:fsize+j] += W.get()[0, 0] * data.get()[0, 0, i // stride, j // stride]

	hostOutData += bias.get()
	assert np.allclose(hostOutData, outdata.get())

	grad = CPUArray.toDevice(np.random.randn(*outdata.shape).astype(np.float32))

	ingrad = convNdBackwardData(grad, W, stride=stride, transpose=True)

	hostInGrad = np.zeros(data.shape, dtype=np.float32)
	for i in range(0, hostInGrad.shape[2]):
		for j in range(0, hostInGrad.shape[3]):
			y, x = i * stride, j * stride
			hostInGrad[0, 0, i, j] += np.dot(W.get()[0, 0].ravel(), grad.get()[0, 0, y:y+fsize, x:x+fsize].ravel())

	assert np.allclose(hostInGrad, ingrad.get())

	wgrad, bgrad = convNdBackwardParams(data, grad, W, bias, stride=stride, transpose=True)

	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)
	for i in range(0, hostOutData.shape[2] - fsize + 1, stride):
		for j in range(0, hostOutData.shape[3] - fsize + 1, stride):
			hostWGrad[0, 0] += grad.get()[0, 0, i:i+fsize, j:j+fsize] * data.get()[0, 0, i // stride, j // stride]

	assert np.allclose(hostWGrad, wgrad.get())

	hostBGrad = np.sum(grad.get())
	assert np.allclose(hostBGrad, bgrad.get())


def maxpool2dTest():
	batchsize, maps, h, w = 1, 1, 8, 8
	data = CPUArray.toDevice(np.random.randn(batchsize, maps, h, w).astype(np.float32))

	outdata, workspace, desc = poolNd(data)

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

	grad = CPUArray.toDevice(np.random.randn(*outdata.shape).astype(np.float32))
	poolNdBackward(data, grad, workspace, desc)


def softmaxTest():
	batchsize, maps = 5, 8
	data = CPUArray.toDevice(np.random.randn(batchsize, maps, 1, 1).astype(np.float32))

	outdata = softmaxNd(data)

	def hostSoftmax(w):
		e = np.exp(w - np.amax(w))
		p = e / np.sum(e)
		return p

	hostData = data.get().reshape(batchsize, maps)
	hostOutData = np.vstack([hostSoftmax(hostData[i]) for i in range(batchsize)])
	assert np.allclose(hostOutData, outdata.get().reshape(batchsize, maps))

	grad = CPUArray.toDevice(np.random.randn(batchsize, maps, 1, 1).astype(np.float32))
	ingrad = softmaxNdBackward(outdata, grad)

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

	lookBehind = (N - 1) // 2
	lookAhead = N - lookBehind

	data = CPUArray.toDevice(np.random.randn(1, 1, h, w).astype(np.float32))
	outdata, workspace, desc = lrn(data, mode=LRNMode.map, N=N, alpha=alpha, beta=beta, K=K)

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


def crossMapLRNTest():
	maps = 10
	N, alpha, beta, K = 5, 1.0, 0.5, 2.0

	lookBehind = (N - 1) // 2
	lookAhead = N - lookBehind

	data = CPUArray.toDevice(np.random.randn(1, maps, 1, 1).astype(np.float32))
	outdata, workspace, desc = lrn(data, mode=LRNMode.cross, N=N, alpha=alpha, beta=beta, K=K)

	hostData = data.get().reshape(maps, ).astype(np.float32)
	norms = np.empty((maps, ), dtype=np.float32)
	for i in range(maps):
		norm = 0.0
		for j in range(max(0, i - lookBehind), min(maps, i + lookAhead)):
			norm += hostData[j]**2
		norms[i] = K + norm * alpha / N

	hostOutData = hostData / norms**beta
	assert np.allclose(hostOutData, outdata.reshape(maps, ).get())

	grad = CPUArray.toDevice(np.random.randn(1, maps, 1, 1).astype(np.float32))
	ingrad = lrnBackward(data, grad, workspace, desc, mode=LRNMode.cross, N=N, alpha=alpha, beta=beta, K=K)

	hostGrad = grad.get().reshape(maps, ).astype(np.float32)
	hostInGrad = np.zeros((maps, ), dtype=np.float32)
	k = 2.0 * alpha * beta / N
	for i in range(maps):
		hostInGrad[i] += hostGrad[i] / norms[i]**beta

		for j in range(max(0, i - lookBehind), min(maps, i + lookAhead)):
			hostInGrad[j] -= hostGrad[i] * k * hostData[i] * hostData[j] / norms[i]**(beta+1)

	assert np.allclose(hostInGrad, ingrad.reshape(maps, ).get())


def batchNorm2dTest():
	batchsize, maps, h, w = 4, 5, 3, 2
	data = CPUArray.toDevice(np.random.randn(batchsize, maps, h, w).astype(np.float32))
	hostData = data.get()

	scale = CPUArray.toDevice(np.random.randn(1, maps, 1, 1).astype(np.float32))
	bias = CPUArray.toDevice(np.random.randn(1, maps, 1, 1).astype(np.float32))
	mean = CPUArray.toDevice(np.zeros((1, maps, 1, 1), dtype=np.float32))
	var = CPUArray.toDevice(np.ones((1, maps, 1, 1), dtype=np.float32))

	outdata, savemean, savevar, desc = batchNormNd(data, scale, bias, mean, var, out=data)

	hostScale, hostBias = scale.get(), bias.get()
	hostNormData = np.empty(hostData.shape, dtype=np.float32)
	hostOutData = np.empty(hostData.shape, dtype=np.float32)
	hostMean = np.zeros(scale.shape, dtype=np.float32)
	hostVar = np.zeros(scale.shape, dtype=np.float32)
	hostInvVar = np.empty(scale.shape, dtype=np.float32)
	for c in range(maps):
		for b in range(batchsize):
			hostMean[0, c, 0, 0] += np.sum(hostData[b, c])
		hostMean[0, c, 0, 0] /= (batchsize * w * h)

		for b in range(batchsize):
			hostVar[0, c, 0, 0] += np.sum((hostData[b, c] - hostMean[0, c, 0, 0])**2)
		hostVar[0, c, 0, 0] /= (batchsize * w * h)

		hostInvVar[0, c, 0, 0] = 1.0 / np.sqrt(hostVar[0, c, 0, 0] + 1e-5)
		hostNormData[:, c, :, :] = (hostData[:, c, :, :] - hostMean[0, c, 0, 0]) * hostInvVar[0, c, 0, 0]
		hostOutData[:, c, :, :] = hostNormData[:, c, :, :] * hostScale[0, c, 0, 0] + hostBias[0, c, 0, 0]

	assert np.allclose(hostMean, mean.get())
	assert np.allclose(hostVar, savevar.get())
	assert np.allclose(hostOutData, outdata.get())

	grad = CPUArray.toDevice(np.random.randn(batchsize, maps, h, w).astype(np.float32))

	data = CPUArray.toDevice(hostData)
	ingrad, scalegrad, biasgrad = batchNormNdBackward(data, grad, scale, bias, savemean, savevar, desc)

	hostGrad = grad.get()
	hostInGrad, hostScaleGrad = np.empty(grad.shape, dtype=np.float32), np.empty(scale.shape, dtype=np.float32)
	hostBiasGrad, hostMeanGrad = np.empty(bias.shape, dtype=np.float32), np.empty(hostMean.shape, dtype=np.float32)
	hostVarGrad = np.empty(hostVar.shape, dtype=np.float32)
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
	assert np.allclose(hostBiasGrad, biasgrad.get())

	batchNormNd(data, scale, bias, mean, var, test=True)


def batchNormTest():
	batchsize, size = 4, 5

	data = CPUArray.toDevice(np.random.randn(batchsize, size, 1, 1).astype(np.float32))
	hostData = data.get().squeeze()

	scale = CPUArray.toDevice(np.random.randn(1, size, 1, 1).astype(np.float32))
	bias = CPUArray.toDevice(np.random.randn(1, size, 1, 1).astype(np.float32))
	mean = CPUArray.zeros((1, size, 1, 1), dtype=np.float32)
	var = CPUArray.toDevice(np.ones((1, size, 1, 1), dtype=np.float32))

	outdata, savemean, savevar, desc = batchNormNd(data, scale, bias, mean, var, out=data)

	hostMean = np.mean(hostData, axis=0, keepdims=False)
	hostVar = np.sum((hostData - hostMean[np.newaxis, :])**2, axis=0) / batchsize
	hostInvVar = 1.0 / np.sqrt(hostVar + 1e-5)

	hostNormData = (hostData - hostMean) * hostInvVar
	hostScale = scale.get().squeeze()
	hostBias = bias.get().squeeze()
	hostOutData = hostNormData * hostScale + hostBias

	assert np.allclose(hostMean, savemean.get().squeeze())
	assert np.allclose(hostVar, savevar.get().squeeze())
	assert np.allclose(hostOutData, outdata.get().squeeze())

	grad = CPUArray.toDevice(np.random.randn(batchsize, size, 1, 1).astype(np.float32))

	data = CPUArray.toDevice(hostData).reshape(batchsize, size, 1, 1)
	ingrad, scalegrad, biasgrad = batchNormNdBackward(data, grad, scale, bias, savemean, savevar, desc)

	hostGrad = grad.get().squeeze()

	hostBiasGrad = np.sum(hostGrad, axis=0)
	hostScaleGrad = np.sum(hostGrad * hostNormData, axis=0)
	hostMeanGrad = np.sum(hostGrad, axis=0) * hostScale * -hostInvVar
	hostVarGrad = np.sum(hostGrad * (hostData - hostMean[np.newaxis, :]), axis=0) * \
				  hostScale[np.newaxis, :] * -0.5 * hostInvVar[np.newaxis, :]**3

	hostInGrad = hostGrad * hostScale[np.newaxis, :] * hostInvVar[np.newaxis, :] + \
				 hostVarGrad * 2 / batchsize * (hostData - hostMean) + hostMeanGrad / batchsize

	assert np.allclose(hostBiasGrad, biasgrad.get().squeeze())
	assert np.allclose(hostScaleGrad, scalegrad.get().squeeze())
	assert np.allclose(hostInGrad, ingrad.get().squeeze())


if __name__ == "__main__":
	unittest()
