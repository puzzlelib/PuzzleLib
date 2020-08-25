import itertools
from enum import Enum

import numpy as np

from PuzzleLib.Cuda.Wrappers.CuDnn import conv2dTest, conv3dTest, convGroupTest
from PuzzleLib.Cuda.Wrappers.CuDnn import deconv2dTest, deconv3dTest, deconvGroupTest, softmax2dTest

from PuzzleLib.Hip.Driver import GPUArray
from PuzzleLib.Hip.ThirdParty import libmiopen


class DataType(Enum):
	float = libmiopen.miopenDataType["miopenFloat"]
	half = libmiopen.miopenDataType["miopenHalf"]


class ConvMode(Enum):
	conv = libmiopen.miopenConvolutionMode["miopenConvolution"]
	transpose = libmiopen.miopenConvolutionMode["miopenTranspose"]


class ConvFwdAlgo(Enum):
	auto = -1
	gemm = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoGEMM"]
	direct = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoDirect"]
	fft = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoFFT"]
	winograd = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoWinograd"]
	implicitGemm = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoImplicitGEMM"]
	staticGemm = libmiopen.miopenConvFwdAlgorithm["miopenConvolutionFwdAlgoStaticCompiledGEMM"]


class ConvBwdFilterAlgo(Enum):
	auto = -1
	gemm = libmiopen.miopenConvBwdWeightsAlgorithm["miopenConvolutionBwdWeightsAlgoGEMM"]
	direct = libmiopen.miopenConvBwdWeightsAlgorithm["miopenConvolutionBwdWeightsAlgoDirect"]
	winograd = libmiopen.miopenConvBwdWeightsAlgorithm["miopenConvolutionBwdWeightsAlgoWinograd"]
	implicitGemm = libmiopen.miopenConvBwdWeightsAlgorithm["miopenConvolutionBwdWeightsAlgoImplicitGEMM"]


class ConvBwdDataAlgo(Enum):
	auto = -1
	gemm = libmiopen.miopenConvBwdDataAlgorithm["miopenConvolutionBwdDataAlgoGEMM"]
	direct = libmiopen.miopenConvBwdDataAlgorithm["miopenConvolutionBwdDataAlgoDirect"]
	fft = libmiopen.miopenConvBwdDataAlgorithm["miopenConvolutionBwdDataAlgoFFT"]
	winograd = libmiopen.miopenConvBwdDataAlgorithm["miopenConvolutionBwdDataAlgoWinograd"]
	transposeGemm = libmiopen.miopenConvBwdDataAlgorithm["miopenTransposeBwdDataAlgoGEMM"]
	implicitGemm = libmiopen.miopenConvBwdDataAlgorithm["miopenConvolutionBwdDataAlgoImplicitGEMM"]


class PoolMode(Enum):
	max = libmiopen.miopenPoolingMode["miopenPoolingMax"]
	avgWithPad = libmiopen.miopenPoolingMode["miopenPoolingAverageInclusive"]
	avgNoPad = libmiopen.miopenPoolingMode["miopenPoolingAverage"]


class SoftMaxAlgo(Enum):
	fast = libmiopen.miopenSoftmaxAlgorithm["MIOPEN_SOFTMAX_FAST"]
	accurate = libmiopen.miopenSoftmaxAlgorithm["MIOPEN_SOFTMAX_ACCURATE"]
	log = libmiopen.miopenSoftmaxAlgorithm["MIOPEN_SOFTMAX_LOG"]


class SoftMaxMode(Enum):
	perActivation = libmiopen.miopenSoftmaxMode["MIOPEN_SOFTMAX_MODE_INSTANCE"]
	spatial = libmiopen.miopenSoftmaxMode["MIOPEN_SOFTMAX_MODE_CHANNEL"]


class BatchNormMode(Enum):
	perActivation = libmiopen.miopenBatchNormMode["miopenBNPerActivation"]
	spatial = libmiopen.miopenBatchNormMode["miopenBNSpatial"]


class LRNMode(Enum):
	map = libmiopen.miopenLRNMode["miopenLRNWithinChannel"]
	cross = libmiopen.miopenLRNMode["miopenLRNCrossChannel"]


toDataType = {
	np.float32: DataType.float,
	np.float16: DataType.half
}


class ConvPerf:
	def __init__(self, algo, time, memory):
		self.algo = algo
		self.time = time
		self.memory = memory


	def toString(self):
		return "%-40s %-25s %-28s" % (
			"Algo %s" % self.algo, "time %.6f secs" % self.time, "memory %.6f mbytes" % (self.memory / 1024**2)
		)


	def __str__(self):
		return self.toString()


	def __repr__(self):
		return self.toString()


class DescTensor:
	__slots__ = ["desc", "shape", "tensor", "ptr"]

	def __init__(self, desc, shape, tensor):
		self.desc, self.shape, self.tensor = desc, shape, tensor
		self.ptr = None if tensor is None else tensor.ptr


class DescConvNd:
	__slots__ = ["desc", "mode", "stride", "pad", "dilation", "groups"]

	def __init__(self, desc, mode, stride, pad, dilation, groups):
		self.desc, self.mode = desc, mode
		self.stride, self.pad, self.dilation, self.groups = stride, pad, dilation, groups


class DescPool2d:
	__slots__ = ["desc", "size", "stride", "pad"]

	def __init__(self, desc, size, stride, pad):
		self.desc, self.size, self.stride, self.pad = desc, size, stride, pad


class DescLRN:
	__slots__ = ["desc", "N", "alpha", "beta", "K"]

	def __init__(self, desc, N, alpha, beta, K):
		self.desc, self.N, self.alpha, self.beta, self.K = desc, N, alpha, beta, K


class DnnContext:
	def __init__(self, backend):
		self.backend = backend
		self.context = libmiopen.miopenCreate()

		self.tensorDescCache = []

		self.convCache = {}
		self.convBackwardDataCache = {}
		self.convBackwardParamsCache = {}


	def __del__(self):
		try:
			libmiopen.miopenDestroy(self.context)

		except AttributeError:
			pass


	@staticmethod
	def getVersion():
		return libmiopen.version


	def enableTensorOps(self, _):
		return self


	def createDescribedNdTensor(self, tensor, dims=None, strides=None, dtype=DataType.float):
		dataType = dtype if tensor is None else toDataType[tensor.dtype.type]

		dims = tensor.shape if dims is None else dims
		strides = tuple(stride // tensor.dtype.itemsize for stride in tensor.strides) if strides is None else strides

		desc = self.tensorDescCache.pop() if len(self.tensorDescCache) > 0 else libmiopen.miopenCreateTensorDescriptor()
		libmiopen.miopenSetTensorDescriptor(desc, dataType.value, dims, strides)

		return DescTensor(desc, dims, tensor)


	def createDescribed1dTensor(self, tensor, ndim):
		dataType = toDataType[tensor.dtype.type]
		assert tensor.ndim == 1

		dims = (1, tensor.dimAt(0)) + (1, ) * (ndim - 2)
		strides = (dims[1], ) + (1, ) * (ndim - 1)

		desc = self.tensorDescCache.pop() if len(self.tensorDescCache) > 0 else libmiopen.miopenCreateTensorDescriptor()
		libmiopen.miopenSetTensorDescriptor(desc, dataType.value, dims, strides)

		return DescTensor(desc, dims, tensor)


	def destroyDescribedTensors(self, *descTensors):
		self.tensorDescCache.extend(descTensor.desc for descTensor in descTensors)


	def toTensorAddTensor(self, descTensor, descBias):
		libmiopen.miopenConvolutionForwardBias(
			self.context, 1.0, descBias.desc, descBias.ptr, 0.0, descTensor.desc, descTensor.ptr
		)


	@staticmethod
	def createDescribedConvNd(ndim, stride=1, pad=0, dilation=1, groups=1):
		stride = tuple(stride for _ in range(ndim - 2)) if isinstance(stride, int) else stride
		pad = tuple(pad for _ in range(ndim - 2)) if isinstance(pad, int) else pad
		dilation = tuple(dilation for _ in range(ndim - 2)) if isinstance(dilation, int) else dilation

		mode = ConvMode.conv

		desc = libmiopen.miopenCreateConvolutionDescriptor()
		libmiopen.miopenInitConvolutionNdDescriptor(desc, pad, stride, dilation, mode.value)

		libmiopen.miopenSetConvolutionGroupCount(desc, groups)
		return DescConvNd(desc, mode, stride, pad, dilation, groups)


	@staticmethod
	def destroyDescribedConv(descConv):
		libmiopen.miopenDestroyConvolutionDescriptor(descConv.desc)


	@staticmethod
	def getConvNdOutShape(descConv, descTensor, descW):
		shape = libmiopen.miopenGetConvolutionNdForwardOutputDim(
			descConv.desc, descTensor.desc, descW.desc, len(descTensor.shape)
		)
		return shape


	@staticmethod
	def getConvNdInShape(descConv, descTensor, descW, postpad=0):
		postpad = tuple(postpad for _ in range(len(descConv.pad))) if isinstance(postpad, int) else postpad
		fsize = descW.shape[2:]

		stride, pad, dilation = descConv.stride, descConv.pad, descConv.dilation
		groups = descConv.groups

		shape = tuple(
			stride[d] * (descTensor.shape[d + 2] - 1) + dilation[d] * (fsize[d] - 1) - 2 * pad[d] + 1 + postpad[d]
			for d in range(len(descConv.pad))
		)

		return (descTensor.shape[0], descW.shape[1] * groups) + shape


	@staticmethod
	def cacheKey(descData, descW, descConv):
		key = (
			descData.shape, descW.shape, descConv.mode,
			descConv.stride, descConv.pad, descConv.dilation, descConv.groups, descData.tensor.dtype
		)
		return key


	def convAlgoGetWorkspace(self, descData, descW, descOutData, descConv):
		size = libmiopen.miopenConvolutionForwardGetWorkSpaceSize(
			self.context, descW.desc, descData.desc, descConv.desc, descOutData.desc
		)

		return GPUArray.empty((size, ), dtype=np.uint8) if size > 0 else None


	def cacheConvAlgo(self, descData, descW, descConv, descOutData, algo):
		key = self.cacheKey(descData, descW, descConv)
		perfResults = self.convCache.get(key, None)

		if perfResults is None:
			workspace = self.convAlgoGetWorkspace(descData, descW, descOutData, descConv)
			ptr, size = (workspace.ptr, workspace.size) if workspace is not None else (None, 0)

			perfResults = libmiopen.miopenFindConvolutionForwardAlgorithm(
				self.context, descData.desc, descData.ptr, descW.desc, descW.ptr, descConv.desc, descOutData.desc,
				descOutData.ptr, len(ConvFwdAlgo), ptr, size, 0
			)

			self.convCache[key] = perfResults

		perf = next(p for p in perfResults if p.algo == algo) if algo != -1 else perfResults[0]
		return perf.algo, perf.memory


	def convBackwardDataAlgoGetWorkspace(self, descGrad, descW, descInGrad, descConv):
		size = libmiopen.miopenConvolutionBackwardDataGetWorkSpaceSize(
			self.context, descGrad.desc, descW.desc, descConv.desc, descInGrad.desc
		)

		return GPUArray.empty((size, ), dtype=np.uint8) if size > 0 else None


	def cacheConvBackwardDataAlgo(self, descGrad, descW, descConv, descInGrad, algo):
		key = self.cacheKey(descGrad, descW, descConv)
		perfResults = self.convBackwardDataCache.get(key, None)

		if perfResults is None:
			workspace = self.convBackwardDataAlgoGetWorkspace(descGrad, descW, descInGrad, descConv)
			ptr, size = (workspace.ptr, workspace.size) if workspace is not None else (None, 0)

			perfResults = libmiopen.miopenFindConvolutionBackwardDataAlgorithm(
				self.context, descGrad.desc, descGrad.ptr, descW.desc, descW.ptr, descConv.desc, descInGrad.desc,
				descInGrad.ptr, len(ConvBwdDataAlgo), ptr, size, 0
			)

			self.convBackwardDataCache[key] = perfResults

		perf = next(p for p in perfResults if p.algo == algo) if algo != -1 else perfResults[0]
		return perf.algo, perf.memory


	def convBackwardParamsAlgoGetWorkspace(self, descGrad, descData, descWGrad, descConv):
		size = libmiopen.miopenConvolutionBackwardWeightsGetWorkSpaceSize(
			self.context, descGrad.desc, descData.desc, descConv.desc, descWGrad.desc
		)

		return GPUArray.empty((size, ), dtype=np.uint8) if size > 0 else None


	def cacheConvBackwardParamsAlgo(self, descGrad, descData, descConv, descWGrad, algo):
		key = self.cacheKey(descGrad, descData, descConv)
		perfResults = self.convBackwardParamsCache.get(key, None)

		if perfResults is None:
			workspace = self.convBackwardParamsAlgoGetWorkspace(descGrad, descData, descWGrad, descConv)
			ptr, size = (workspace.ptr, workspace.size) if workspace is not None else (None, 0)

			perfResults = libmiopen.miopenFindConvolutionBackwardWeightsAlgorithm(
				self.context, descGrad.desc, descGrad.ptr, descData.desc, descData.ptr, descConv.desc,
				descWGrad.desc, descWGrad.ptr, len(ConvBwdFilterAlgo), ptr, size, 0
			)

			self.convBackwardParamsCache[key] = perfResults

		perf = next(p for p in perfResults if p.algo == algo) if algo != -1 else perfResults[0]
		return perf.algo, perf.memory


	def convNd(self, data, W, bias=None, stride=1, pad=0, dilation=1, groups=1, algo=ConvFwdAlgo.auto.value,
			   out=None, allocator=None):
		assert data.ndim == W.ndim and data.shape[1] == W.shape[1] * groups

		descData = self.createDescribedNdTensor(data)
		descW = self.createDescribedNdTensor(W)

		descConv = self.createDescribedConvNd(W.ndim, stride, pad, dilation, groups)
		outshape = self.getConvNdOutShape(descConv, descData, descW)

		out = GPUArray.empty(outshape, dtype=data.dtype, allocator=allocator) if out is None else out
		descOutData = self.createDescribedNdTensor(out)

		algo, size = self.cacheConvAlgo(descData, descW, descConv, descOutData, algo)

		workspace = GPUArray.empty((size, ), dtype=np.uint8, allocator=allocator) if size > 0 else None
		ptr, size = (workspace.ptr, workspace.nbytes) if workspace is not None else (None, 0)

		libmiopen.miopenConvolutionForward(
			self.context, 1.0, descData.desc, descData.ptr, descW.desc, descW.ptr, descConv.desc, algo,
			0.0, descOutData.desc, descOutData.ptr, ptr, size
		)

		if bias is not None:
			descBias = self.createDescribed1dTensor(bias, data.ndim)

			self.toTensorAddTensor(descOutData, descBias)
			self.destroyDescribedTensors(descBias)

		self.destroyDescribedConv(descConv)
		self.destroyDescribedTensors(descData, descOutData, descW)

		return out


	def convNdBackwardData(self, grad, W, bias=None, data=None, stride=1, pad=0, dilation=1, postpad=0, groups=1,
						   algo=ConvBwdDataAlgo.auto.value, out=None, allocator=None):
		assert grad.ndim == W.ndim and grad.shape[1] == W.shape[0]

		descGrad = self.createDescribedNdTensor(grad)
		descW = self.createDescribedNdTensor(W)

		descConv = self.createDescribedConvNd(W.ndim, stride, pad, dilation, groups)
		inshape = self.getConvNdInShape(descConv, descGrad, descW, postpad) if data is None else data.shape

		out = GPUArray.empty(inshape, dtype=grad.dtype, allocator=allocator) if out is None else out
		descInGrad = self.createDescribedNdTensor(out)

		algo, size = self.cacheConvBackwardDataAlgo(descGrad, descW, descConv, descInGrad, algo)

		workspace = GPUArray.empty((size, ), dtype=np.uint8, allocator=allocator) if size > 0 else None
		ptr, size = (workspace.ptr, workspace.nbytes) if workspace is not None else (None, 0)

		libmiopen.miopenConvolutionBackwardData(
			self.context, 1.0, descGrad.desc, descGrad.ptr, descW.desc, descW.ptr, descConv.desc, algo,
			0.0, descInGrad.desc, descInGrad.ptr, ptr, size
		)

		if bias is not None:
			descBias = self.createDescribed1dTensor(bias, grad.ndim)
			self.toTensorAddTensor(descInGrad, descBias)

			self.destroyDescribedTensors(descBias)

		self.destroyDescribedConv(descConv)
		self.destroyDescribedTensors(descGrad, descInGrad, descW)

		return out


	def convNdBackwardParams(self, data, grad, W, stride=1, pad=0, dilation=1, groups=1, withbias=False, deconv=False,
							 wgrad=None, bgrad=None, scale=1.0, momentum=0.0, algo=ConvBwdFilterAlgo.auto.value,
							 allocator=None):
		assert data.ndim == grad.ndim and grad.shape[1] == W.shape[0] and data.shape[1] == W.shape[1] * groups

		descData = self.createDescribedNdTensor(data)
		descGrad = self.createDescribedNdTensor(grad)

		descConv = self.createDescribedConvNd(W.ndim, stride, pad, dilation, groups)
		wg, wAccMode = None, False

		if wgrad is not None and (scale != 1.0 or momentum != 0.0):
			wAccMode = True
			wg = GPUArray.empty(W.shape, dtype=W.dtype, allocator=allocator)
		else:
			wgrad = GPUArray.empty(W.shape, dtype=W.dtype, allocator=allocator) if wgrad is None else wgrad
			wg = wgrad

		descWGrad = self.createDescribedNdTensor(wg)
		algo, size = self.cacheConvBackwardParamsAlgo(descGrad, descData, descConv, descWGrad, algo)

		workspace = GPUArray.empty((size, ), dtype=np.uint8, allocator=allocator) if size > 0 else None
		ptr, size = (workspace.ptr, workspace.nbytes) if workspace is not None else (None, 0)

		libmiopen.miopenConvolutionBackwardWeights(
			self.context, 1.0, descGrad.desc, descGrad.ptr, descData.desc, descData.ptr, descConv.desc, algo,
			0.0, descWGrad.desc, descWGrad.ptr, ptr, size
		)

		if wAccMode:
			self.backend.addKer(wgrad.dtype)(wgrad, wgrad, momentum, wg, scale)

		if withbias:
			tensor = descData if deconv else descGrad
			biasshape = (tensor.shape[1], )

			bg, bAccMode = None, False

			if bgrad is not None and (scale != 1.0 or momentum != 0.0):
				bAccMode = True
				bg = GPUArray.empty(biasshape, dtype=data.dtype, allocator=allocator)
			else:
				bgrad = GPUArray.empty(biasshape, dtype=data.dtype, allocator=allocator) if bgrad is None else bgrad
				bg = bgrad

			descBGrad = self.createDescribed1dTensor(bg, data.ndim)

			libmiopen.miopenConvolutionBackwardBias(
				self.context, 1.0, tensor.desc, tensor.ptr, 0.0, descBGrad.desc, descBGrad.ptr
			)

			if bAccMode:
				self.backend.addKer(bgrad.dtype)(bgrad, bgrad, momentum, bg, scale)

			self.destroyDescribedTensors(descBGrad)

		self.destroyDescribedConv(descConv)
		self.destroyDescribedTensors(descData, descGrad, descWGrad)

		return (wgrad, bgrad) if withbias else wgrad


	def convNdbenchmark(self, datashape, Wshape, dtype, stride=1, pad=0, dilation=1, groups=1, algoCount=10,
						exhaustive=False):
		assert len(datashape) == len(Wshape)

		descData = self.createDescribedNdTensor(GPUArray.empty(datashape, dtype=dtype))
		descW = self.createDescribedNdTensor(GPUArray.empty(Wshape, dtype=dtype))

		descConv = self.createDescribedConvNd(len(Wshape), stride, pad, dilation, groups)
		outshape = self.getConvNdOutShape(descConv, descData, descW)

		descWGrad = self.createDescribedNdTensor(GPUArray.empty(Wshape, dtype=dtype))
		descOutData = self.createDescribedNdTensor(GPUArray.empty(outshape, dtype=dtype))

		workspace = self.convAlgoGetWorkspace(descData, descW, descOutData, descConv)
		ptr, size = (workspace.ptr, workspace.size) if workspace is not None else (None, 0)

		perfResults = libmiopen.miopenFindConvolutionForwardAlgorithm(
			self.context, descData.desc, descData.ptr, descW.desc, descW.ptr, descConv.desc,
			descOutData.desc, descOutData.ptr, algoCount, ptr, size, exhaustive
		)

		millisInSec = 1e-3
		fwdResults = [(perf.algo, perf.time * millisInSec, perf.memory) for perf in perfResults]

		workspace = self.convBackwardDataAlgoGetWorkspace(descOutData, descW, descData, descConv)
		ptr, size = (workspace.ptr, workspace.size) if workspace is not None else (None, 0)

		perfResults = libmiopen.miopenFindConvolutionBackwardDataAlgorithm(
			self.context, descOutData.desc, descOutData.ptr, descW.desc, descW.ptr, descConv.desc,
			descData.desc, descData.ptr, algoCount, ptr, size, exhaustive
		)

		bwdDataResults = [(perf.algo, perf.time * millisInSec, perf.memory) for perf in perfResults]

		workspace = self.convBackwardParamsAlgoGetWorkspace(descOutData, descW, descData, descConv)
		ptr, size = (workspace.ptr, workspace.size) if workspace is not None else (None, 0)

		perfResults = libmiopen.miopenFindConvolutionBackwardWeightsAlgorithm(
			self.context, descOutData.desc, descOutData.ptr, descData.desc, descData.ptr, descConv.desc,
			descWGrad.desc, descWGrad.ptr, algoCount, ptr, size, exhaustive
		)

		bwdParamResults = [(perf.algo, perf.time * millisInSec, perf.memory) for perf in perfResults]

		self.destroyDescribedTensors(descData, descOutData, descW, descWGrad)
		self.destroyDescribedConv(descConv)

		return fwdResults, bwdDataResults, bwdParamResults


	@staticmethod
	def createDescribedPool2d(size=2, stride=2, pad=0, mode=PoolMode.max.value):
		size = (size, size) if isinstance(size, int) else size
		hsize, wsize = size

		stride = (stride, stride) if isinstance(stride, int) else stride
		hstride, wstride = stride

		pad = (pad, pad) if isinstance(pad, int) else pad
		hpad, wpad = pad

		desc = libmiopen.miopenCreatePoolingDescriptor()
		libmiopen.miopenSet2dPoolingDescriptor(desc, mode, hsize, wsize, hpad, wpad, hstride, wstride)

		return DescPool2d(desc, size, stride, pad)


	@staticmethod
	def destroyDescribedPool(descPool):
		libmiopen.miopenDestroyPoolingDescriptor(descPool.desc)


	@staticmethod
	def getPool2dOutShape(descPool, descTensor):
		hsize, wsize = descPool.size
		hpad, wpad = descPool.pad
		hstride, wstride = descPool.stride

		outh = (descTensor.shape[2] + 2 * hpad - hsize) // hstride + 1
		outw = (descTensor.shape[3] + 2 * wpad - wsize) // wstride + 1

		return descTensor.shape[:2] + (outh, outw)


	def poolNd(self, data, size=2, stride=2, pad=0, mode=PoolMode.max.value, test=False, out=None, allocator=None):
		assert data.ndim == 4
		descData = self.createDescribedNdTensor(data)

		descPool = self.createDescribedPool2d(size, stride, pad, mode)
		outshape = self.getPool2dOutShape(descPool, descData)

		out = GPUArray.empty(outshape, dtype=data.dtype, allocator=allocator) if out is None else out
		descOutData = self.createDescribedNdTensor(out)

		workspace, ptr, size = None, None, 0
		if not test:
			size = libmiopen.miopenPoolingGetWorkSpaceSize(descOutData.desc)

			workspace = GPUArray.empty((size, ), dtype=np.uint8, allocator=allocator)
			ptr = workspace.ptr

		libmiopen.miopenPoolingForward(
			self.context, descPool.desc, 1.0, descData.desc, descData.ptr, 0.0, descOutData.desc,
			descOutData.ptr, not test, ptr, size
		)

		self.destroyDescribedTensors(descData, descOutData)
		self.destroyDescribedPool(descPool)

		return out if test else (out, workspace)


	def poolNdBackward(self, grad, indata, outdata, workspace, size=2, stride=2, pad=0, mode=PoolMode.max.value,
					   out=None, allocator=None):
		assert grad.ndim == 4
		descGrad = self.createDescribedNdTensor(grad)

		descInData = self.createDescribedNdTensor(indata)
		descOutData = self.createDescribedNdTensor(outdata)

		descPool = self.createDescribedPool2d(size, stride, pad, mode)

		out = GPUArray.empty(indata.shape, dtype=grad.dtype, allocator=allocator) if out is None else out
		descInGrad = self.createDescribedNdTensor(out)

		libmiopen.miopenPoolingBackward(
			self.context, descPool.desc, 1.0, descOutData.desc, descOutData.ptr, descGrad.desc, descGrad.ptr,
			descInData.desc, descInData.ptr, 0.0, descInGrad.desc, descInGrad.ptr, workspace.ptr
		)

		self.destroyDescribedTensors(descInData, descOutData, descGrad, descInGrad)
		self.destroyDescribedPool(descPool)

		return out


	def softmaxNd(self, data, mode=SoftMaxMode.spatial.value, algo=SoftMaxAlgo.accurate.value,
				  out=None, allocator=None):
		descData = self.createDescribedNdTensor(data)

		out = GPUArray.empty(data.shape, dtype=data.dtype, allocator=allocator) if out is None else out
		descOutData = self.createDescribedNdTensor(out)

		libmiopen.miopenSoftmaxForward(
			self.context, 1.0, descData.desc, descData.ptr,
			0.0, descOutData.desc, descOutData.ptr, algo, mode
		)

		self.destroyDescribedTensors(descData, descOutData)
		return out


	def softmaxNdBackward(self, grad, outdata, mode=SoftMaxMode.spatial.value, algo=SoftMaxAlgo.accurate.value,
						  out=None, allocator=None):
		descGrad = self.createDescribedNdTensor(grad)
		descOutData = self.createDescribedNdTensor(outdata)

		out = GPUArray.empty(grad.shape, dtype=grad.dtype, allocator=allocator) if out is None else out
		descInGrad = self.createDescribedNdTensor(out)

		libmiopen.miopenSoftmaxBackward(
			self.context, 1.0, descOutData.desc, descOutData.ptr, descGrad.desc, descGrad.ptr,
			0.0, descInGrad.desc, descInGrad.ptr, algo, mode
		)

		self.destroyDescribedTensors(descOutData, descGrad, descInGrad)
		return out


	def batchNormNd(self, data, mean, var, scale, bias, epsilon=1e-5, factor=1.0, test=False,
					mode=BatchNormMode.spatial.value, out=None, allocator=None):
		assert mean.ndim == 1 and var.ndim == 1 and scale.ndim == 1 and bias.ndim == 1
		assert data.dimAt(1) == mean.dimAt(0)

		descData = self.createDescribedNdTensor(data)
		descScale = self.createDescribed1dTensor(scale, data.ndim)

		out = GPUArray.empty(data.shape, dtype=data.dtype, allocator=allocator) if out is None else out
		descOutData = self.createDescribedNdTensor(out)

		savemean, saveinvvar = None, None

		if test:
			libmiopen.miopenBatchNormalizationForwardInference(
				self.context, mode, 1.0, 0.0, descData.desc, descData.ptr, descOutData.desc, out.ptr,
				descScale.desc, descScale.ptr, bias.ptr, mean.ptr, var.ptr, epsilon
			)

		else:
			savemean = GPUArray.empty(mean.shape, dtype=data.dtype, allocator=allocator)
			saveinvvar = GPUArray.empty(var.shape, dtype=data.dtype, allocator=allocator)

			libmiopen.miopenBatchNormalizationForwardTraining(
				self.context, mode, 1.0, 0.0, descData.desc, descData.ptr, descOutData.desc, out.ptr,
				descScale.desc, descScale.ptr, bias.ptr, factor, mean.ptr, var.ptr, epsilon, savemean.ptr,
				saveinvvar.ptr
			)

		self.destroyDescribedTensors(descData, descOutData, descScale)
		return out if test else (out, savemean, saveinvvar)


	def batchNormNdBackward(self, grad, data, scale, savemean=None, saveinvvar=None, epsilon=1e-5,
							mode=BatchNormMode.spatial.value, out=None, allocator=None):
		assert data.ndim == grad.ndim

		descGrad = self.createDescribedNdTensor(grad)
		descData = self.createDescribedNdTensor(data)
		descScale = self.createDescribed1dTensor(scale, grad.ndim)

		out = GPUArray.empty(grad.shape, dtype=grad.dtype, allocator=allocator) if out is None else out
		descInGrad = self.createDescribedNdTensor(out)

		scalegrad = GPUArray.empty(scale.shape, dtype=scale.dtype, allocator=allocator)
		bgrad = GPUArray.empty(scale.shape, dtype=scale.dtype, allocator=allocator)

		libmiopen.miopenBatchNormalizationBackward(
			self.context, mode, 1.0, 0.0, 1.0, 0.0, descData.desc, descData.ptr, descGrad.desc, descGrad.ptr,
			descInGrad.desc, descInGrad.ptr, descScale.desc, descScale.ptr, scalegrad.ptr, bgrad.ptr,
			epsilon, None if savemean is None else savemean.ptr, None if saveinvvar is None else saveinvvar.ptr
		)

		self.destroyDescribedTensors(descData, descGrad, descInGrad, descScale)
		return out, scalegrad, bgrad


	@staticmethod
	def createDescribedLRN(mode, N=5, alpha=1e-4, beta=0.75, K=2.0):
		desc = libmiopen.miopenCreateLRNDescriptor()
		libmiopen.miopenSetLRNDescriptor(desc, mode, N, alpha, beta, K)

		return DescLRN(desc, N, alpha, beta, K)


	@staticmethod
	def destroyDescribedLRN(descLRN):
		libmiopen.miopenDestroyLRNDescriptor(descLRN.desc)


	def lrn(self, data, N=5, alpha=1e-4, beta=0.75, K=2.0, mode=LRNMode.map.value, test=False,
			out=None, allocator=None):
		descData = self.createDescribedNdTensor(data)
		descLRN = self.createDescribedLRN(mode, N, alpha, beta, K)

		out = GPUArray.empty(data.shape, dtype=data.dtype, allocator=allocator) if out is None else out
		descOutData = self.createDescribedNdTensor(out)

		workspace = None

		if not test:
			size = libmiopen.miopenLRNGetWorkSpaceSize(descOutData.desc)

			if size > 0:
				workspace = GPUArray.empty((size, ), dtype=np.uint8, allocator=allocator)

		libmiopen.miopenLRNForward(
			self.context, descLRN.desc, 1.0, descData.desc, descData.ptr, 0.0,
			descOutData.desc, descOutData.ptr, not test, None if workspace is None else workspace.ptr
		)

		self.destroyDescribedTensors(descData, descOutData)
		self.destroyDescribedLRN(descLRN)

		return out if test else (out, workspace)


	def lrnBackward(self, grad, indata, outdata, workspace, N=5, alpha=1e-4, beta=0.75, K=2.0, mode=LRNMode.map,
					out=None, allocator=None):
		descGrad = self.createDescribedNdTensor(grad)

		descInData = self.createDescribedNdTensor(indata)
		descOutData = self.createDescribedNdTensor(outdata)

		out = GPUArray.empty(grad.shape, dtype=grad.dtype, allocator=allocator) if out is None else out
		descInGrad = self.createDescribedNdTensor(out)

		descLRN = self.createDescribedLRN(mode, N, alpha, beta, K)

		libmiopen.miopenLRNBackward(
			self.context, descLRN.desc, 1.0, descOutData.desc, descOutData.ptr, descGrad.desc, descGrad.ptr,
			descInData.desc, descInData.ptr, 0.0, descInGrad.desc, descInGrad.ptr, workspace.ptr
		)

		self.destroyDescribedTensors(descGrad, descInData, descOutData, descInGrad)
		self.destroyDescribedLRN(descLRN)

		return out


def unittest():
	from PuzzleLib.Hip import Backend

	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx, initmode=1)

		for dtype, atol in bnd.dtypesSupported():
			conv2dTest(bnd, dtype, atol)
			conv3dTest(bnd, dtype, atol)
			convGroupTest(bnd, dtype, atol)

			deconv2dTest(bnd, dtype, atol)
			deconv3dTest(bnd, dtype, atol)
			deconvGroupTest(bnd, dtype, atol)

			maxpool2dTest(bnd, dtype, atol)
			softmax2dTest(bnd, dtype, atol)


def maxpool2dTest(bnd, dtype, atol):
	batchsize, maps, h, w = 3, 2, 6, 6
	size, stride, pad = 3, 2, 1

	hostData = np.full(shape=(batchsize, maps, h + 2 * pad, w + 2 * pad), fill_value=np.finfo(dtype).min, dtype=dtype)
	hostData[:, :, pad:-pad, pad:-pad] = np.random.randn(batchsize, maps, h, w).astype(dtype)

	data = GPUArray.toGpu(hostData[:, :, pad:-pad, pad:-pad])
	outdata, workspace = bnd.dnn.poolNd(data, size=size, stride=stride, pad=pad, mode=PoolMode.max.value)

	hostOutData = np.empty(outdata.shape, dtype=dtype)

	for b, c, y, x in itertools.product(
		range(batchsize), range(maps), range(hostOutData.shape[2]), range(hostOutData.shape[3])
	):
		hostOutData[b, c, y, x] = np.max(hostData[b, c, y * stride:y * stride + size, x * stride:x * stride + size])

	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.poolNdBackward(
		grad, data, outdata, workspace, size=size, stride=stride, pad=pad, mode=PoolMode.max.value
	)

	hostInGrad = np.zeros(hostData.shape, dtype=dtype)

	for b, c, y, x, dy, dx in itertools.product(
		range(batchsize), range(maps), range(hostOutData.shape[2]), range(hostOutData.shape[3]),
		range(size), range(size)
	):
		if hostData[b, c, y * stride + dy, x * stride + dx] == hostOutData[b, c, y, x]:
			hostInGrad[b, c, y * stride + dy, x * stride + dx] += hostGrad[b, c, y, x]

	hostInGrad = hostInGrad[:, :, pad:-pad, pad:-pad].astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)


if __name__ == "__main__":
	unittest()
