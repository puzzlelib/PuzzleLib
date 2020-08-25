from enum import Enum
import numpy as np

from PuzzleLib.Cuda.GPUBackend import GPUBackend

from PuzzleLib.Cuda import Driver as CudaDriver
from PuzzleLib.Cuda.Driver import CudaError, CuRand, CuBlas, CuDnn

from PuzzleLib.Cuda.GPUArray import extendGPUArray
from PuzzleLib.Cuda.SourceModule import SourceModule, ElementwiseKernel, ElementHalf2Kernel, ReductionKernel
from PuzzleLib.Cuda.Utils import SharedArray, prod


cudaWarpBit, cudaBlockBit = 5, 9
cudaWarpSize, cudaBlockSize = 1 << cudaWarpBit, 1 << cudaBlockBit


class CudaSourceModule(SourceModule):
	Driver = CudaDriver


class CudaEltwiseKernel(ElementwiseKernel):
	Driver = CudaDriver
	SourceModule = CudaSourceModule

	warpBit, warpSize = cudaWarpBit, cudaWarpSize
	blockBit, blockSize = cudaBlockBit, cudaBlockSize


class CudaEltHalf2Kernel(ElementHalf2Kernel):
	Driver = CudaDriver
	SourceModule = CudaSourceModule

	warpBit, warpSize = cudaWarpBit, cudaWarpSize
	blockBit, blockSize = cudaBlockBit, cudaBlockSize


class CudaReductionKernel(ReductionKernel):
	Driver = CudaDriver
	SourceModule = CudaSourceModule

	warpBit, warpSize = cudaWarpBit, cudaWarpSize
	blockBit, blockSize = cudaBlockBit, cudaBlockSize


class CudaSharedArray(SharedArray):
	GPUArray = CudaDriver.GPUArray


class CudaBackend(GPUBackend):
	BackendName = "Cuda"

	warpSize = cudaWarpSize
	nthreads = 1024

	Driver = CudaDriver
	GPUArray = extendGPUArray(CudaDriver, CudaEltwiseKernel, CudaEltHalf2Kernel, CudaReductionKernel)
	Error = CudaError

	SourceModule = CudaSourceModule
	ElementwiseKernel = CudaEltwiseKernel
	ElementHalf2Kernel = CudaEltHalf2Kernel
	ReductionKernel = CudaReductionKernel

	SharedArray = CudaSharedArray
	Rand, Blas, Dnn = CuRand, CuBlas, CuDnn


	class GroupFormat(Enum):
		gbp = CuBlas.GROUPFORMAT_GBP
		bgp = CuBlas.GROUPFORMAT_BGP


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


	class BatchNormMode(Enum):
		perActivation = CuDnn.BATCHNORM_MODE_PER_ACTIVATION
		spatial = CuDnn.BATCHNORM_MODE_SPATIAL
		spatialPersistent = CuDnn.BATCHNORM_MODE_SPATIAL_PERSISTENT


	class RNNAlgo(Enum):
		standard = CuDnn.RNN_ALGO_STANDARD
		persistStatic = CuDnn.RNN_ALGO_PERSIST_STATIC
		persistDynamic = CuDnn.RNN_ALGO_PERSIST_DYNAMIC


	class RNNMode(Enum):
		relu = CuDnn.RNN_MODE_RELU
		tanh = CuDnn.RNN_MODE_TANH
		lstm = CuDnn.RNN_MODE_LSTM
		gru = CuDnn.RNN_MODE_GRU


	class DirectionMode(Enum):
		uni = CuDnn.RNN_DIRECTION_UNIDIRECTIONAL
		bi = CuDnn.RNN_DIRECTION_BIDIRECTIONAL


	class ConvPerf:
		def __init__(self, algo, tm, memory, determinism, mathType):
			self.algo = algo
			self.time = tm
			self.memory = memory
			self.determinism = determinism == 1
			self.mathType = CudaBackend.MathType(mathType)


		def toString(self):
			return "%-40s %-25s %-28s %-20s %s" % (
				"Algo %s" % self.algo, "time %.6f secs" % self.time,
				"memory %.6f mbytes" % (self.memory / 1024 ** 2), "determinism=%s" % self.determinism,
				"mathType=%s" % self.mathType
			)


		def __str__(self):
			return self.toString()


		def __repr__(self):
			return self.toString()


	def createRnn(self, insize, hsize, dtype, layers=1, algo=None, mode=None, direction=None, dropout=0.0,
				  seed=0, batchsize=0):
		algo = self.RNNAlgo.standard if algo is None else algo
		mode = self.RNNMode.lstm if mode is None else mode
		direction = self.DirectionMode.uni if direction is None else direction

		rnn = self.Dnn.Rnn(
			self.dnn, insize, hsize, np.dtype(dtype), layers, algo.value, mode.value, direction.value,
			dropout, seed, batchsize
		)

		W = self.GPUArray.empty((rnn.wsize, ), dtype=dtype)
		params = self.acquireRnnParams(rnn, W)

		return rnn, W, params


	def deviceSupportsBatchHint(self):
		return self.device.computeCapability() >= (6, 1)


	def acquireRnnParams(self, rnn, W):
		mode = self.RNNMode(rnn.mode)

		if mode == self.RNNMode.relu or mode == self.RNNMode.tanh:
			return self.acquireNativeRnnParams(rnn, W)
		elif mode == self.RNNMode.lstm:
			return self.acquireLSTMParams(rnn, W)
		elif mode == self.RNNMode.gru:
			return self.acquireGRUParams(rnn, W)
		else:
			raise NotImplementedError(mode.value)


	def getRnnParam(self, rnn, W, layer, linLayer, Wshape):
		Wtuple, biasTuple = rnn.getParam(W, layer, linLayer)

		Woffset, wsize = Wtuple
		biasOffset, biasSize = biasTuple

		dtype, gpudata = W.dtype, W.gpudata
		Wbytes, biasBytes = wsize * dtype.itemsize, biasSize * dtype.itemsize

		assert prod(Wshape) == wsize
		w = self.GPUArray(Wshape, dtype=W.dtype, gpudata=W.gpudata[Woffset:Woffset + Wbytes])

		bias = self.GPUArray((biasSize, ), dtype=W.dtype, gpudata=W.gpudata[biasOffset:biasOffset + biasBytes])
		return w, bias


	def acquireNativeRnnParams(self, rnn, W):
		direction = self.DirectionMode(rnn.direction)

		linLayers = 2
		layers = rnn.layers if direction == self.DirectionMode.uni else rnn.layers * 2

		layerTypes = {0: "w", 1: "r"}

		params = []
		for layer in range(layers):
			layerparams = {}
			for linLayer in range(linLayers):
				if linLayer == 0:
					if layer == 0 or layer == 1 and direction == self.DirectionMode.bi:
						size = rnn.insize
					else:
						size = 2 * rnn.hsize if direction == self.DirectionMode.bi else rnn.hsize

					shape = (rnn.hsize, size)

				elif linLayer == 1:
					shape = (rnn.hsize, rnn.hsize)

				else:
					assert False

				w, bias = self.getRnnParam(rnn, W, layer, linLayer, shape)
				T = layerTypes[linLayer]

				Wname = "%si" % T
				assert Wname not in layerparams

				biasname = "b%si" % T
				assert biasname not in layerparams

				layerparams[Wname] = w
				layerparams[biasname] = bias

			params.append(layerparams)

		return params


	def acquireLSTMParams(self, rnn, W):
		direction = self.DirectionMode(rnn.direction)

		linLayers = 8
		layers = rnn.layers if direction == self.DirectionMode.uni else rnn.layers * 2

		layerTypes = {
			0: "i", 4: "i",
			1: "f", 5: "f",
			2: "c", 6: "c",
			3: "o", 7: "o"
		}

		params = []
		for layer in range(layers):
			layerparams = {}
			for linLayer in range(linLayers):
				if linLayer < 4:
					if layer == 0 or layer == 1 and direction == self.DirectionMode.bi:
						size = rnn.insize
					else:
						size = 2 * rnn.hsize if direction == self.DirectionMode.bi else rnn.hsize

					shape, wtype = (rnn.hsize, size), "w"

				else:
					shape, wtype = (rnn.hsize, rnn.hsize), "r"

				w, bias = self.getRnnParam(rnn, W, layer, linLayer, shape)
				T = layerTypes[linLayer]

				Wname = "%s%s" % (wtype, T)
				assert Wname not in layerparams

				biasname = "b%s%s" % (wtype, T)
				assert biasname not in layerparams

				layerparams[Wname] = w
				layerparams[biasname] = bias

			params.append(layerparams)

		return params


	def acquireGRUParams(self, rnn, W):
		direction = self.DirectionMode(rnn.direction)

		linLayers = 6
		layers = rnn.layers if direction == self.DirectionMode.uni else rnn.layers * 2

		layerTypes = {
			0: "r", 3: "r",
			1: "i", 4: "i",
			2: "h", 5: "h"
		}

		params = []
		for layer in range(layers):
			layerparams = {}
			for linLayer in range(linLayers):
				if linLayer < 3:
					if layer == 0 or layer == 1 and direction == self.DirectionMode.bi:
						size = rnn.insize
					else:
						size = 2 * rnn.hsize if direction == self.DirectionMode.bi else rnn.hsize

					shape, wtype = (rnn.hsize, size), "w"

				else:
					shape, wtype = (rnn.hsize, rnn.hsize), "r"

				w, bias = self.getRnnParam(rnn, W, layer, linLayer, shape)
				T = layerTypes[linLayer]

				Wname = "%s%s" % (wtype, T)
				assert Wname not in layerparams

				biasname = "b%s%s" % (wtype, T)
				assert biasname not in layerparams

				layerparams[Wname] = w
				layerparams[biasname] = bias

			params.append(layerparams)

		return params


backendCache = {}


def getDeviceCount():
	return CudaBackend.Driver.Device.count()


def getBackend(deviceIdx=0, initmode=0, logger=None):
	bnd = backendCache.get(deviceIdx, None)

	if bnd is None:
		bnd = CudaBackend(deviceIdx, initmode, logger=logger)
		backendCache[deviceIdx] = bnd

	else:
		bnd.updateBackend(initmode, logger=logger)

	return bnd
