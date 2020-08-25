from enum import Enum

from PuzzleLib.Cuda.GPUBackend import GPUBackend
from PuzzleLib.Cuda.Kernels.Memory import MemoryModule

from PuzzleLib.Hip import Driver as HipDriver
from PuzzleLib.Hip.Driver import HipError, HipRand, RocBlas

from PuzzleLib.Hip.GPUArray import HipGPUArray
from PuzzleLib.Hip.SourceModule import hipWarpSize, hipBlockSize
from PuzzleLib.Hip.SourceModule import HipSourceModule, HipEltwiseKernel, HipEltHalf2Kernel, HipReductionKernel
from PuzzleLib.Hip.Utils import HipSharedArray

from PuzzleLib.Hip.Kernels.CTC import HipCTCModule
from PuzzleLib.Hip.Wrappers import MIOpen, MIOpenRnn


class HipBackend(GPUBackend):
	BackendName = "Hip"

	warpSize = hipWarpSize
	nthreads = hipBlockSize

	Driver = HipDriver
	GPUArray = HipGPUArray
	Error = HipError

	SharedArray = HipSharedArray
	Rand, Blas, Dnn = HipRand, RocBlas, MIOpen

	SourceModule = HipSourceModule
	ElementwiseKernel = HipEltwiseKernel
	ElementHalf2Kernel = HipEltHalf2Kernel
	ReductionKernel = HipReductionKernel


	class GroupFormat(Enum):
		gbp = RocBlas.GROUPFORMAT_GBP
		bgp = RocBlas.GROUPFORMAT_BGP


	ConvPerf = MIOpen.ConvPerf

	ConvFwdAlgo = MIOpen.ConvFwdAlgo
	ConvBwdDataAlgo = MIOpen.ConvBwdDataAlgo
	ConvBwdFilterAlgo = MIOpen.ConvBwdFilterAlgo

	PoolMode = MIOpen.PoolMode
	SoftMaxMode = MIOpen.SoftMaxMode

	BatchNormMode = MIOpen.BatchNormMode
	LRNMode = MIOpen.LRNMode

	RNNAlgo, RNNMode, DirectionMode = MIOpenRnn.RNNAlgo, MIOpenRnn.RNNMode, MIOpenRnn.DirectionMode


	def __init__(self, deviceIdx, initmode=0, logger=None):
		self.memmod = None
		super().__init__(deviceIdx, initmode, logger=logger)


	def initLibs(self, logger=None):
		self.blas = self.Blas.BlasContext().enableTensorOps(True)

		if logger is not None:
			logger.debug("Created %s context (Using version: %s)", self.Blas.__name__, self.blas.getVersion())

		self.dnn = self.Dnn.DnnContext(self).enableTensorOps(True)

		if logger is not None:
			logger.debug("Created %s context (Using version: %s)", self.Dnn.__name__, self.dnn.getVersion())


	def initKernels(self):
		super().initKernels()

		self.ctcmod = HipCTCModule(self)
		self.memmod = MemoryModule(self)


	def createRnn(self, insize, hsize, dtype, layers=1, algo=None, mode=None, direction=None, dropout=0.0,
				  seed=0, batchsize=0):
		mode = self.RNNMode.lstm if mode is None else mode
		direction = self.DirectionMode.uni if direction is None else direction

		rnn = MIOpenRnn.Rnn(self.dnn, insize, hsize, dtype, layers, mode, direction)

		W = self.GPUArray.empty(rnn.descW.shape, dtype=rnn.dtype)
		params = self.acquireRnnParams(rnn, W)

		return rnn, W, params


	def acquireRnnParams(self, rnn, W):
		if rnn.mode == self.RNNMode.relu or rnn.mode == self.RNNMode.tanh:
			return self.acquireNativeRnnParams(rnn, W)
		elif rnn.mode == self.RNNMode.lstm:
			return self.acquireLSTMParams(rnn, W)
		elif rnn.mode == self.RNNMode.gru:
			return self.acquireGRUParams(rnn, W)
		else:
			raise NotImplementedError(rnn.mode.value)


	def getRnnParam(self, rnn, layer, W, linLayer, shape):
		linLayerMatDesc = MIOpenRnn.libmiopen.miopenCreateTensorDescriptor()

		size = MIOpenRnn.libmiopen.miopenGetRNNLayerParamSize(
			self.dnn.context, rnn.desc, layer, rnn.descData.desc, linLayer
		)
		w = self.GPUArray.empty((size // W.dtype.itemsize, ), dtype=W.dtype)

		MIOpenRnn.libmiopen.miopenGetRNNLayerParam(
			self.dnn.context, rnn.desc, layer, rnn.descData.desc, rnn.descW.desc, W.ptr, linLayer,
			linLayerMatDesc, w.ptr
		)

		MIOpenRnn.libmiopen.miopenDestroyTensorDescriptor(linLayerMatDesc)

		linLayerBiasDesc = MIOpenRnn.libmiopen.miopenCreateTensorDescriptor()
		size = MIOpenRnn.libmiopen.miopenGetRNNLayerBiasSize(self.dnn.context, rnn.desc, layer, linLayer)

		bias = self.GPUArray.empty((size // W.dtype.itemsize, ), dtype=W.dtype)

		MIOpenRnn.libmiopen.miopenGetRNNLayerBias(
			self.dnn.context, rnn.desc, layer, rnn.descData.desc, rnn.descW.desc, W.ptr, linLayer,
			linLayerBiasDesc, bias.ptr
		)

		MIOpenRnn.libmiopen.miopenDestroyTensorDescriptor(linLayerBiasDesc)
		return w.reshape(shape), bias


	def acquireNativeRnnParams(self, rnn, W):
		linLayers = 2
		layers = rnn.layers if rnn.direction == self.DirectionMode.uni else rnn.layers * 2

		layerTypes = {0: "w", 1: "r"}

		params = []
		for layer in range(layers):
			layerparams = {}
			for linLayer in range(linLayers):
				if linLayer == 0:
					if layer == 0 or layer == 1 and rnn.direction == self.DirectionMode.bi:
						size = rnn.insize
					else:
						size = 2 * rnn.hsize if rnn.direction == self.DirectionMode.bi else rnn.hsize

					shape = (rnn.hsize, size)

				elif linLayer == 1:
					shape = (rnn.hsize, rnn.hsize)

				else:
					assert False

				w, bias = self.getRnnParam(rnn, layer, W, linLayer, shape)
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
		linLayers = 8
		layers = rnn.layers if rnn.direction == self.DirectionMode.uni else rnn.layers * 2

		layerTypes = {
			0: "i", 4: "i",
			1: "f", 5: "f",
			2: "o", 6: "o",
			3: "c", 7: "c"
		}

		params = []
		for layer in range(layers):
			layerparams = {}
			for linLayer in range(linLayers):
				if linLayer < 4:
					if layer == 0 or layer == 1 and rnn.direction == self.DirectionMode.bi:
						size = rnn.insize
					else:
						size = 2 * rnn.hsize if rnn.direction == self.DirectionMode.bi else rnn.hsize

					shape, wtype = (rnn.hsize, size), "w"

				else:
					shape, wtype = (rnn.hsize, rnn.hsize), "r"

				w, bias = self.getRnnParam(rnn, layer, W, linLayer, shape)
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
		linLayers = 6
		layers = rnn.layers if rnn.direction == self.DirectionMode.uni else rnn.layers * 2

		layerTypes = {
			0: "i", 3: "i",
			1: "r", 4: "r",
			2: "h", 5: "h"
		}

		params = []
		for layer in range(layers):
			layerparams = {}
			for linLayer in range(linLayers):
				if linLayer < 3:
					if layer == 0 or layer == 1 and rnn.direction == self.DirectionMode.bi:
						size = rnn.insize
					else:
						size = 2 * rnn.hsize if rnn.direction == self.DirectionMode.bi else rnn.hsize

					shape, wtype = (rnn.hsize, size), "w"

				else:
					shape, wtype = (rnn.hsize, rnn.hsize), "r"

				w, bias = self.getRnnParam(rnn, layer, W, linLayer, shape)
				T = layerTypes[linLayer]

				Wname = "%s%s" % (wtype, T)
				assert Wname not in layerparams

				biasname = "b%s%s" % (wtype, T)
				assert biasname not in layerparams

				layerparams[Wname] = w
				layerparams[biasname] = bias

			params.append(layerparams)

		return params


	def updateRnnParams(self, rnn, W, params):
		if rnn.mode == self.RNNMode.relu or rnn.mode == self.RNNMode.tanh:
			self.updateNativeRnnParams(rnn, W, params)
		elif rnn.mode == self.RNNMode.lstm:
			self.updateLSTMParams(rnn, W, params)
		elif rnn.mode == self.RNNMode.gru:
			self.updateGRUParams(rnn, W, params)
		else:
			raise NotImplementedError(rnn.mode.value)


	def setRnnParam(self, rnn, layer, W, linLayer, linLayerMat, linLayerBias):
		descLinLayerMat = self.dnn.createDescribedNdTensor(linLayerMat)
		descLinLayerBias = self.dnn.createDescribedNdTensor(linLayerBias)

		MIOpenRnn.libmiopen.miopenSetRNNLayerParam(
			self.dnn.context, rnn.desc, layer, rnn.descData.desc, rnn.descW.desc, W.ptr,
			linLayer, descLinLayerMat.desc, descLinLayerMat.ptr
		)

		MIOpenRnn.libmiopen.miopenSetRNNLayerBias(
			self.dnn.context, rnn.desc, layer, rnn.descData.desc, rnn.descW.desc, W.ptr,
			linLayer, descLinLayerBias.desc, descLinLayerBias.ptr
		)

		self.dnn.destroyDescribedTensors(descLinLayerMat, descLinLayerBias)


	def updateNativeRnnParams(self, rnn, W, params):
		linLayers = {"wi": 0, "ri": 1}

		for layer, subparams in enumerate(params):
			for name, param in subparams.items():
				if name[0] != "b":
					self.setRnnParam(rnn, layer, W, linLayers[name], param, subparams["b%s" % name])


	def updateLSTMParams(self, rnn, W, params):
		layerBases = {"w": 0, "r": 4}
		layerTypes = {"i": 0, "f": 1, "o": 2, "c": 3}

		for layer, subparams in enumerate(params):
			for name, param in subparams.items():
				if name[0] != "b":
					linLayer = layerBases[name[0]] + layerTypes[name[1]]
					self.setRnnParam(rnn, layer, W, linLayer, param, subparams["b%s" % name])


	def updateGRUParams(self, rnn, W, params):
		layerBases = {"w": 0, "r": 3}
		layerTypes = {"i": 0, "r": 1, "h": 2}

		for layer, subparams in enumerate(params):
			for name, param in subparams.items():
				if name[0] != "b":
					linLayer = layerBases[name[0]] + layerTypes[name[1]]
					self.setRnnParam(rnn, layer, W, linLayer, param, subparams["b%s" % name])


backendCache = {}


def getDeviceCount():
	return HipBackend.Driver.Device.count()


def getBackend(deviceIdx=0, initmode=0, logger=None):
	bnd = backendCache.get(deviceIdx, None)

	if bnd is None:
		bnd = HipBackend(deviceIdx, initmode, logger=logger)
		backendCache[deviceIdx] = bnd

	else:
		bnd.updateBackend(initmode, logger=logger)

	return bnd
