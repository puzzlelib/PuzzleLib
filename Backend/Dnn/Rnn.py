from PuzzleLib import Config


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
	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.opencl:
		initOpenCL()
	elif Config.isCPUBased(Config.backend):
		initCPU()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda.Utils import getDeviceComputeCap, memoryPool
	from PuzzleLib.Cuda.Wrappers import CuDnnRnn

	global RNNMode, DirectionMode
	RNNMode = CuDnnRnn.RNNMode
	DirectionMode = CuDnnRnn.DirectionMode

	def wrapCreateRnn(insize, hsize, layers, mode, direction, dropout, seed, batchsize):
		import numpy as np
		rnn, W, params = CuDnnRnn.createRnn(
			insize, hsize, np.float32, layers, mode=mode, direction=direction, dropout=dropout,
			seed=seed, batchsize=0 if batchsize is None else batchsize
		)

		return rnn, W, {i: layer for i, layer in enumerate(params)}

	def wrapAcquireRnnParams(descRnn, w):
		params = CuDnnRnn.acquireRnnParams(descRnn, w)
		return w, params

	global createRnn, acquireRnnParams, updateRnnParams
	createRnn = wrapCreateRnn
	acquireRnnParams = wrapAcquireRnnParams
	updateRnnParams = lambda descRnn, W, params: None

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
	deviceSupportsBatchHint = lambda: getDeviceComputeCap() >= (6, 1)


def initOpenCL():
	from PuzzleLib.OpenCL.Wrappers import MIOpenRnn

	global RNNMode, DirectionMode
	RNNMode = MIOpenRnn.RNNMode
	DirectionMode = MIOpenRnn.DirectionMode

	def wrapCreateRnn(insize, hsize, layers, mode, direction, _, seed, batchsize):
		assert batchsize is None
		return MIOpenRnn.createRnn(insize, hsize, layers, mode, direction)

	global createRnn, acquireRnnParams, updateRnnParams
	createRnn = wrapCreateRnn
	acquireRnnParams = MIOpenRnn.acquireRnnParams
	updateRnnParams = MIOpenRnn.updateRnnParams

	global forwardRnn, backwardDataRnn, backwardParamsRnn
	forwardRnn = MIOpenRnn.forwardRnn
	backwardDataRnn = MIOpenRnn.backwardDataRnn
	backwardParamsRnn = MIOpenRnn.backwardParamsRnn

	global deviceSupportsBatchHint
	deviceSupportsBatchHint = lambda: False


def initCPU():
	global deviceSupportsBatchHint
	deviceSupportsBatchHint = lambda: False


autoinit()
