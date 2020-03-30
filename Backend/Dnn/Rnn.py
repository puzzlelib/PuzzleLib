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
	if not Config.shouldInit():
		return

	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.hip:
		initHip()
	elif Config.isCPUBased(Config.backend):
		initCPU()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda import Backend
	initGPU(Backend)


def initHip():
	from PuzzleLib.Hip import Backend
	initGPU(Backend)


def initGPU(Backend):
	backend = Backend.getBackend(Config.deviceIdx, initmode=1)
	memoryPool = backend.memoryPool

	global RNNMode, DirectionMode
	RNNMode = backend.RNNMode
	DirectionMode = backend.DirectionMode

	def wrapCreateRnn(insize, hsize, layers, mode, direction, dropout, seed, batchsize):
		import numpy as np

		rnn, W, params = backend.createRnn(
			insize, hsize, np.float32, layers, mode=mode, direction=direction, dropout=dropout,
			seed=seed, batchsize=0 if batchsize is None else batchsize
		)

		return rnn, W, {i: layer for i, layer in enumerate(params)}

	def wrapAcquireRnnParams(descRnn, w):
		params = backend.acquireRnnParams(descRnn, w)
		return w, params

	def wrapUpdateRnnParams(descRnn, w, params):
		params = [params[layer]for layer in sorted(params.keys())]
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


def initCPU():
	global deviceSupportsBatchHint
	deviceSupportsBatchHint = lambda: False


autoinit()
