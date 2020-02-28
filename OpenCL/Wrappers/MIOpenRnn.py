from enum import Enum
from collections import namedtuple

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.ThirdParty import libmiopen
from PuzzleLib.OpenCL.Utils import queue, memoryPool as memPool
from PuzzleLib.OpenCL.Wrappers.MIOpen import context, DataType, createDescribedNdTensor, destroyDescribedTensors


class RNNMode(Enum):
	relu = libmiopen.miopenRNNMode["miopenRNNRELU"]
	tanh = libmiopen.miopenRNNMode["miopenRNNTANH"]
	lstm = libmiopen.miopenRNNMode["miopenLSTM"]
	gru = libmiopen.miopenRNNMode["miopenGRU"]


class DirectionMode(Enum):
	uni = libmiopen.miopenRNNDirectionMode["miopenRNNunidirection"]
	bi = libmiopen.miopenRNNDirectionMode["miopenRNNbidirection"]


class RNNInputMode(Enum):
	linearInput = libmiopen.miopenRNNInputMode["miopenRNNlinear"]
	skipInput = libmiopen.miopenRNNInputMode["miopenRNNskip"]


class RNNAlgo(Enum):
	default = libmiopen.miopenRNNAlgo["miopenRNNdefault"]


class BiasMode(Enum):
	noBias = libmiopen.miopenRNNBiasMode["miopenRNNNoBias"]
	withBias = libmiopen.miopenRNNBiasMode["miopenRNNwithBias"]


class GEMMAlgoMode(Enum):
	gemm = libmiopen.miopenRNNGEMMalgoMode["miopenRNNAlgoGEMM"]


DescRNN = namedtuple("DescRnn", "desc insize hsize layers mode inputMode dir algo")


def createDescribedRnn(insize, hsize, layers=1, mode=RNNMode.lstm, inputMode=RNNInputMode.linearInput,
					   direction=DirectionMode.uni, dataType=DataType.float, algo=RNNAlgo.default):
	desc = libmiopen.miopenCreateRNNDescriptor()

	if dataType != DataType.float:
		raise NotImplementedError()

	libmiopen.miopenSetRNNDescriptor(desc, hsize, layers, inputMode.value, direction.value, mode.value,
									 BiasMode.withBias.value, algo.value, dataType.value)

	return DescRNN(desc=desc, insize=insize, hsize=hsize, layers=layers, mode=mode, inputMode=inputMode, dir=direction,
				   algo=algo)


def destroyDescribedRnn(descRnn):
	libmiopen.miopenDestroyRNNDescriptor(descRnn.desc)


def createRnnParams(descRnn, insize, dataType=DataType.float, w=None):
	descData = createDescribedNdTensor(dims=(1, insize), strides=(insize, 1), tensor=None)

	if dataType != DataType.float:
		raise NotImplementedError()

	wsize = libmiopen.miopenGetRNNParamsSize(context, descRnn.desc, descData.desc, dataType.value)
	nparams = wsize // np.float32(0).itemsize

	if w is None:
		w = Driver.empty(queue, (nparams, ), dtype=np.float32, allocator=memPool)
	elif w.nbytes != wsize:
		raise RuntimeError("Bad weights buffer size (got %s, expected %s)" % (w.nbytes, wsize))

	destroyDescribedTensors(descData)
	return w


def acquireRnnParams(descRnn, w):
	descData = createDescribedNdTensor(dims=(1, descRnn.insize), strides=(descRnn.insize, 1), tensor=None)
	descW = createDescribedNdTensor(None, None, w)

	if descRnn.mode == RNNMode.relu or descRnn.mode == RNNMode.tanh:
		params = acquireNativeRnnParams(descRnn, descData, descW)
	elif descRnn.mode == RNNMode.lstm:
		params = acquireLSTMParams(descRnn, descData, descW)
	elif descRnn.mode == RNNMode.gru:
		params = acquireGRUParams(descRnn, descData, descW)
	else:
		raise NotImplementedError()

	destroyDescribedTensors(descData, descW)
	return w, params


def acquireNativeRnnParams(descRnn, descData, descW):
	linLayers = 2
	layers = descRnn.layers if descRnn.dir == DirectionMode.uni else descRnn.layers * 2

	params = {}
	for layer in range(layers):
		layerparams = {}
		for linLayer in range(linLayers):
			linLayerMat, linLayerBias = getRNNParam(descRnn, layer, descData, descW, linLayer)

			if linLayer == 0:
				if layer == 0 or layer == 1 and descRnn.dir == DirectionMode.bi:
					size = descRnn.insize
				else:
					size = 2 * descRnn.hsize if descRnn.dir == DirectionMode.bi else descRnn.hsize

				linLayerMat = linLayerMat.reshape(descRnn.hsize, size)
				typ = "w"

			elif linLayer == 1:
				linLayerMat = linLayerMat.reshape(descRnn.hsize, descRnn.hsize)
				typ = "r"

			else:
				raise RuntimeError("Bad linear layer index %s" % linLayer)

			layerparams["%si" % typ] = linLayerMat
			layerparams["b%si" % typ] = linLayerBias.reshape(linLayerBias.shape[0])

		params[layer] = layerparams

	return params


def acquireLSTMParams(descRnn, descData, descW):
	linLayers = 8
	layers = descRnn.layers if descRnn.dir == DirectionMode.uni else descRnn.layers * 2

	params = {}
	for layer in range(layers):
		layerparams = {}
		for linLayer in range(linLayers):
			linLayerMat, linLayerBias = getRNNParam(descRnn, layer, descData, descW, linLayer)

			if linLayer == 0 or linLayer == 4:
				typ = "i"
			elif linLayer == 1 or linLayer == 5:
				typ = "f"
			elif linLayer == 2 or linLayer == 6:
				typ = "o"
			elif linLayer == 3 or linLayer == 7:
				typ = "c"
			else:
				raise RuntimeError("Bad linear layer index: %s" % linLayer)

			if linLayer < 4:
				if layer == 0 or layer == 1 and descRnn.dir == DirectionMode.bi:
					size = descRnn.insize
				else:
					size = 2 * descRnn.hsize if descRnn.dir == DirectionMode.bi else descRnn.hsize

				linLayerMat = linLayerMat.reshape(descRnn.hsize, size)
				wtype = "w"

			else:
				linLayerMat = linLayerMat.reshape(descRnn.hsize, descRnn.hsize)
				wtype = "r"

			layerparams["%s%s" % (wtype, typ)] = linLayerMat
			layerparams["b%s%s" % (wtype, typ)] = linLayerBias.reshape(linLayerBias.shape[0])

		params[layer] = layerparams

	return params


def acquireGRUParams(descRnn, descData, descW):
	linLayers = 6
	layers = descRnn.layers if descRnn.dir == DirectionMode.uni else descRnn.layers * 2

	params = {}
	for layer in range(layers):
		layerparams = {}
		for linLayer in range(linLayers):
			linLayerMat, linLayerBias = getRNNParam(descRnn, layer, descData, descW, linLayer)

			if linLayer == 0 or linLayer == 3:
				typ = "i"
			elif linLayer == 1 or linLayer == 4:
				typ = "r"
			elif linLayer == 2 or linLayer == 5:
				typ = "h"
			else:
				raise RuntimeError("Bad linear layer index: %s" % linLayer)

			if linLayer < 3:
				if layer == 0 or layer == 1 and descRnn.dir == DirectionMode.bi:
					size = descRnn.insize
				else:
					size = 2 * descRnn.hsize if descRnn.dir == DirectionMode.bi else descRnn.hsize

				linLayerMat = linLayerMat.reshape(descRnn.hsize, size)
				wtype = "w"

			else:
				linLayerMat = linLayerMat.reshape(descRnn.hsize, descRnn.hsize)
				wtype = "r"

			layerparams["%s%s" % (wtype, typ)] = linLayerMat
			layerparams["b%s%s" % (wtype, typ)] = linLayerBias.reshape(linLayerBias.shape[0])

		params[layer] = layerparams

	return params


def getRNNParam(descRnn, layer, descData, descW, linLayer):
	linLayerMatDesc = libmiopen.miopenCreateTensorDescriptor()
	size = libmiopen.miopenGetRNNLayerParamSize(context, descRnn.desc, layer, descData.desc, linLayer)
	linLayerMat = Driver.empty(queue, (size // np.float32(0).itemsize, ), dtype=np.float32, allocator=memPool)

	libmiopen.miopenGetRNNLayerParam(context, descRnn.desc, layer, descData.desc,
									 descW.desc, descW.ptr, linLayer, linLayerMatDesc, linLayerMat.int_ptr)

	_, dims, _ = libmiopen.miopenGetTensorDescriptor(linLayerMatDesc)
	libmiopen.miopenDestroyTensorDescriptor(linLayerMatDesc)

	linLayerMat = linLayerMat.reshape(*dims)

	linLayerBiasDesc = libmiopen.miopenCreateTensorDescriptor()
	size = libmiopen.miopenGetRNNLayerBiasSize(context, descRnn.desc, layer, linLayer)
	linLayerBias = Driver.empty(queue, (size // np.float32(0).itemsize, ), dtype=np.float32, allocator=memPool)

	libmiopen.miopenGetRNNLayerBias(context, descRnn.desc, layer, descData.desc,
									descW.desc, descW.ptr, linLayer, linLayerBiasDesc, linLayerBias.int_ptr)

	_, dims, _ = libmiopen.miopenGetTensorDescriptor(linLayerBiasDesc)
	libmiopen.miopenDestroyTensorDescriptor(linLayerBiasDesc)

	linLayerBias = linLayerBias.reshape(*dims)

	return linLayerMat, linLayerBias


def createRnn(insize, hsize, layers=1, mode=RNNMode.lstm, direction=DirectionMode.uni):
	descRnn = createDescribedRnn(insize, hsize, layers=layers, mode=mode, direction=direction)

	w = createRnnParams(descRnn, insize)
	_, params = acquireRnnParams(descRnn, w)

	return descRnn, w, params


def destroyRnn(descRnn):
	destroyDescribedRnn(descRnn)


def updateRnnParams(descRnn, w, params):
	descData = createDescribedNdTensor(dims=(1, descRnn.insize), strides=(descRnn.insize, 1), tensor=None)
	descW = createDescribedNdTensor(None, None, w)

	if descRnn.mode == RNNMode.relu or descRnn.mode == RNNMode.tanh:
		updateNativeRnnParams(descRnn, descData, descW, params)
	elif descRnn.mode == RNNMode.lstm:
		updateLSTMParams(descRnn, descData, descW, params)
	elif descRnn.mode == RNNMode.gru:
		updateGRUParams(descRnn, descData, descW, params)
	else:
		raise NotImplementedError()

	destroyDescribedTensors(descData, descW)


def updateNativeRnnParams(descRnn, descData, descW, params):
	for layer, subparams in params.items():
		for name, param in subparams.items():
			if name[0] == "w":
				setRnnParam(descRnn, layer, descData, descW, 0, param, subparams["bwi"])
			elif name[0] == "r":
				setRnnParam(descRnn, layer, descData, descW, 1, param, subparams["bri"])


def updateLSTMParams(descRnn, descData, descW, params):
	for layer, subparams in params.items():
		for name, param in subparams.items():
			if name[0] == "w":
				linLayer = 0
			elif name[0] == "r":
				linLayer = 4
			else:
				continue

			if name[1] == "f":
				linLayer += 1
			elif name[1] == "o":
				linLayer += 2
			elif name[1] == "c":
				linLayer += 3

			setRnnParam(descRnn, layer, descData, descW, linLayer, param, subparams["b%s" % name])


def updateGRUParams(descRnn, descData, descW, params):
	for layer, subparams in params.items():
		for name, param in subparams.items():
			if name[0] == "w":
				linLayer = 0
			elif name[0] == "r":
				linLayer = 3
			else:
				continue

			if name[1] == "r":
				linLayer += 1
			elif name[1] == "h":
				linLayer += 2

			setRnnParam(descRnn, layer, descData, descW, linLayer, param, subparams["b%s" % name])


def setRnnParam(descRnn, layer, descData, descW, linLayer, linLayerMat, linLayerBias):
	descLinLayerMat = createDescribedNdTensor(None, None, linLayerMat)
	libmiopen.miopenSetRNNLayerParam(context, descRnn.desc, layer, descData.desc,
									 descW.desc, descW.ptr, linLayer, descLinLayerMat.desc, descLinLayerMat.ptr)

	descLinLayerBias = createDescribedNdTensor(None, None, linLayerBias)
	libmiopen.miopenSetRNNLayerBias(context, descRnn.desc, layer, descData.desc,
									descW.desc, descW.ptr, linLayer, descLinLayerBias.desc, descLinLayerBias.ptr)

	destroyDescribedTensors(descLinLayerMat, descLinLayerBias)


def forwardRnn(data, w, descRnn, inithidden=None, initcells=None, test=False):
	assert data.ndim == 3 and data.dtype == np.float32 and descRnn.insize == data.shape[2]
	assert w.ndim == 1 and w.dtype == np.float32

	seqlen, batchsize, _ = data.shape

	if descRnn.dir == DirectionMode.uni:
		hsize = descRnn.hsize
		dims, strides = (descRnn.layers, batchsize, hsize), (batchsize * hsize, hsize, 1)
	else:
		hsize = 2 * descRnn.hsize
		dims, strides = (2 * descRnn.layers, batchsize, descRnn.hsize), (batchsize * descRnn.hsize, descRnn.hsize, 1)

	if inithidden is not None:
		assert inithidden.dtype == np.float32 and inithidden.shape == dims
	else:
		inithidden = Driver.zeros(queue, dims, dtype=np.float32, allocator=memPool)

	if descRnn.mode == RNNMode.lstm:
		if initcells is not None:
			assert initcells.dtype == np.float32 and initcells.shape == dims
		else:
			initcells = Driver.zeros(queue, dims, dtype=np.float32, allocator=memPool)

	descHx = createDescribedNdTensor(dims, strides, inithidden)
	descCx = createDescribedNdTensor(dims, strides, initcells)

	descHy = createDescribedNdTensor(None, None, Driver.empty(queue, dims, dtype=np.float32, allocator=memPool))
	descCy = createDescribedNdTensor(None, None, Driver.empty(queue, dims, dtype=np.float32, allocator=memPool))

	outdata = Driver.empty(queue, data.shape[:2] + (hsize, ), dtype=np.float32, allocator=memPool)

	descDatas = []
	descOutDatas = []

	for d in range(data.shape[0]):
		descDatas.append(createDescribedNdTensor(None, None, data[0]))
		descOutDatas.append(createDescribedNdTensor(None, None, outdata[0]))

	indescs, outdescs = [d.desc for d in descDatas], [d.desc for d in descOutDatas]
	descW = createDescribedNdTensor(None, None, w)

	reserveSize = libmiopen.miopenGetRNNTrainingReserveSize(context, descRnn.desc, seqlen, indescs)
	reserveSpace = Driver.zeros(queue, (reserveSize // np.float32(0).itemsize, ), dtype=np.float32, allocator=memPool)

	workspaceSize = libmiopen.miopenGetRNNWorkspaceSize(context, descRnn.desc, seqlen, indescs)
	workspace = Driver.empty(queue, (workspaceSize, ), dtype=np.uint8, allocator=memPool)

	tup = outdata
	if not test:
		tup = (outdata, (workspace, reserveSpace))
		libmiopen.miopenRNNForwardTraining(context, descRnn.desc, seqlen, indescs, data.int_ptr, descHx.desc,
										   descHx.ptr, descCx.desc, descCx.ptr, descW.desc, descW.ptr, outdescs,
										   outdata.int_ptr, descHy.desc, descHy.ptr, descCy.desc, descCy.ptr,
										   workspace.int_ptr, workspaceSize, reserveSpace.int_ptr,
										   reserveSize)
	else:
		libmiopen.miopenRNNForwardInference(context, descRnn.desc, seqlen, indescs, data.int_ptr, descHx.desc,
											descHx.ptr, descCx.desc, descCx.ptr, descW.desc, descW.ptr, outdescs,
											outdata.int_ptr, descHy.desc, descHy.ptr, descCy.desc, descCy.ptr,
											workspace.int_ptr, workspaceSize)

	destroyDescribedTensors(*descDatas, *descOutDatas, descHx, descCx, descHy, descCy, descW)
	return tup


def backwardDataRnn(grad, outdata, w, trainReserve, descRnn, inithidden=None, initcells=None):
	assert grad.ndim == 3 and grad.dtype == np.float32
	assert outdata.shape == grad.shape and outdata.dtype == grad.dtype
	_, batchsize, _ = grad.shape

	useHidden = True if inithidden is not None else False
	useCells = True if initcells is not None else False

	seqlen = outdata.shape[0]
	assert w.ndim == 1 and w.dtype == np.float32

	if descRnn.dir == DirectionMode.uni:
		assert grad.shape[-1] == descRnn.hsize
		dims, strides = (descRnn.layers, batchsize, descRnn.hsize), (batchsize * descRnn.hsize, descRnn.hsize, 1)
	else:
		assert grad.shape[-1] == 2 * descRnn.hsize
		dims, strides = (2 * descRnn.layers, batchsize, descRnn.hsize), (batchsize * descRnn.hsize, descRnn.hsize, 1)

	if inithidden is not None:
		assert inithidden.dtype == np.float32 and inithidden.shape == dims
	else:
		inithidden = Driver.zeros(queue, dims, dtype=np.float32, allocator=memPool)

	if descRnn.mode == RNNMode.lstm:
		if initcells is not None:
			assert initcells.dtype == np.float32 and initcells.shape == dims
		else:
			initcells = Driver.zeros(queue, dims, dtype=np.float32, allocator=memPool)

	descHx = createDescribedNdTensor(dims, strides, inithidden)
	descCx = createDescribedNdTensor(dims, strides, initcells)

	descDHx = createDescribedNdTensor(None, None, Driver.empty(queue, dims, dtype=np.float32, allocator=memPool))
	descDCx = createDescribedNdTensor(None, None, Driver.empty(queue, dims, dtype=np.float32, allocator=memPool))

	descDHy = createDescribedNdTensor(None, None, Driver.zeros(queue, dims, dtype=np.float32, allocator=memPool))
	descDCy = createDescribedNdTensor(None, None, Driver.zeros(queue, dims, dtype=np.float32, allocator=memPool))

	ingrad = Driver.zeros(queue, outdata.shape[:2] + (descRnn.insize, ), dtype=np.float32, allocator=memPool)

	descInGrads, descGrads, descOutDatas = [], [], []

	for d in range(seqlen):
		descInGrads.append(createDescribedNdTensor(None, None, ingrad[0]))
		descGrads.append(createDescribedNdTensor(None, None, grad[0]))
		descOutDatas.append(createDescribedNdTensor(None, None, outdata[0]))

	ingraddescs, graddescs = [d.desc for d in descInGrads], [d.desc for d in descGrads]
	outdatadescs = [d.desc for d in descOutDatas]
	descW = createDescribedNdTensor(None, None, w)

	workspace, reserveSpace = trainReserve

	libmiopen.miopenRNNBackwardData(context, descRnn.desc, seqlen, outdatadescs, outdata.int_ptr, graddescs,
									grad.int_ptr, descDHy.desc, descDHy.ptr, descDCy.desc, descDCy.ptr,
									descW.desc, descW.ptr, descHx.desc, descHx.ptr, descCx.desc, descCx.ptr,
									ingraddescs, ingrad.int_ptr, descDHx.desc, descDHx.ptr,
									descDCx.desc, descDCx.ptr, workspace.int_ptr, workspace.nbytes,
									reserveSpace.int_ptr, reserveSpace.nbytes)

	destroyDescribedTensors(*descInGrads, *descGrads, *descOutDatas, descHx, descCx, descDHx, descDCx, descW)

	tup = (ingrad, trainReserve)
	if useHidden: tup = tup + (descDHx.tensor, )
	if useCells: tup = tup + (descDCx.tensor, )

	return tup


def backwardParamsRnn(data, outdata, w, trainReserve, descRnn, inithidden=None):
	assert data.ndim == 3 and data.dtype == np.float32 and descRnn.insize == data.shape[2]
	assert outdata.ndim == 3 and outdata.dtype == data.dtype
	assert w.ndim == 1 and w.dtype == np.float32

	seqlen, batchsize, _ = data.shape

	if descRnn.dir == DirectionMode.uni:
		assert outdata.shape[2] == descRnn.hsize
		dims, strides = (descRnn.layers, batchsize, descRnn.hsize), (batchsize * descRnn.hsize, descRnn.hsize, 1)
	else:
		assert outdata.shape[2] == 2 * descRnn.hsize
		dims, strides = (2 * descRnn.layers, batchsize, descRnn.hsize), (batchsize * descRnn.hsize, descRnn.hsize, 1)

	if inithidden is not None:
		assert inithidden.dtype == np.float32 and inithidden.shape == dims
	else:
		inithidden = Driver.zeros(queue, dims, dtype=np.float32, allocator=memPool)

	descHx = createDescribedNdTensor(dims, strides, inithidden)

	descDatas = []
	descOutDatas = []

	for d in range(data.shape[0]):
		descDatas.append(createDescribedNdTensor(None, None, data[0]))
		descOutDatas.append(createDescribedNdTensor(None, None, outdata[0]))

	indescs, outdescs = [d.desc for d in descDatas], [d.desc for d in descOutDatas]

	dw = Driver.zeros(queue, w.shape, dtype=np.float32, allocator=memPool)
	descDw = createDescribedNdTensor(None, None, dw)

	workspace, reserveSpace = trainReserve

	libmiopen.miopenRNNBackwardWeights(context, descRnn.desc, seqlen, indescs, data.int_ptr,
									   descHx.desc, descHx.ptr, outdescs, outdata.int_ptr,
									   descDw.desc, descDw.ptr, workspace.int_ptr, workspace.nbytes,
									   reserveSpace.int_ptr, reserveSpace.nbytes)

	destroyDescribedTensors(*descDatas, *descOutDatas, descHx, descDw)

	return dw


def randomWInit(descRnn, w, params):
	for layer in params.values():
		for paramName, param in layer.items():
			param.set(np.random.randn(*param.shape).astype(np.float32))

	updateRnnParams(descRnn, w, params)


def unittest():
	reluTest()
	tanhTest()
	lstmTest()
	gruTest()


def reluTest():
	seqlen, batchsize, insize, hsize = 4, 3, 4, 5

	descRnn, w, params = createRnn(insize, hsize, mode=RNNMode.relu)
	randomWInit(descRnn, w, params)

	data = Driver.to_device(queue, np.random.randn(seqlen, batchsize, insize).astype(np.float32))
	inithidden = Driver.to_device(queue, np.random.randn(1, batchsize, hsize).astype(np.float32))

	outdata, trainReserve = forwardRnn(data, w, descRnn, inithidden=inithidden)

	hostData = data.get()
	hostOutData = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)

	hostOutData[0] = inithidden.get()
	for d in range(seqlen):
		res = np.dot(hostData[d], params[0]["wi"].get().T) + np.dot(hostOutData[d], params[0]["ri"].get().T) + \
			  params[0]["bwi"].get() + params[0]["bri"].get()
		hostOutData[d + 1] = (res > 0.0) * res

	extHostOutData = hostOutData
	hostOutData = hostOutData[1:]
	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))
	ingrad, trainReserve, dhx = backwardDataRnn(grad, outdata, w, trainReserve, descRnn, inithidden=inithidden)

	hostGrad = grad.get()

	hostAccGrad = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)
	hostInGrad = np.zeros((seqlen, batchsize, insize), dtype=np.float32)

	for d in range(seqlen):
		acc = (hostGrad[seqlen - d - 1] + np.dot(hostAccGrad[seqlen - d], params[0]["ri"].get())) * (
					hostOutData[seqlen - d - 1] > 0)

		hostAccGrad[seqlen - d - 1] = acc
		hostInGrad[seqlen - d - 1] = np.dot(acc, params[0]["wi"].get())

	assert np.allclose(hostInGrad, ingrad.get())

	dw = backwardParamsRnn(data, outdata, w, trainReserve, descRnn, inithidden=inithidden)
	dw, dwparams = acquireRnnParams(descRnn, dw)

	hostRiGrad = np.zeros(params[0]["ri"].shape, dtype=np.float32)
	hostWiGrad = np.zeros(params[0]["wi"].shape, dtype=np.float32)
	hostBriGrad = np.zeros(params[0]["bri"].shape, dtype=np.float32)
	hostBwiGrad = np.zeros(params[0]["bwi"].shape, dtype=np.float32)

	for d in range(seqlen):
		hostRiGrad += np.dot(hostAccGrad[seqlen - d - 1].T, extHostOutData[seqlen - d - 1])
		hostWiGrad += np.dot(hostAccGrad[seqlen - d - 1].T, hostData[seqlen - d - 1])
		hostBriGrad += np.sum(hostAccGrad[seqlen - d - 1], axis=0)
		hostBwiGrad += np.sum(hostAccGrad[seqlen - d - 1], axis=0)

	assert np.allclose(hostRiGrad, dwparams[0]["ri"].get())
	assert np.allclose(hostWiGrad, dwparams[0]["wi"].get())
	assert np.allclose(hostBriGrad, dwparams[0]["bri"].get())
	assert np.allclose(hostBwiGrad, dwparams[0]["bwi"].get())

	hostDhx = np.dot(hostAccGrad[0], params[0]["ri"].get())
	assert np.allclose(hostDhx, dhx.get())

	destroyRnn(descRnn)


def tanhTest():
	seqlen, batchsize, insize, hsize = 3, 3, 3, 2

	descRnn, w, params = createRnn(insize, hsize, mode=RNNMode.tanh, direction=DirectionMode.bi)
	randomWInit(descRnn, w, params)

	data = Driver.to_device(queue, np.random.randn(seqlen, batchsize, insize).astype(np.float32))
	outdata, trainReserve = forwardRnn(data, w, descRnn)

	hostData = data.get()
	hostOutData = np.zeros((seqlen + 2, batchsize, 2 * hsize), dtype=np.float32)

	for d in range(seqlen):
		res = np.dot(hostData[d], params[0]["wi"].get().T) + \
			  np.dot(hostOutData[d, :, :hsize], params[0]["ri"].get().T) + params[0]["bwi"].get()+params[0]["bri"].get()
		hostOutData[d + 1, :, :hsize] = np.tanh(res)

		res = np.dot(hostData[seqlen - d-1], params[1]["wi"].get().T) + \
		np.dot(hostOutData[seqlen+1-d,:,hsize:],params[1]["ri"].get().T)+params[1]["bwi"].get() + params[1]["bri"].get()
		hostOutData[seqlen - d, :, hsize:] = np.tanh(res)

	extHostOutData = hostOutData
	hostOutData = hostOutData[1:seqlen+1]
	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))

	ingrad, trainReserve = backwardDataRnn(grad, outdata, w, trainReserve, descRnn)

	hostGrad = grad.get()

	hostAccGrad = np.zeros((seqlen + 2, batchsize, 2 * hsize), dtype=np.float32)
	hostInGrad = np.zeros((seqlen, batchsize, insize), dtype=np.float32)

	for d in range(seqlen):
		acc = (hostGrad[seqlen-d-1, :, :hsize] + np.dot(hostAccGrad[seqlen+1-d, :, :hsize], params[0]["ri"].get())) * \
			  (1.0 - hostOutData[seqlen - d - 1, :, :hsize]**2)

		hostAccGrad[seqlen - d, :, :hsize] = acc
		hostInGrad[seqlen - d - 1] += np.dot(acc, params[0]["wi"].get())

		acc = (hostGrad[d, :, hsize:] + np.dot(hostAccGrad[d, :, hsize:], params[1]["ri"].get())) * \
			  (1.0 - hostOutData[d, :, hsize:]**2)

		hostAccGrad[d+1, :, hsize:] = acc
		hostInGrad[d] += np.dot(acc, params[1]["wi"].get())

	assert np.allclose(hostInGrad, ingrad.get())

	dw = backwardParamsRnn(data, outdata, w, trainReserve, descRnn)
	dw, dwparams = acquireRnnParams(descRnn, dw)

	hostRi0Grad = np.zeros(params[0]["ri"].shape, dtype=np.float32)
	hostRi1Grad = np.zeros(params[1]["ri"].shape, dtype=np.float32)
	hostWi0Grad = np.zeros(params[0]["wi"].shape, dtype=np.float32)
	hostWi1Grad = np.zeros(params[1]["wi"].shape, dtype=np.float32)

	hostBri0Grad = np.zeros(params[0]["bri"].shape, dtype=np.float32)
	hostBri1Grad = np.zeros(params[1]["bri"].shape, dtype=np.float32)
	hostBwi0Grad = np.zeros(params[0]["bwi"].shape, dtype=np.float32)
	hostBwi1Grad = np.zeros(params[1]["bwi"].shape, dtype=np.float32)

	for d in range(seqlen):
		hostRi0Grad += np.dot(hostAccGrad[seqlen - d+1, :, :hsize].T, extHostOutData[seqlen - d, :, :hsize])
		hostWi0Grad += np.dot(hostAccGrad[seqlen - d, :, :hsize].T, hostData[seqlen - d - 1])
		hostRi1Grad += np.dot(hostAccGrad[d, :, hsize:].T, extHostOutData[d+1, :, hsize:])
		hostWi1Grad += np.dot(hostAccGrad[d+1, :, hsize:].T, hostData[d])

		hostBri0Grad += np.sum(hostAccGrad[seqlen-d, :, :hsize], axis=0)
		hostBwi0Grad += np.sum(hostAccGrad[seqlen-d, :, :hsize], axis=0)
		hostBri1Grad += np.sum(hostAccGrad[d+1, :, hsize:], axis=0)
		hostBwi1Grad += np.sum(hostAccGrad[d+1, :, hsize:], axis=0)

	assert np.allclose(hostRi0Grad, dwparams[0]["ri"].get())
	assert np.allclose(hostWi0Grad, dwparams[0]["wi"].get())
	assert np.allclose(hostRi1Grad, dwparams[1]["ri"].get())
	assert np.allclose(hostWi1Grad, dwparams[1]["wi"].get())

	assert np.allclose(hostBri0Grad, dwparams[0]["bri"].get())
	assert np.allclose(hostBwi0Grad, dwparams[0]["bwi"].get())
	assert np.allclose(hostBri1Grad, dwparams[1]["bri"].get())
	assert np.allclose(hostBwi1Grad, dwparams[1]["bwi"].get())

	destroyRnn(descRnn)


def lstmTest():
	seqlen, batchsize, insize, hsize = 4, 2, 4, 2

	descRnn, w, params = createRnn(insize, hsize, mode=RNNMode.lstm)
	randomWInit(descRnn, w, params)
	params = params[0]

	data = Driver.to_device(queue, np.random.randn(seqlen, batchsize, insize).astype(np.float32))
	inithidden = Driver.to_device(queue, np.random.randn(1, batchsize, hsize).astype(np.float32))
	initcells = Driver.to_device(queue, np.ones((1, batchsize, hsize), dtype=np.float32))

	outdata, trainReserve = forwardRnn(data, w, descRnn, inithidden=inithidden, initcells=initcells)

	hostData = data.get()

	hostOutData = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)
	hostCells = np.empty((seqlen + 1, batchsize, hsize), dtype=np.float32)

	hostOutData[0] = inithidden.get()
	hostCells[0] = initcells.get()

	hostStates = np.zeros((seqlen + 2, batchsize, hsize * 4), dtype=np.float32)
	hostW = np.empty((insize + hsize, 4 * hsize), dtype=np.float32)
	hostBias = np.empty((4 * hsize, ), dtype=np.float32)

	hostW[:insize, :hsize] = params["wc"].get().T
	hostW[:insize, hsize:2 * hsize] = params["wi"].get().T
	hostW[:insize, 2 * hsize:3 * hsize] = params["wf"].get().T
	hostW[:insize, 3 * hsize:] = params["wo"].get().T

	hostW[insize:, :hsize] = params["rc"].get().T
	hostW[insize:, hsize:2 * hsize] = params["ri"].get().T
	hostW[insize:, 2 * hsize:3 * hsize] = params["rf"].get().T
	hostW[insize:, 3 * hsize:] = params["ro"].get().T

	hostBias[:hsize] = params["bwc"].get() + params["brc"].get()
	hostBias[hsize:2 * hsize] = params["bwi"].get() + params["bri"].get()
	hostBias[2 * hsize: 3 * hsize] = params["bwf"].get() + params["brf"].get()
	hostBias[3 * hsize:] = params["bwo"].get() + params["bro"].get()

	def lstmAct(dat, hsz):
		dat[:, :hsz] = np.tanh(dat[:, :hsz])
		dat[:, hsz:] = 1.0 / (np.exp(-dat[:, hsz:]) + 1.0)
		return dat

	for d in range(seqlen):
		inp = np.hstack((hostData[d], hostOutData[d]))
		outp = lstmAct(np.dot(inp, hostW) + hostBias, hsize)
		hostStates[d+1] = outp

		ct = outp[:,2*hsize:3*hsize] * hostCells[d] + outp[:,hsize:2*hsize] * outp[:,:hsize]

		hostCells[d+1] = ct
		hostOutData[d+1] = outp[:, 3*hsize:] * np.tanh(ct)

	extHostOutData = hostOutData
	hostOutData = hostOutData[1:]
	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))

	ingrad, trainReserve, dhx, dcx = backwardDataRnn(grad, outdata, w, trainReserve, descRnn,
													 inithidden=inithidden, initcells=initcells)
	dw = backwardParamsRnn(data, outdata, w, trainReserve, descRnn, inithidden=inithidden)
	dw, dwparams = acquireRnnParams(descRnn, dw)

	dwparams = dwparams[0]
	hostDw = np.zeros(hostW.shape, dtype=np.float32)
	hostDb = np.zeros(hostBias.shape, dtype=np.float32)

	hostGrad = grad.get()

	hostAccCellsGrad = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)
	hostAccHiddenGrad = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)
	hostInGrad = np.zeros((seqlen, batchsize, insize), dtype=np.float32)

	def lstmActBwd(gr, dat, hsz):
		gr[:, :hsz] = gr[:, :hsz] * (1.0 - dat[:, :hsz]**2)
		gr[:, hsz:] = gr[:, hsz:] * dat[:, hsz:] * (1.0 - dat[:, hsz:])
		return gr

	for d in range(seqlen):
		dh = hostGrad[seqlen - 1 - d] + hostAccHiddenGrad[seqlen - d]
		dc = dh * hostStates[seqlen - d, :, 3 * hsize:] * (1 - np.tanh(hostCells[seqlen - d])**2) + \
			 hostAccCellsGrad[seqlen - d] * hostStates[seqlen + 1 - d, :, 2 * hsize:3 * hsize]

		layergr = np.empty((batchsize, 4 * hsize), dtype=np.float32)
		layergr[:, :hsize] = dc * hostStates[seqlen - d, :, hsize:2 * hsize]
		layergr[:, hsize:2 * hsize] = dc * hostStates[seqlen - d, :, :hsize]
		layergr[:, 2 * hsize:3 * hsize] = dc * hostCells[seqlen - 1 - d]
		layergr[:, 3 * hsize:] = dh * np.tanh(hostCells[seqlen - d])

		layergr = lstmActBwd(layergr, hostStates[seqlen - d], hsize)
		ingr = np.dot(layergr, hostW.T)

		indata = np.hstack((hostData[seqlen - 1 - d], extHostOutData[seqlen - 1 - d]))
		hostDw += np.dot(indata.T, layergr)
		hostDb += np.sum(layergr, axis=0)

		hostAccHiddenGrad[seqlen - 1 - d] = ingr[:, insize:]
		hostAccCellsGrad[seqlen - 1 - d] = dc
		hostInGrad[seqlen - 1 - d] = ingr[:, :insize]

	assert np.allclose(hostInGrad, ingrad.get())

	assert np.allclose(hostDw[:insize, :hsize], dwparams["wc"].get().T)
	assert np.allclose(hostDw[:insize, hsize:2 * hsize], dwparams["wi"].get().T)
	assert np.allclose(hostDw[:insize, 2 * hsize:3 * hsize], dwparams["wf"].get().T)
	assert np.allclose(hostDw[:insize, 3 * hsize:], dwparams["wo"].get().T)

	assert np.allclose(hostDw[insize:, :hsize], dwparams["rc"].get().T)
	assert np.allclose(hostDw[insize:, hsize:2 * hsize], dwparams["ri"].get().T)
	assert np.allclose(hostDw[insize:, 2 * hsize:3 * hsize], dwparams["rf"].get().T)
	assert np.allclose(hostDw[insize:, 3 * hsize:], dwparams["ro"].get().T)

	assert np.allclose(2.0 * hostDb[:hsize], dwparams["bwc"].get() + dwparams["brc"].get())
	assert np.allclose(2.0 * hostDb[hsize:2 * hsize], dwparams["bwi"].get() + dwparams["bri"].get())
	assert np.allclose(2.0 * hostDb[2 * hsize: 3 * hsize], dwparams["bwf"].get() + dwparams["brf"].get())
	assert np.allclose(2.0 * hostDb[3 * hsize:], dwparams["bwo"].get() + dwparams["bro"].get())

	destroyRnn(descRnn)


def gruTest():
	seqlen, batchsize, insize, hsize = 3, 3, 4, 2

	descRnn, w, params = createRnn(insize, hsize, mode=RNNMode.gru)
	randomWInit(descRnn, w, params)
	params = params[0]

	data = Driver.to_device(queue, np.random.randn(seqlen, batchsize, insize).astype(np.float32))
	inithidden = Driver.to_device(queue, np.random.randn(1, batchsize, hsize).astype(np.float32))

	outdata, trainReserve = forwardRnn(data, w, descRnn, inithidden=inithidden)

	hostData = data.get()
	hostOutData = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)

	hostOutData[0] = inithidden.get()

	hostStates = np.zeros((seqlen + 1, batchsize, hsize * 4), dtype=np.float32)
	hostHts = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)
	hostW = np.zeros((insize + hsize, 4 * hsize), dtype=np.float32)
	hostBias = np.empty((4 * hsize, ), dtype=np.float32)

	hostW[:insize, hsize:2 * hsize] = params["wh"].get().T
	hostW[:insize, 2 * hsize:3 * hsize] = params["wr"].get().T
	hostW[:insize, 3 * hsize:] = params["wi"].get().T

	hostW[insize:, :hsize] = params["rh"].get().T
	hostW[insize:, 2 * hsize:3 * hsize] = params["rr"].get().T
	hostW[insize:, 3 * hsize:] = params["ri"].get().T

	hostBias[:hsize] = params["brh"].get()
	hostBias[hsize:2 * hsize] = params["bwh"].get()
	hostBias[2 * hsize: 3 * hsize] = params["bwr"].get() + params["brr"].get()
	hostBias[3 * hsize:] = params["bwi"].get() + params["bri"].get()

	def gruAct(dat, hsz):
		dat[:, 2 * hsz:] = 1.0 / (np.exp(-dat[:, 2 * hsz:]) + 1.0)
		return dat

	for d in range(seqlen):
		inp = np.hstack((hostData[d], hostOutData[d]))
		outp = gruAct(np.dot(inp, hostW) + hostBias, hsize)
		hostStates[d + 1] = outp

		ht = np.tanh(outp[:, hsize:2 * hsize] + outp[:, 2 * hsize: 3 * hsize] * outp[:, :hsize])
		it = outp[:, 3 * hsize:]
		hostOutData[d + 1] = (1.0 - it) * ht + it * hostOutData[d]
		hostHts[d + 1] = ht

	extHostOutData = hostOutData
	hostOutData = hostOutData[1:]
	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))

	ingrad, trainReserve, dhx = backwardDataRnn(grad, outdata, w, trainReserve, descRnn, inithidden=inithidden)
	dw = backwardParamsRnn(data, outdata, w, trainReserve, descRnn, inithidden=inithidden)
	dw, dwparams = acquireRnnParams(descRnn, dw)

	dwparams = dwparams[0]
	hostDw = np.zeros(hostW.shape, dtype=np.float32)
	hostDb = np.zeros(hostBias.shape, dtype=np.float32)

	hostGrad = grad.get()

	hostAccGrad = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)
	hostInGrad = np.zeros((seqlen, batchsize, insize), dtype=np.float32)

	def gruActBwd(gr, dat, hsz):
		gr[:, 2 * hsz:] = gr[:, 2 * hsz:] * dat[:, 2 * hsz:] * (1.0 - dat[:, 2 * hsz:])
		return gr

	for d in range(seqlen):
		dh = hostGrad[seqlen-1 - d] + hostAccGrad[seqlen - d]
		dht = (1 - hostStates[seqlen - d, :, 3 * hsize:]) * dh

		layergr = np.empty((batchsize, 4 * hsize), dtype=np.float32)
		layergr[:, :hsize] = dht * (1.0 - hostHts[seqlen - d]**2) * hostStates[seqlen - d, :, 2 * hsize:3 * hsize]
		layergr[:, hsize:2 * hsize] = dht * (1.0 - hostHts[seqlen - d]**2)
		layergr[:, 2 * hsize:3 * hsize] = dht * (1.0 - hostHts[seqlen - d]**2) * hostStates[seqlen - d, :, :hsize]
		layergr[:, 3 * hsize:] = dh * (extHostOutData[seqlen-1 - d] - hostHts[seqlen - d])

		layergr = gruActBwd(layergr, hostStates[seqlen - d], hsize)
		ingr = np.dot(layergr, hostW.T)

		indata = np.hstack((hostData[seqlen - 1 - d], extHostOutData[seqlen - 1 - d]))
		hostDw += np.dot(indata.T, layergr)
		hostDb += np.sum(layergr, axis=0)

		hostAccGrad[seqlen - 1 - d] = dh * hostStates[seqlen - d, :, 3 * hsize:] + ingr[:, insize:]
		hostInGrad[seqlen - 1 - d] = ingr[:, :insize]

	assert np.allclose(hostInGrad, ingrad.get())

	assert np.allclose(hostDw[:insize, hsize:2 * hsize], dwparams["wh"].get().T)
	assert np.allclose(hostDw[:insize, 2 * hsize:3 * hsize], dwparams["wr"].get().T)
	assert np.allclose(hostDw[:insize, 3 * hsize:], dwparams["wi"].get().T)

	assert np.allclose(hostDw[insize:, :hsize], dwparams["rh"].get().T)
	assert np.allclose(hostDw[insize:, 2 * hsize:3 * hsize], dwparams["rr"].get().T)
	assert np.allclose(hostDw[insize:, 3 * hsize:], dwparams["ri"].get().T)

	assert np.allclose(hostDb[:hsize], dwparams["brh"].get())
	assert np.allclose(hostDb[hsize:2 * hsize], dwparams["bwh"].get())
	assert np.allclose(2 * hostDb[2 * hsize: 3 * hsize], dwparams["bwr"].get() + dwparams["brr"].get())
	assert np.allclose(2 * hostDb[3 * hsize:], dwparams["bwi"].get() + dwparams["bri"].get())

	destroyRnn(descRnn)


if __name__ == "__main__":
	unittest()
