import os

import numpy as np

from PuzzleLib.Containers.Container import Container
from PuzzleLib.Containers.Sequential import Sequential
from PuzzleLib.Containers.Parallel import Parallel
from PuzzleLib.Containers.Graph import Graph

from PuzzleLib.Modules.Activation import Activation, sigmoid, tanh, relu, leakyRelu, clip
from PuzzleLib.Modules.Add import Add
from PuzzleLib.Modules.AvgPool2D import AvgPool2D
from PuzzleLib.Modules.BatchNorm import BatchNorm
from PuzzleLib.Modules.BatchNorm1D import BatchNorm1D
from PuzzleLib.Modules.BatchNorm2D import BatchNorm2D
from PuzzleLib.Modules.Concat import Concat
from PuzzleLib.Modules.Conv1D import Conv1D
from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.CrossMapLRN import CrossMapLRN
from PuzzleLib.Modules.Deconv2D import Deconv2D
from PuzzleLib.Modules.Dropout import Dropout
from PuzzleLib.Modules.Flatten import Flatten
from PuzzleLib.Modules.GroupLinear import GroupLinear
from PuzzleLib.Modules.Identity import Identity
from PuzzleLib.Modules.Linear import Linear
from PuzzleLib.Modules.MaxPool2D import MaxPool2D
from PuzzleLib.Modules.MoveAxis import MoveAxis
from PuzzleLib.Modules.MulAddConst import MulAddConst
from PuzzleLib.Modules.Pad1D import Pad1D, PadMode
from PuzzleLib.Modules.PRelu import PRelu
from PuzzleLib.Modules.Replicate import Replicate
from PuzzleLib.Modules.Reshape import Reshape
from PuzzleLib.Modules.RNN import RNN
from PuzzleLib.Modules.SoftMax import SoftMax
from PuzzleLib.Modules.Split import Split
from PuzzleLib.Modules.Sum import Sum
from PuzzleLib.Modules.SwapAxes import SwapAxes
from PuzzleLib.Modules.Upsample2D import Upsample2D

from PuzzleLib.Converter.TensorRT import Driver
from PuzzleLib.Converter.TensorRT.DataCalibrator import CalibratorError
from PuzzleLib.Converter.TensorRT.RTEngine import RTEngine, DataType, RTEngineType, genEngineName


class ConverterError(Exception):
	pass


def buildRTEngine(net, inshape, savepath, dtype, calibrator=None, workspace=1 << 22, returnEngine=True, log=True):
	outshape = net.dataShapeFrom(inshape)
	batchsize, inshape = inshape[0], inshape[1:]

	engineName = genEngineName(net.name, dtype, (batchsize, ) + inshape, outshape)
	savepath = os.path.join(savepath, engineName)

	convert(net, batchsize, inshape, dtype, calibrator, workspace, savepath, log)

	if returnEngine:
		return RTEngine(savepath, RTEngineType.puzzle, (batchsize, ) + inshape, outshape, log)


def buildRTEngineFromCaffe(model, inshape, outshape, outlayers, savepath, dtype, calibrator=None, workspace=1 << 22,
						   returnEngine=True, log=True):
	prototxt, caffemodel = model

	engineName = genEngineName(os.path.splitext(os.path.basename(caffemodel))[0], dtype, inshape, outshape)
	savepath = os.path.join(savepath, engineName)

	batchsize, inshape = inshape[0], inshape[1:]
	Driver.buildRTEngineFromCaffe(
		prototxt, caffemodel, batchsize, outlayers, dtype, calibrator, workspace, savepath, log
	)

	if returnEngine:
		return RTEngine(savepath, RTEngineType.caffe, (batchsize, ) + inshape, outshape, log)


def buildRTEngineFromOnnx(model, inshape, outshape, savepath, dtype, calibrator=None, workspace=1 << 22,
						  returnEngine=True, log=True):
	engineName = genEngineName(os.path.splitext(os.path.basename(model))[0], dtype, inshape, outshape)
	savepath = os.path.join(savepath, engineName)

	batchsize, inshape = inshape[0], inshape[1:]
	Driver.buildRTEngineFromOnnx(model, batchsize, dtype, calibrator, workspace, savepath, log)

	if returnEngine:
		return RTEngine(savepath, RTEngineType.onnx, (batchsize, ) + inshape, outshape, log)


def convert(net, batchsize, inshape, dtype, calibrator, workspace, savepath, log):
	graph = Driver.createNetwork(log)

	if dtype == DataType.float16:
		if not graph.platformHasFastFp16():
			raise ConverterError("Platform has no fast fp16 support")

		graph.setFp16Mode(True)

	elif dtype == DataType.int8:
		if not graph.platformHasFastInt8():
			raise ConverterError("Platform has no fast int8 support")

		graph.setInt8Mode(True)
		graph.setInt8Calibrator(calibrator)

	if not isinstance(inshape, list):
		inshape = [inshape]

	if calibrator is not None:
		calshape = calibrator.getDataShape()
		assert len(inshape) == 1

		if calshape != inshape[0]:
			raise CalibratorError("Calibrator data has shape %s, network has input shape %s" % (calshape, inshape[0]))

	inputs = []
	for i, shape in enumerate(inshape):
		inputs.append(graph.addInput("data_%s" % i, DataType.float32, shape))

	holder = []
	output = convertModule(net, net.name, graph, inputs, holder)

	for i, out in enumerate(output):
		out.setName("output_%s" % i)
		graph.markOutput(out)

	graph.setMaxBatchSize(batchsize)
	graph.setMaxWorkspaceSize(workspace)

	graph.buildCudaEngine(savepath)


def numpyPtr(ary):
	return ary.__array_interface__["data"][0]


def convertModule(module, fullname, graph, inputs, holder):
	if isinstance(module, Container):
		if isinstance(module, Sequential):
			return convertSequential(module, fullname, graph, inputs, holder)

		elif isinstance(module, Parallel):
			return convertParallel(module, fullname, graph, inputs, holder)

		elif isinstance(module, Graph):
			return convertGraph(module, fullname, graph, inputs, holder)

		else:
			raise NotImplementedError(module.__class__.__name__)

	else:
		if isinstance(module, Add):
			return convertAdd(fullname, graph, inputs)
		elif isinstance(module, Concat):
			return convertConcat(module, fullname, graph, inputs)

		assert len(inputs) == 1
		inp = inputs[0]

		if isinstance(module, (Conv2D, Deconv2D)):
			return convertConv2D(module, fullname, graph, inp, holder)

		elif isinstance(module, BatchNorm2D):
			return convertBatchNorm2D(module, fullname, graph, inp, holder)

		elif isinstance(module, Conv1D):
			return convertConv1D(module, fullname, graph, inp, holder)

		elif isinstance(module, BatchNorm1D):
			return convertBatchNorm1D(module, fullname, graph, inp, holder)

		elif isinstance(module, Activation):
			return convertActivation(module, fullname, graph, inp)

		elif isinstance(module, (Identity, Dropout)):
			return convertIdentity(inp)

		elif isinstance(module, Replicate):
			return convertReplicate(module, inp)

		elif isinstance(module, (MaxPool2D, AvgPool2D)):
			return convertPool2D(module, fullname, graph, inp)

		elif isinstance(module, CrossMapLRN):
			return convertCrossMapLRN(module, fullname, graph, inp)

		elif isinstance(module, Flatten):
			return convertFlatten(inp, fullname, graph)

		elif isinstance(module, Linear):
			return convertLinear(module, fullname, graph, inp, holder)

		elif isinstance(module, SoftMax):
			return convertSoftmax(fullname, graph, inp)

		elif isinstance(module, SwapAxes):
			return convertSwapAxes(module, fullname, graph, inp)

		elif isinstance(module, MoveAxis):
			return convertMoveAxis(module, fullname, graph, inp)

		elif isinstance(module, Split):
			return convertSplit(module, fullname, graph, inp)

		elif isinstance(module, Reshape):
			return convertReshape(module, fullname, graph, inp)

		elif isinstance(module, GroupLinear):
			return convertGroupLinear(module, fullname, graph, inp, holder)

		elif isinstance(module, MulAddConst):
			return convertMulAddConst(module, fullname, graph, inp, holder)

		elif isinstance(module, BatchNorm):
			return convertBatchNorm(module, fullname, graph, inp, holder)

		elif isinstance(module, Sum):
			return convertSum(module, fullname, graph, inp)

		elif isinstance(module, RNN):
			return convertRNN(module, fullname, graph, inp, holder)

		elif isinstance(module, Upsample2D):
			return convertUpsample2D(module, fullname, graph, inp)

		elif isinstance(module, PRelu):
			return convertPRelu(module, fullname, graph, inp, holder)

		elif isinstance(module, Pad1D):
			return convertPad1D(module, fullname, graph, inp)

		else:
			raise NotImplementedError(module.__class__.__name__)


def convertSequential(seq, fullname, graph, inputs, holder):
	for child in seq.graph:
		name = None if child.name is None else "%s.%s" % (fullname, child.name)
		inputs = convertModule(child, name, graph, inputs, holder)

	return inputs


def convertParallel(parallel, fullname, graph, inputs, holder):
	assert len(inputs) == len(parallel.graph)

	outputs = []
	for i, child in enumerate(parallel.graph):
		name = None if child.name is None else "%s.%s" % (fullname, child.name)
		outputs.append(convertModule(child, name, graph, [inputs[i]], holder)[0])

	return outputs


def convertNode(node, fullname, graph, inputs, nodes, holder):
	name = None if node.name is None else "%s.%s" % (fullname, node.name)
	inputs = [inputs[node.name]] if len(node.bwds) == 0 else [nodes[output.name] for output, _ in node.bwds]

	outputs = convertModule(node.module, name, graph, inputs, holder)
	assert len(outputs) == 1

	nodes[node.name] = outputs[0]


def convertGraph(hostgraph, fullname, devgraph, inputs, holder):
	assert len(inputs) == len(hostgraph.inputs)

	nodes = {}
	inputs = {node.name: inputs[i] for i, node in enumerate(hostgraph.inputs)}

	for i, inp in enumerate(hostgraph.inputs):
		inp.traverseForward(inp, convertNode, fullname, devgraph, inputs, nodes, holder)

	hostgraph.reset()
	outputs = [nodes[output.name] for output in hostgraph.outputs]

	return outputs


def convertAdd(fullname, graph, inputs):
	assert len(inputs) == 2
	output = graph.addAdd(inputs[0], inputs[1], fullname)

	return [output]


def convertConcat(module, fullname, graph, inputs):
	assert module.axis == 1
	output = graph.addConcatenation(inputs, fullname)

	return [output]


def convertConv2D(module, fullname, graph, inp, holder):
	assert module.groups == 1
	assert not isinstance(module, Deconv2D) or module.dilation == (1, 1)

	W = module.W.get().flatten()
	holder.append(W)

	b = module.b.get().flatten() if module.useBias else None
	if b is None:
		bptr, bsize = 0, 0
	else:
		bptr, bsize = numpyPtr(b), b.size
		holder.append(b)

	isDeconvolution = True if isinstance(module, Deconv2D) else False
	outmaps = module.W.shape[1] if isDeconvolution else module.W.shape[0]

	output = graph.addConvolution(
		inp, outmaps, module.W.shape[2:], numpyPtr(W), W.size, bptr, bsize, module.stride, module.pad, module.dilation,
		isDeconvolution, fullname
	)
	return [output]


def convertBatchNorm2D(module, fullname, graph, inp, holder):
	mean = module.mean.get()
	var = module.var.get()

	scale = module.scale.get()
	bias = module.bias.get()

	eps = module.epsilon

	shift = (bias - scale * mean / np.sqrt(var + eps)).flatten()
	scale = (scale / np.sqrt(var + eps)).flatten()
	power = np.ones(mean.shape, dtype=mean.dtype).flatten()

	holder.extend([shift, scale, power])
	output = graph.addScale(inp, numpyPtr(shift), numpyPtr(scale), numpyPtr(power), shift.size, fullname)

	return [output]


def convertConv1D(module, fullname, graph, inp, holder):
	shape = tuple(inp.shape)
	assert len(shape) == 2

	output = graph.addReshape(inp, (shape[0], 1, shape[1]), "%s_inshape" % fullname)
	[output] = convertConv2D(module, fullname, graph, output, holder)

	shape = output.shape
	output = graph.addReshape(output, (shape[0], shape[2]), "%s_outshape" % fullname)

	return [output]


def convertBatchNorm1D(module, fullname, graph, inp, holder):
	shape = tuple(inp.shape)
	assert len(shape) == 2

	output = graph.addReshape(inp, (shape[0], 1, shape[1]), "%s_inshape" % fullname)
	[output] = convertBatchNorm2D(module, fullname, graph, output, holder)

	output = graph.addReshape(output, shape, "%s_outshape" % fullname)
	return [output]


def convertActivation(module, fullname, graph, inp):
	actType = module.getBlueprint()["scheme"]["activation"]
	assert actType in {relu, sigmoid, tanh, leakyRelu, clip}

	typ = {
		relu: Driver.ActivationType.relu,
		leakyRelu: Driver.ActivationType.leakyRelu,
		clip: Driver.ActivationType.clip,
		sigmoid: Driver.ActivationType.sigmoid,
		tanh: Driver.ActivationType.tanh
	}[actType]

	alpha, beta = 0, 0
	if actType == leakyRelu:
		alpha = module.actArgs[0]
	elif actType == clip:
		alpha, beta = module.actArgs

	output = graph.addActivation(inp, typ, alpha, beta, fullname)
	return [output]


def convertIdentity(inp):
	return [inp]


def convertReplicate(module, inp):
	return [inp] * module.times


def convertPool2D(module, fullname, graph, inp):
	output = graph.addPooling(
		inp, True if isinstance(module, AvgPool2D) else False, module.size, module.stride, module.pad, fullname
	)

	return [output]


def convertCrossMapLRN(module, fullname, graph, inp):
	output = graph.addCrossMapLRN(inp, module.N, module.alpha, module.beta, module.K, fullname)
	return [output]


def convertFlatten(inp, fullname, graph):
	output = graph.addFlatten(inp, fullname)
	return [output]


def convertLinear(module, fullname, graph, inp, holder):
	W = module.W.get().T.flatten()
	holder.append(W)

	b = module.b.get().flatten() if module.useBias else None
	if b is None:
		bptr, bsize = 0, 0
	else:
		bptr, bsize = numpyPtr(b), b.size
		holder.append(b)

	output = graph.addLinear(inp, module.W.shape[1], numpyPtr(W), W.size, bptr, bsize, fullname)
	return [output]


def convertSoftmax(fullname, graph, inp):
	output = graph.addSoftMax(inp, fullname)
	return [output]


def convertSwapAxes(module, fullname, graph, inp):
	if module.axis1 > 0 and module.axis2 > 0:
		output = graph.addSwapAxes(inp, module.axis1 - 1, module.axis2 - 1, fullname)

	else:
		output = inp
		print(
			"Warning on module %s: ignoring swap for axes %s and %s - " % (fullname, module.axis1, module.axis2) + \
			"TensorRT does not support shuffles on batch axis. Assuming correct order of model batch shuffles ...",
			flush=True
		)

	return [output]


def convertMoveAxis(module, fullname, graph, inp):
	assert module.src > 0 and module.dst > 0

	output = graph.addMoveAxis(inp, module.src - 1, module.dst - 1, fullname)
	return [output]


def convertSplit(module, fullname, graph, inp):
	assert module.axis == 1

	output = graph.addSplit(inp, module.axis, module.sections, fullname)
	return output


def convertReshape(module, fullname, graph, inp):
	output = graph.addReshape(inp, module.shape[1:], fullname)
	return [output]


def convertGroupLinear(module, fullname, graph, inp, holder):
	assert module.inmode == "full" and module.wmode == "one"
	assert not module.transpW

	groups = inp.shape[0]

	W = module.W.get()[0].flatten()
	holder.append(W)

	b = module.b.get().flatten() if module.useBias else None
	if b is None:
		bptr, bsize = 0, 0
	else:
		b = np.tile(b, reps=groups)
		bptr, bsize = numpyPtr(b), b.size

		holder.append(b)

	output = graph.addGroupLinear(
		inp, groups, module.W.shape[1], module.W.shape[2], numpyPtr(W), W.size, bptr, b.size, fullname
	)
	return [output]


def convertMulAddConst(module, fullname, graph, inp, holder):
	c = inp.shape[0]

	shift = np.array([module.b] * c, dtype=np.float32)
	scale = np.array([module.a] * c, dtype=np.float32)
	power = np.ones((c, ), dtype=np.float32)

	holder.extend([shift, scale, power])

	output = graph.addScale(inp, numpyPtr(shift), numpyPtr(scale), numpyPtr(power), c, fullname)
	return [output]


def convertBatchNorm(module, fullname, graph, inp, holder):
	shape = tuple(inp.shape)
	assert len(shape) == 1

	output = graph.addReshape(inp, shape + (1, 1), "%s_inshape" % fullname)
	[output] = convertBatchNorm2D(module, fullname, graph, output, holder)

	output = graph.addReshape(output, shape, "%s_outshape" % fullname)
	return [output]


def convertSum(module, fullname, graph, inp):
	output = graph.addSum(inp, module.axis, fullname)
	return [output]


def convertRNN(module, fullname, graph, inp, holder):
	assert module.getSequences
	seqlen = inp.shape[0]

	mode = {
		"relu": Driver.RNNMode.relu,
		"tanh": Driver.RNNMode.tanh,
		"lstm": Driver.RNNMode.lstm,
		"gru": Driver.RNNMode.gru
	}[module.mode]

	direction = {
		"uni": Driver.RNNDirection.uni,
		"bi": Driver.RNNDirection.bi
	}[module.direction]

	inputMode = Driver.RNNInputMode.linear

	params = [
		{name: param.get() for name, param in module.params[key].items()} for key in sorted(module.params.keys())
	]
	holder.append(params)

	keys = {
		"relu": ["wi", "ri"],
		"tanh": ["wi", "ri"],
		"lstm": ["wf", "wi", "wc", "wo", "rf", "ri", "rc", "ro"],
		"gru": ["wi", "wr", "wh", "ri", "rr", "rh"]
	}[module.mode]

	Wdata = [numpyPtr(par[key]) for par in params for key in keys]
	Wlen = [par[key].size for par in params for key in keys]

	biasdata = [numpyPtr(par["b%s" % key]) for par in params for key in keys]
	biaslen = [par["b%s" % key].size for par in params for key in keys]

	output = graph.addRNN(
		inp, module.layers, module.hsize, seqlen, mode, direction, inputMode, Wdata, Wlen, biasdata, biaslen, fullname
	)

	return [output]


def convertUpsample2D(module, fullname, graph, inp):
	assert module.mode == "nearest"
	output = graph.addUpsample(inp, module.scale, fullname)

	return [output]


def convertPRelu(module, fullname, graph, inp, holder):
	assert not module.sharedMaps

	slopes = module.slopes.get()
	holder.append(slopes)

	output = graph.addPRelu(inp, numpyPtr(slopes), slopes.size, fullname)
	return [output]


def convertPad1D(module, fullname, graph, inp):
	assert module.mode == PadMode.reflect

	output = graph.addReflectPad(inp, module.pad, fullname)
	return [output]
