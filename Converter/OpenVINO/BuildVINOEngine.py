import os

import numpy as np

from PuzzleLib.Containers.Container import Container
from PuzzleLib.Containers.Sequential import Sequential
from PuzzleLib.Containers.Parallel import Parallel
from PuzzleLib.Containers.Graph import Graph

from PuzzleLib.Modules.Activation import Activation, relu, leakyRelu, sigmoid
from PuzzleLib.Modules.Add import Add
from PuzzleLib.Modules.AvgPool2D import AvgPool2D
from PuzzleLib.Modules.BatchNorm import BatchNorm
from PuzzleLib.Modules.BatchNorm2D import BatchNorm2D
from PuzzleLib.Modules.Concat import Concat
from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.Dropout import Dropout
from PuzzleLib.Modules.Flatten import Flatten
from PuzzleLib.Modules.Identity import Identity
from PuzzleLib.Modules.Linear import Linear
from PuzzleLib.Modules.MaxPool2D import MaxPool2D
from PuzzleLib.Modules.MulAddConst import MulAddConst
from PuzzleLib.Modules.Replicate import Replicate
from PuzzleLib.Modules.SoftMax import SoftMax
from PuzzleLib.Modules.Split import Split
from PuzzleLib.Modules.Upsample2D import Upsample2D

from PuzzleLib.Converter.OpenVINO import Driver
from PuzzleLib.Converter.OpenVINO.VINOEngine import VINOEngine, genEngineName


def buildVINOEngine(net, inshape, savepath, returnEngine=True):
	outshape = net.dataShapeFrom(inshape)

	inshape = inshape if isinstance(inshape, list) else [inshape]
	outshape = outshape if isinstance(outshape, list) else [outshape]

	batchsize = inshape[0][0]
	inshape, outshape = [sh[1:] for sh in inshape], [sh[1:] for sh in outshape]

	engineName = genEngineName(net.name, inshape, outshape)

	xmlpath, binpath = os.path.join(savepath, "%s.xml" % engineName), os.path.join(savepath, "%s.bin" % engineName)
	convert(net, inshape, xmlpath, binpath)

	if returnEngine:
		return VINOEngine(batchsize, xmlpath, binpath, inshape, outshape)


def convert(net, inshape, xmlpath, binpath):
	graph = Driver.createNetwork(net.name)

	inshape = inshape if isinstance(inshape, list) else [inshape]
	inputs = [graph.addInput("data_%s" % i, shape) for i, shape in enumerate(inshape)]

	output = convertModule(net, net.name, graph, inputs)

	for i, out in enumerate(output):
		graph.markOutput(out, "outdata_%s" % i)

	graph.build(xmlpath, binpath)


def numpyPtr(ary):
	return ary.__array_interface__["data"][0]


def convertModule(module, fullname, graph, inputs):
	if isinstance(module, Container):
		if isinstance(module, Sequential):
			return convertSequential(module, fullname, graph, inputs)

		elif isinstance(module, Parallel):
			return convertParallel(module, fullname, graph, inputs)

		elif isinstance(module, Graph):
			return convertGraph(module, fullname, graph, inputs)

		else:
			raise NotImplementedError(module.__class__.__name__)

	else:
		inshape = [tuple(inp.shape) for inp in inputs]
		shape = module.dataShapeFrom(inshape[0] if len(inshape) == 1 else inshape)

		if isinstance(module, Add):
			return convertAdd(fullname, graph, inputs)

		elif isinstance(module, Concat):
			return convertConcat(module, fullname, graph, inputs, shape)

		assert len(inputs) == 1
		inp = inputs[0]

		if isinstance(module, Conv2D):
			return convertConv(module, fullname, graph, inp, shape)

		elif isinstance(module, (BatchNorm2D, BatchNorm)):
			return convertBatchNorm(module, fullname, graph, inp)

		elif isinstance(module, Activation):
			return convertActivation(module, fullname, graph, inp)

		elif isinstance(module, (Identity, Dropout)):
			return convertIdentity(inp)

		elif isinstance(module, Replicate):
			return convertReplicate(module, inp)

		elif isinstance(module, (MaxPool2D, AvgPool2D)):
			return convertPool2D(module, fullname, graph, inp, shape)

		elif isinstance(module, Flatten):
			return convertFlatten(inp, fullname, graph, shape)

		elif isinstance(module, Linear):
			return convertLinear(module, fullname, graph, inp, shape)

		elif isinstance(module, SoftMax):
			return convertSoftmax(fullname, graph, inp)

		elif isinstance(module, MulAddConst):
			return convertMulAddConst(module, fullname, graph, inp)

		elif isinstance(module, Split):
			return convertSplit(module, fullname, graph, inp, shape)

		elif isinstance(module, Upsample2D):
			return convertUpsample2D(module, fullname, graph, inp)

		else:
			raise NotImplementedError(module.__class__.__name__)


def convertSequential(seq, fullname, graph, inputs):
	for child in seq.graph:
		name = None if child.name is None else "%s.%s" % (fullname, child.name)
		inputs = convertModule(child, name, graph, inputs)

	return inputs


def convertParallel(parallel, fullname, graph, inputs):
	assert len(inputs) == len(parallel.graph)

	outputs = []
	for i, child in enumerate(parallel.graph):
		name = None if child.name is None else "%s.%s" % (fullname, child.name)
		outputs.append(convertModule(child, name, graph, [inputs[i]])[0])

	return outputs


def convertNode(node, fullname, graph, inputs, nodes):
	name = None if node.name is None else "%s.%s" % (fullname, node.name)
	inputs = [inputs[node.name]] if len(node.bwds) == 0 else [nodes[output.name] for output, _ in node.bwds]

	outputs = convertModule(node.module, name, graph, inputs)
	assert len(outputs) == 1

	nodes[node.name] = outputs[0]


def convertGraph(hostgraph, fullname, devgraph, inputs):
	assert len(inputs) == len(hostgraph.inputs)

	nodes = {}
	inputs = {node.name: inputs[i] for i, node in enumerate(hostgraph.inputs)}

	for i, inp in enumerate(hostgraph.inputs):
		inp.traverseForward(inp, convertNode, fullname, devgraph, inputs, nodes)

	hostgraph.reset()
	outputs = [nodes[output.name] for output in hostgraph.outputs]

	return outputs


def convertAdd(fullname, graph, inputs):
	assert len(inputs) == 2
	output = graph.addAdd(inputs[0], inputs[1], fullname)

	return [output]


def convertConcat(module, fullname, graph, inputs, shape):
	assert module.axis == 1
	output = graph.addConcat(inputs, shape, fullname)

	return [output]


def convertConv(module, fullname, graph, inp, shape):
	assert module.groups == 1 and module.dilation == (1, 1)

	W = module.W.get()

	b = module.b.get() if module.useBias else None
	bptr = 0 if b is None else numpyPtr(b)

	output = graph.addConvolution(inp, shape, module.W.shape[2:], numpyPtr(W), bptr, module.stride, module.pad, fullname)
	return [output]


def convertBatchNorm(module, fullname, graph, inp):
	mean, var = module.mean.get(), module.var.get()
	scale, bias = module.scale.get(), module.bias.get()

	eps = module.epsilon

	shift = (bias - scale * mean / np.sqrt(var + eps)).ravel()
	scale = (scale / np.sqrt(var + eps)).ravel()

	output = graph.addScale(inp, scale.shape[0], numpyPtr(scale), numpyPtr(shift), fullname)
	return [output]


def convertActivation(module, fullname, graph, inp):
	actType = module.getBlueprint()["scheme"]["activation"]

	typ = {
		relu: Driver.ActivationType.relu,
		leakyRelu: Driver.ActivationType.relu,
		sigmoid: Driver.ActivationType.sigmoid
	}[actType]

	alpha = 0.0
	if actType == leakyRelu:
		alpha = module.actArgs[0]

	output = graph.addActivation(inp, typ, alpha, fullname)
	return [output]


def convertIdentity(inp):
	return [inp]


def convertReplicate(module, inp):
	return [inp] * module.times


def convertPool2D(module, fullname, graph, inp, shape):
	output = graph.addPooling(
		inp, shape, True if isinstance(module, AvgPool2D) else False, module.size, module.stride, module.pad, fullname
	)

	return [output]


def convertFlatten(inp, fullname, graph, shape):
	output = graph.addFlatten(inp, shape, fullname)
	return [output]


def convertLinear(module, fullname, graph, inp, shape):
	W = module.W.get().T.ravel()

	b = module.b.get().ravel() if module.useBias else None
	bptr = 0 if b is None else numpyPtr(b)

	output = graph.addLinear(inp, shape, numpyPtr(W), bptr, fullname)
	return [output]


def convertSoftmax(fullname, graph, inp):
	output = graph.addSoftmax(inp, fullname)
	return [output]


def convertMulAddConst(module, fullname, graph, inp):
	c = inp.shape[1]

	shift = np.array([module.b] * c, dtype=np.float32)
	scale = np.array([module.a] * c, dtype=np.float32)

	output = graph.addScale(inp, c, numpyPtr(scale), numpyPtr(shift), fullname)
	return [output]


def convertSplit(module, fullname, graph, inp, shape):
	assert module.axis == 1

	output = graph.addSplit(inp, module.axis, shape, fullname)
	return output


def convertUpsample2D(module, fullname, graph, inp):
	assert module.mode == "nearest"

	output = graph.addUpsample(inp, module.scale, fullname)
	return [output]
