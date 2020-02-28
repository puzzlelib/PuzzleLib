import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers.Sequential import Sequential
from PuzzleLib.Containers.Parallel import Parallel
from PuzzleLib.Containers.Graph import Graph
from PuzzleLib.Containers.Node import Node

from PuzzleLib.Modules.Identity import Identity
from PuzzleLib.Modules.Replicate import Replicate
from PuzzleLib.Modules.ToList import ToList
from PuzzleLib.Modules.Glue import Glue


class ConverterError(Exception):
	pass


def toGraph(module, unsafe=False, nodesOnly=False, assumeUniqueNames=False):
	inputs, outputs = convertToGraph(module, None, None, assumeUniqueNames)

	graph = Graph(inputs=inputs, outputs=outputs, unsafe=unsafe, nodesOnly=nodesOnly, name=module.name)
	return graph


def convertToGraph(module, inputs, name, assumeUniqueNames):
	if isinstance(module, Sequential):
		return convertSequential(module, inputs, name, assumeUniqueNames)
	elif isinstance(module, Parallel):
		return convertParallel(module, inputs, name, assumeUniqueNames)
	elif isinstance(module, Graph):
		return convertGraph(module, inputs, name, assumeUniqueNames)
	else:
		return convertModule(module, inputs, name, assumeUniqueNames)


def convertSequential(seq, inputs, name, assumeUniqueNames):
	outputs = inputs

	for mod in seq.graph:
		if assumeUniqueNames:
			modname = None
		else:
			modname = "%s_%s" % (name, mod.name) if name is not None else mod.name

		newInputs, outputs = convertToGraph(mod, outputs, name=modname, assumeUniqueNames=assumeUniqueNames)
		inputs = inputs if inputs is not None else newInputs

	return inputs, outputs


def convertParallel(parallel, inputs, name, assumeUniqueNames):
	overwriteInputs = False
	outputs = []

	if inputs is None:
		overwriteInputs = True
		inputs = []

	for mod in parallel.graph:
		if assumeUniqueNames:
			modname = None
		else:
			modname = "%s_%s" % (name, mod.name) if name is not None else mod.name

		newInputs, newOutputs = convertToGraph(mod, inputs, name=modname, assumeUniqueNames=assumeUniqueNames)

		if overwriteInputs:
			inputs.extend(newInputs)

		outputs.extend(newOutputs)

	return inputs, outputs


def convertGraph(graph, inputs, name, assumeUniqueNames):
	nodes = {}

	for node in graph.nodes.values():
		if assumeUniqueNames:
			modname = None
		else:
			modname = node.name if name is None else "%s_%s" % (name, node.name)

		name = node.name

		newInputs, newOutputs = convertToGraph(node.module, None, name=modname, assumeUniqueNames=assumeUniqueNames)
		nodes[node.name] = newInputs, newOutputs, name

	for nodeInputs, nodeOutputs, name in nodes.values():
		if not isinstance(nodeInputs, list):
			nodeInputs = [nodeInputs]

		for inp in nodeInputs:
			inp.addBackwards([(nodes[n.name][1][0], slots) for n, slots in graph.nodes[name].bwds])

	newInputs = [nodes[inp.name][0] for inp in graph.inputs]
	newOutputs = [nodes[output.name][1] for output in graph.outputs]

	for i, inp in enumerate(newInputs):
		inp.addBackwards(inputs[i])

	return inputs, newOutputs


def convertModule(module, inputs, name, _):
	if isinstance(module, (Identity, Replicate, ToList)):
		return inputs, inputs

	if isinstance(module, Glue):
		raise ConverterError("Cannot convert Glue module - result may be unpredictable")

	node = Node(module, parents=inputs, name=name)
	inputs = inputs if inputs is not None else node

	return inputs, [node]


def netTest():
	from PuzzleLib.Models.Nets.ResNet import loadResNet
	net = loadResNet(None, layers="50", initscheme="xavier")

	data = gpuarray.to_gpu(np.random.randn(1, 3, 224, 224).astype(np.float32))
	outdata = net(data)

	graph = toGraph(net)
	graphdata = graph(data)

	assert np.allclose(outdata.get(), graphdata.get())


def graphTest():
	from PuzzleLib.Modules import Linear, Activation, relu, Add

	net = Sequential()

	net.append(Linear(10, 10))
	net.append(Activation(relu))

	inp = Linear(10, 10).node()
	node = Activation(relu).node(inp)

	subseq = Sequential()
	subseq.append(Linear(10, 10))
	subseq.append(Activation(relu))

	node2 = subseq.node(node)
	node3 = Linear(10, 10).node(node)

	node = Add(name="add").node(node2, node3)

	graph = Graph(inputs=inp, outputs=node)
	net.append(graph)

	net.append(Linear(10, 10))
	net.append(Activation(relu))

	data = gpuarray.to_gpu(np.random.randn(1, 10).astype(np.float32))
	outdata = net(data)

	net.reset()

	graph = toGraph(net)
	graphdata = graph(data)

	assert np.allclose(outdata.get(), graphdata.get())


def unittest():
	netTest()
	graphTest()


if __name__ == "__main__":
	unittest()
