import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers.Container import ContainerError, Container
from PuzzleLib.Containers.Node import Node


class Graph(Container):
	def __init__(self, inputs, outputs, unsafe=False, nodesOnly=False, name=None):
		super().__init__(name)
		self.unsafe = unsafe

		self.inputs = [inputs] if not isinstance(inputs, list) else inputs

		impureInputs = [inp.name for inp in self.inputs if len(inp.bwds) > 0]
		if len(impureInputs) > 0:
			raise ContainerError("Found input nodes with parents: %s" % ", ".join(impureInputs))

		self.outputs = [outputs] if not isinstance(outputs, list) else outputs

		impureOutputs = [output.name for output in self.outputs if len(output.fwds) > 0]
		if len(impureOutputs) > 0:
			raise ContainerError("Found output nodes with ancestors: %s" % ", ".join(impureOutputs))

		self.nodes = {}
		for inp in self.inputs:
			inp.traverseForward(inp, lambda node: self.gatherTopology(node, nodesOnly))

		unvisited = [output.name for output in self.outputs if not output.fwdVisited]
		if len(unvisited) > 0:
			raise ContainerError("Could not visit output nodes: %s" % ", ".join(unvisited))

		self.reset()


	def gatherTopology(self, node, nodesOnly):
		if not nodesOnly:
			self.append(node.module)

		assert node.name not in self.nodes
		self.nodes[node.name] = node

		if getattr(node.module, "inplace", False) and not self.unsafe:
			for fwd in node.fwds:
				if len(fwd[0].bwds) > 1:
					raise ContainerError("Invalid inplace mode - module %s has non-trivial ancestor %s" %
										 (node.module, fwd[0]))

			for bwd in node.bwds:
				if len(bwd[0].fwds) > 1:
					raise ContainerError("Invalid inplace mode - module %s has non-trivial parent %s" %
										 (node.module, bwd[0]))


	def getBlueprint(self):
		blueprint = super().getBlueprint()

		blueprint["graph"] = {node.name: [(n.name, slots) for n, slots in node.bwds] for node in self.nodes.values()}
		blueprint["inputs"] = [inp.name for inp in self.inputs]
		blueprint["outputs"] = [output.name for output in self.outputs]

		return blueprint


	def getNodeByName(self, name):
		return self.nodes[name]


	def optimizeForShape(self, shape, memlimit=None):
		self.graphDataShape(shape, lambda module, sh: module.optimizeForShape(sh, memlimit))


	def updateData(self, data):
		data = data if isinstance(data, list) else [data]

		for i, inp in enumerate(self.inputs):
			inp.forward(data[i])

		self.data = self.outputs[0].data if len(self.outputs) == 1 else [output.data for output in self.outputs]
		self.clearTraverse()


	def dataShapeFrom(self, shape):
		return self.graphDataShape(shape, None)


	def graphDataShape(self, shape, onmodule):
		shape = shape if isinstance(shape, list) else [shape]

		inshapes = {inp.name: shape[i] for i, inp in enumerate(self.inputs)}
		shapes = {}

		for i, inp in enumerate(self.inputs):
			inp.traverseForward(inp, Node.dataShapeFrom, inshapes, shapes, onmodule)

		outshapes = [shapes[output.name] for output in self.outputs]
		if len(self.outputs) == 1:
			outshapes = outshapes[0]

		self.clearTraverse()
		return outshapes


	def backward(self, grad, updParamGrads=True, updGrad=True, scale=1.0, momentum=1.0):
		grad = grad if isinstance(grad, list) else [grad]

		for i, output in enumerate(self.outputs):
			output.backward(grad[i], updParamGrads=updParamGrads, updGrad=updGrad, scale=scale, momentum=momentum)

		self.grad = self.inputs[0].grad if len(self.inputs) == 1 else [inp.grad for inp in self.inputs]
		self.clearTraverse()


	def gradShapeFrom(self, shape):
		shape = shape if isinstance(shape, list) else [shape]

		outshapes = {output.name: shape[i] for i, output in enumerate(self.outputs)}
		shapes = {}

		for i, output in enumerate(self.outputs):
			output.traverseBackward(output, Node.gradShapeFrom, outshapes, shapes)

		inshape = [shapes[inp.name] for inp in self.inputs]
		if len(self.inputs) == 1:
			inshape = inshape[0]

		self.clearTraverse()
		return inshape


	def updateGrad(self, grad):
		assert False


	def reset(self):
		super().reset()

		for node in self.nodes.values():
			node.reset()


	def clearTraverse(self):
		for node in self.nodes.values():
			node.clearTraverse()


def unittest():
	calcTest()
	matchTest()


def calcTest():
	from PuzzleLib.Modules import Linear, Split, Concat, Activation, relu

	v1 = Linear(100, 50, name="v1").node()
	h1 = Split(axis=1, sections=(20, 20, 10), name="h1").node(v1)

	v2 = Linear(100, 50, name="v2").node()
	h2 = Concat(axis=1, name="h2").node((h1, [1, 2]), v2)
	h3 = Activation(relu, name="h3").node(h2)

	h4 = Concat(axis=1, name="h4").node((h1, 0), h3)

	mlp = Graph(inputs=[v1, v2], outputs=h4)

	v1data = gpuarray.to_gpu(np.random.randn(5, 100).astype(np.float32))
	v2data = gpuarray.to_gpu(np.random.randn(5, 100).astype(np.float32))

	mlp.optimizeForShape([v1data.shape, v2data.shape])
	mlp([v1data, v2data])

	assert mlp.data.shape == (5, 100)
	assert mlp.dataShapeFrom([v1data.shape, v2data.shape]) == mlp.data.shape

	grad = gpuarray.to_gpu(np.random.randn(*mlp.data.shape).astype(np.float32))
	mlp.backward(grad)

	assert len(mlp.grad) == 2 and mlp.grad[0].shape == mlp.grad[1].shape == (5, 100)
	assert mlp.gradShapeFrom(grad.shape) == [gr.shape for gr in mlp.grad]


def matchTest():
	from PuzzleLib.Containers import Sequential, Parallel
	from PuzzleLib.Modules import Linear, Activation, sigmoid, Replicate, Concat

	seq = Sequential()
	seq.append(Linear(128, 64, name="linear-1"))
	seq.append(Activation(sigmoid))
	seq.append(Replicate(times=2))

	parallel = Parallel()
	parallel.append(Linear(64, 10, name="linear-2"))
	parallel.append(Linear(64, 5, name="linear-3"))
	seq.append(parallel)

	seq.append(Concat(axis=1))

	v1 = Linear(128, 64, name="linear-1").node()
	h1 = Activation(sigmoid).node(v1)

	h2 = Linear(64, 10, name="linear-2").node(h1)
	h3 = Linear(64, 5, name="linear-3").node(h1)

	h4 = Concat(axis=1).node(h2, h3)

	mlp = Graph(inputs=v1, outputs=h4)

	mlp.getByName("linear-1").W.set(seq.getByName("linear-1").W)
	mlp.getByName("linear-1").b.set(seq.getByName("linear-1").b)

	mlp.getByName("linear-2").W.set(seq.getByName("linear-2").W)
	mlp.getByName("linear-2").b.set(seq.getByName("linear-2").b)

	mlp.getByName("linear-3").W.set(seq.getByName("linear-3").W)
	mlp.getByName("linear-3").b.set(seq.getByName("linear-3").b)

	data = gpuarray.to_gpu(np.random.randn(32, 128).astype(np.float32))
	seq(data)
	mlp(data)

	assert np.allclose(seq.data.get(), mlp.data.get())

	grad = gpuarray.to_gpu(np.random.randn(32, 15).astype(np.float32))
	seq.backward(grad)
	mlp.backward(grad)

	assert np.allclose(seq.grad.get(), mlp.grad.get())


if __name__ == "__main__":
	unittest()
