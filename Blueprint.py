import json, os, importlib.util

import numpy as np

from PuzzleLib.Config import libname
from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import Module
from PuzzleLib.Containers.Node import Node


class BlueprintError(Exception):
	pass


class BlueprintFactory:
	def __init__(self):
		self.containers = {}
		self.modules = {}

		paths = ["Containers", "Modules"]

		ignores = [
			{"Node", "Container", "__init__"},
			{"Module", "__init__"}
		]

		factories = [self.containers, self.modules]

		for path, ignore, factory in zip(paths, ignores, factories):
			factoryPath = os.path.join(os.path.dirname(__file__), path)
			for file in os.listdir(factoryPath):
				if file.endswith(".py") and os.path.splitext(file)[0] not in ignore:
					filepath = os.path.abspath(os.path.join(factoryPath, file))

					spec = importlib.util.spec_from_file_location(os.path.basename(filepath)[:-3], filepath)
					mod = importlib.util.module_from_spec(spec)

					spec.loader.exec_module(mod)

					name = os.path.splitext(file)[0]
					factory[name] = getattr(mod, name)


	def build(self, blueprint, log=False, logwidth=20):
		classname = blueprint["classname"]
		scheme = blueprint["scheme"]

		if classname in self.containers:
			graph = blueprint["graph"]
			elements = blueprint["modules"]

			if classname in {"Sequential", "Parallel"}:
				mod = self.containers[classname](name=scheme["name"])

				for name in graph:
					cl = self.build(elements[name], log=log)
					mod.append(cl)

			elif classname == "Graph":
				nodes = {name: Node(self.build(bprint, log=log)) for name, bprint in elements.items()}
				for node in nodes.values():
					node.addBackwards([(nodes[name], slots) for name, slots in graph[node.name]])

				inputs = [nodes[name] for name in blueprint["inputs"]]
				outputs = [nodes[name] for name in blueprint["outputs"]]

				mod = self.containers[classname](inputs, outputs, name=scheme["name"])

			else:
				raise NotImplementedError(classname)

		elif classname in self.modules:
			cl = self.modules[classname]

			if log:
				fmt = "[%s] Loading module named %-" + str(logwidth) + "s type %-" + str(logwidth) + "s ..."
				print(fmt % (libname, "'%s'" % scheme["name"], classname), end="")

			if "initscheme" in scheme:
				scheme["initscheme"] = "none"

			mod = cl(**scheme)

			if log:
				print(" Done")

		else:
			raise BlueprintError("Cannot build module with class name '%s'" % classname)

		return mod


def load(hdf, name=None, assumeUniqueNames=False, log=False, logwidth=20):
	with Module.ensureHdf(hdf, "r") as hdf:
		blueprint = json.loads(str(np.array(hdf["blueprint"])))

		if log:
			print("[%s] Building model from blueprint ..." % libname)

		mod = BlueprintFactory().build(blueprint, log=log, logwidth=logwidth)

		if log:
			print("[%s] Loading model data ..." % libname)

		mod.load(hdf, name=name, assumeUniqueNames=assumeUniqueNames)

	return mod


def unittest():
	fileTest()
	memoryTest()
	graphTest()


def buildNet():
	from PuzzleLib.Containers import Sequential, Parallel
	from PuzzleLib.Modules import Linear, Activation, relu, Replicate, Concat

	seq = Sequential()

	seq.append(Linear(20, 10, name="linear-1"))
	seq.append(Activation(relu, name="relu-1"))

	seq.append(Linear(10, 5, name="linear-2"))
	seq.append(Activation(relu, name="relu-2"))

	seq.append(Replicate(times=2, name="repl"))
	seq.append(Parallel().append(Linear(5, 2, name="linear-3-1")).append(Linear(5, 3, name="linear-3-2")))
	seq.append(Concat(axis=1, name="concat"))

	return seq


def buildGraph():
	from PuzzleLib.Containers import Graph
	from PuzzleLib.Modules import Linear, Activation, relu, Concat

	inp = Linear(20, 10, name="linear-1").node()
	h = Activation(relu, name="relu-1").node(inp)

	h1 = Linear(10, 5, name="linear-2").node(h)
	h2 = Linear(10, 5, name="linear-3").node(h)

	output = Concat(axis=1, name="concat").node(h1, h2)
	graph = Graph(inputs=inp, outputs=output)

	return graph


def fileTest():
	seq = buildNet()

	data = gpuarray.to_gpu(np.random.randn(32, 20).astype(np.float32))
	origOutData = seq(data)

	try:
		seq.save("./TestData/seq.hdf", withBlueprint=True)
		newSeq = load("./TestData/seq.hdf", log=True)

	finally:
		if os.path.exists("./TestData/seq.hdf"):
			os.remove("./TestData/seq.hdf")

	newOutData = newSeq(data)
	assert np.allclose(origOutData.get(), newOutData.get())


def memoryTest():
	seq = buildNet()

	data = gpuarray.to_gpu(np.random.randn(32, 20).astype(np.float32))
	origOutData = seq(data)

	mmap = seq.save(withBlueprint=True)
	newSeq = load(mmap, log=True)

	newOutData = newSeq(data)
	assert np.allclose(origOutData.get(), newOutData.get())


def graphTest():
	graph = buildGraph()

	data = gpuarray.to_gpu(np.random.randn(32, 20).astype(np.float32))
	origOutData = graph(data)

	try:
		graph.save("./TestData/graph.hdf", withBlueprint=True)
		newGraph = load("./TestData/graph.hdf", log=True)

	finally:
		if os.path.exists("./TestData/graph.hdf"):
			os.remove("./TestData/graph.hdf")

	newOutData = newGraph(data)
	assert np.allclose(origOutData.get(), newOutData.get())


if __name__ == "__main__":
	unittest()
