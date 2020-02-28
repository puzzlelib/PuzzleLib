import numpy as np

from PuzzleLib import Config

Config.backend = Config.Backend.intel
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers import Graph
from PuzzleLib.Modules import Linear, Activation, relu, Add

from PuzzleLib.Converter.OpenVINO.Tests.Common import benchModels
from PuzzleLib.Converter.OpenVINO.BuildVINOEngine import buildVINOEngine


def main():
	batchsize, insize = 16, 1000

	inNode = Linear(insize, 1000, name="linear1").node()
	node = Activation(relu, name="relu1").node(inNode)

	node1 = Linear(1000, 800, name="linear2").node(node)
	node1 = Activation(relu, name="relu2").node(node1)

	node2 = Linear(1000, 800, name="linear3").node(node)
	node2 = Activation(relu, name="relu3").node(node2)

	outNode = Add(name="add").node(node1, node2)

	graph = Graph(inputs=inNode, outputs=outNode, name="graph")

	data = gpuarray.to_gpu(np.random.randn(batchsize, insize).astype(np.float32))

	engine = buildVINOEngine(graph, (batchsize, insize), savepath="../TestData")

	outdata = graph(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())
	benchModels(graph, engine, data)


if __name__ == "__main__":
	main()
