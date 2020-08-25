import numpy as np

from PuzzleLib import Config
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Models.Nets.WaveToLetter import loadW2L

from PuzzleLib.Converter.TensorRT.Tests.Common import benchModels
from PuzzleLib.Converter.TensorRT.BuildRTEngine import buildRTEngine


def main():
	inmaps = 161
	net = loadW2L(None, inmaps, nlabels=29)

	data = gpuarray.to_gpu(np.random.randn(1, inmaps, 200).astype(np.float32))
	engine = buildRTEngine(net, inshape=data.shape, savepath="../TestData")

	net.evalMode()
	outdata = net(data)

	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get(), atol=1e-7)
	benchModels(net, engine, data)


if __name__ == "__main__":
	main()
