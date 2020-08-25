import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Models.Nets.UNet import loadUNet

from PuzzleLib.Converter.TensorRT.Tests.Common import benchModels
from PuzzleLib.Converter.TensorRT.BuildRTEngine import buildRTEngine


def main():
	net = loadUNet(None)
	data = gpuarray.to_gpu(np.random.randn(1, 1, 256, 256).astype(np.float32))

	engine = buildRTEngine(net, inshape=data.shape, savepath="../TestData")

	net.evalMode()
	outdata = net(data)

	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())
	benchModels(net, engine, data)


if __name__ == "__main__":
	main()
