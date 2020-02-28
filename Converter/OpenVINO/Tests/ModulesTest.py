import numpy as np

from PuzzleLib import Config

Config.backend = Config.Backend.intel
Config.globalEvalMode = True

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers import Sequential, Parallel

from PuzzleLib.Modules.BatchNorm import BatchNorm
from PuzzleLib.Modules.Concat import Concat
from PuzzleLib.Modules.MulAddConst import MulAddConst
from PuzzleLib.Modules.Split import Split
from PuzzleLib.Modules.SoftMax import SoftMax
from PuzzleLib.Modules.Upsample2D import Upsample2D

from PuzzleLib.Converter.OpenVINO.BuildVINOEngine import buildVINOEngine


def batchNormTest():
	batchsize, size = 16, 10

	mod = BatchNorm(size, name="bn")
	mod.evalMode()

	data = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))

	engine = buildVINOEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def concatTest():
	batchsize, height, width = 4, 5, 8
	maps1, maps2 = 3, 2

	mod = Concat(axis=1, name="concat")
	data = [
		gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32)) for maps in [maps1, maps2]
	]

	engine = buildVINOEngine(mod, [subdata.shape for subdata in data], savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def mulAddConstTest():
	batchsize, maps, height, width = 4, 3, 5, 8

	mod = MulAddConst(a=1.5, b=-2.0, name="muladd")
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32))

	engine = buildVINOEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def splitTest():
	batchsize, maps, height, width = 2, 6, 4, 5

	net = Sequential(name="split")
	net.append(Split(axis=1, sections=(2, 4)))
	net.append(Parallel().append(SoftMax()).append(SoftMax()))

	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32))
	engine = buildVINOEngine(net, data.shape, savepath="../TestData")

	outdata = net(data)
	enginedata = engine(data)

	assert all(np.allclose(outdat.get(), enginedat.get()) for outdat, enginedat in zip(outdata, enginedata))


def upsample2dTest():
	batchsize, maps, height, width = 4, 3, 5, 8

	mod = Upsample2D(scale=2, name="upsample")
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32))

	engine = buildVINOEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def main():
	batchNormTest()
	concatTest()
	mulAddConstTest()
	splitTest()
	upsample2dTest()


if __name__ == "__main__":
	main()
