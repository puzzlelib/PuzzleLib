import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers.Sequential import Sequential

from PuzzleLib.Modules.Activation import Activation, relu, leakyRelu, clip
from PuzzleLib.Modules.BatchNorm import BatchNorm
from PuzzleLib.Modules.BatchNorm1D import BatchNorm1D
from PuzzleLib.Modules.InstanceNorm2D import InstanceNorm2D
from PuzzleLib.Modules.Conv1D import Conv1D
from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.CrossMapLRN import CrossMapLRN
from PuzzleLib.Modules.Deconv2D import Deconv2D
from PuzzleLib.Modules.GroupLinear import GroupLinear
from PuzzleLib.Modules.MulAddConst import MulAddConst
from PuzzleLib.Modules.Pad1D import Pad1D, PadMode
from PuzzleLib.Modules.PRelu import PRelu
from PuzzleLib.Modules.Reshape import Reshape
from PuzzleLib.Modules.RNN import RNN
from PuzzleLib.Modules.Split import Split
from PuzzleLib.Modules.SwapAxes import SwapAxes
from PuzzleLib.Modules.Upsample2D import Upsample2D

from PuzzleLib.Converter.TensorRT.BuildRTEngine import buildRTEngine


def deconv2dTest():
	batchsize, inmaps, inh, inw = 2, 3, 4, 5
	outmaps = 5

	mod = Deconv2D(inmaps, outmaps, size=2, stride=2, postpad=1, name="deconv", useBias=False)
	data = gpuarray.to_gpu(np.random.randn(batchsize, inmaps, inh, inw).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def crossMapLRNTest():
	batchsize, maps, height, width = 2, 5, 3, 4

	mod = CrossMapLRN(name="lrn")
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def groupLinearTest():
	batchsize, insize, outsize = 4, 3, 5
	groups = 2

	mod = GroupLinear(None, insize, outsize, wmode="one", name="groupLinear")
	mod.b.set(np.random.randn(1, outsize).astype(np.float32))

	data = gpuarray.to_gpu(np.random.randn(batchsize, groups, insize).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def mulAddConstTest():
	batchsize, maps, height, width = 4, 3, 5, 8

	mod = MulAddConst(a=1.5, b=-2.0, name="muladd")
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def batchNormTest():
	batchsize, size = 16, 10

	mod = BatchNorm(size, name="bn")
	mod.evalMode()

	data = gpuarray.to_gpu(np.random.randn(batchsize, size).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def batchNorm1dTest():
	batchsize, maps, size = 2, 3, 5

	mod = BatchNorm1D(maps, size, name="bn1d")
	mod.evalMode()

	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, size).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def instanceNorm2dTest():
	batchsize, maps, inh, inw = 5, 3, 4, 6

	mod = InstanceNorm2D(maps, name="instnorm")
	mod.evalMode()

	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, inh, inw).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def conv1dTest():
	batchsize, inmaps, insize = 2, 3, 5
	outmaps = 4

	mod = Conv1D(inmaps, outmaps, size=2, stride=2, name="conv1d", useBias=False)
	data = gpuarray.to_gpu(np.random.randn(batchsize, inmaps, insize).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def splitTest():
	batchsize, maps, height, width = 2, 6, 4, 5

	mod = Split(axis=1, sections=(2, 4), name="split")
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert all(np.allclose(od.get(), ed.get()) for od, ed in zip(outdata, enginedata))


def rnnTest():
	batchsize, inmaps, inh, inw = 4, 2, 3, 3
	outmaps, hsize = 4, 1

	seq = Sequential(name="rnn")

	seq.append(Conv2D(inmaps, outmaps, 3, pad=1))
	seq.append(Activation(relu))
	seq.append(Reshape(shape=(batchsize, outmaps, inh * inw)))

	seq.append(SwapAxes(0, 1))
	seq.append(RNN(inh * inw, hsize, layers=2, direction="bi", mode="tanh", getSequences=True, hintBatchSize=batchsize))
	seq.append(SwapAxes(0, 1))

	data = gpuarray.to_gpu(np.random.randn(batchsize, inmaps, inh, inw).astype(np.float32))

	engine = buildRTEngine(seq, data.shape, savepath="../TestData")

	outdata = seq(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def lstmTest():
	batchsize, seqlen, insize = 4, 6, 5
	hsize = 3

	seq = Sequential(name="lstm")

	seq.append(SwapAxes(0, 1))
	seq.append(RNN(insize, hsize, mode="lstm", getSequences=True, hintBatchSize=batchsize))
	seq.append(SwapAxes(0, 1))

	data = gpuarray.to_gpu(np.random.randn(batchsize, seqlen, insize).astype(np.float32))

	engine = buildRTEngine(seq, data.shape, savepath="../TestData")

	outdata = seq(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def gruTest():
	batchsize, seqlen, insize = 5, 6, 4
	hsize = 3

	seq = Sequential(name="gru")

	seq.append(SwapAxes(0, 1))
	seq.append(RNN(insize, hsize, mode="gru", getSequences=True, hintBatchSize=batchsize))
	seq.append(SwapAxes(0, 1))

	data = gpuarray.to_gpu(np.random.randn(batchsize, seqlen, insize).astype(np.float32))

	engine = buildRTEngine(seq, data.shape, savepath="../TestData")

	outdata = seq(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def upsample2dTest():
	batchsize, maps, height, width = 4, 3, 5, 8

	mod = Upsample2D(scale=2, name="upsample")
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def leakyReluTest():
	batchsize, maps, height, width = 4, 3, 5, 8

	mod = Activation(leakyRelu, name="leakyrelu")
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def clipTest():
	batchsize, maps, height, width = 4, 3, 5, 8

	mod = Activation(clip, name="clip")
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def preluTest():
	batchsize, maps, height, width = 4, 3, 5, 8

	mod = PRelu(maps=maps, name="prelu")
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, height, width).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def pad1dTest():
	batchsize, maps, size = 4, 5, 7
	lpad, rpad = 2, 3

	mod = Pad1D(pad=(lpad, rpad), mode=PadMode.reflect, name="reflectpad")
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, size).astype(np.float32))

	engine = buildRTEngine(mod, data.shape, savepath="../TestData")

	outdata = mod(data)
	enginedata = engine(data)

	assert np.allclose(outdata.get(), enginedata.get())


def main():
	deconv2dTest()
	crossMapLRNTest()
	groupLinearTest()
	mulAddConstTest()

	batchNormTest()
	batchNorm1dTest()
	instanceNorm2dTest()

	conv1dTest()
	splitTest()

	rnnTest()
	lstmTest()
	gruTest()

	leakyReluTest()
	clipTest()

	upsample2dTest()
	preluTest()
	pad1dTest()


if __name__ == "__main__":
	main()
