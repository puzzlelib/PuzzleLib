import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool

from PuzzleLib.Containers.Sequential import Sequential

from PuzzleLib.Modules.Conv1D import Conv1D
from PuzzleLib.Modules.BatchNorm1D import BatchNorm1D
from PuzzleLib.Modules.Dropout import Dropout
from PuzzleLib.Modules.Activation import Activation, clip
from PuzzleLib.Modules.Pad1D import Pad1D, PadMode


def convBlock(inmaps, outmaps, size, stride, pad, dropout, initscheme, dilation=1, bnAct=True, name=None):
	block = Sequential()

	if pad > 0:
		block.append(Pad1D(pad, mode=PadMode.reflect))

	block.append(
		Conv1D(
			inmaps, outmaps, size=size, stride=stride, pad=0, dilation=dilation, useBias=True, initscheme=initscheme,
			name="%s_conv" % name
		)
	)

	if bnAct:
		block.append(BatchNorm1D(outmaps, epsilon=0.001, name="%s_bn" % name))
		block.append(Activation(clip, args=(0.0, 20.0)))

	if dropout > 0.0:
		block.append(Dropout(p=dropout))

	return block


def loadW2L(modelpath, inmaps, nlabels, initscheme=None, name="w2l"):
	net = Sequential(name=name)

	net.extend(convBlock(inmaps, 256, size=11, stride=2, pad=5, dropout=0.2, initscheme=initscheme, name="conv1d_0"))

	net.extend(convBlock(256, 256, size=11, stride=1, pad=5, dropout=0.2, initscheme=initscheme, name="conv1d_1"))
	net.extend(convBlock(256, 256, size=11, stride=1, pad=5, dropout=0.2, initscheme=initscheme, name="conv1d_2"))
	net.extend(convBlock(256, 256, size=11, stride=1, pad=5, dropout=0.2, initscheme=initscheme, name="conv1d_3"))

	net.extend(convBlock(256, 384, size=13, stride=1, pad=6, dropout=0.2, initscheme=initscheme, name="conv1d_4"))
	net.extend(convBlock(384, 384, size=13, stride=1, pad=6, dropout=0.2, initscheme=initscheme, name="conv1d_5"))
	net.extend(convBlock(384, 384, size=13, stride=1, pad=6, dropout=0.2, initscheme=initscheme, name="conv1d_6"))

	net.extend(convBlock(384, 512, size=17, stride=1, pad=8, dropout=0.2, initscheme=initscheme, name="conv1d_7"))
	net.extend(convBlock(512, 512, size=17, stride=1, pad=8, dropout=0.2, initscheme=initscheme, name="conv1d_8"))
	net.extend(convBlock(512, 512, size=17, stride=1, pad=8, dropout=0.2, initscheme=initscheme, name="conv1d_9"))

	net.extend(convBlock(512, 640, size=21, stride=1, pad=10, dropout=0.3, initscheme=initscheme, name="conv1d_10"))
	net.extend(convBlock(640, 640, size=21, stride=1, pad=10, dropout=0.3, initscheme=initscheme, name="conv1d_11"))
	net.extend(convBlock(640, 640, size=21, stride=1, pad=10, dropout=0.3, initscheme=initscheme, name="conv1d_12"))

	net.extend(convBlock(640, 768, size=25, stride=1, pad=12, dropout=0.3, initscheme=initscheme, name="conv1d_13"))
	net.extend(convBlock(768, 768, size=25, stride=1, pad=12, dropout=0.3, initscheme=initscheme, name="conv1d_14"))
	net.extend(convBlock(768, 768, size=25, stride=1, pad=12, dropout=0.3, initscheme=initscheme, name="conv1d_15"))

	net.extend(convBlock(
		768, 896, size=29, stride=1, pad=28, dropout=0.4, initscheme=initscheme, dilation=2, name="conv1d_16"
	))

	net.extend(convBlock(896, 1024, size=1, stride=1, pad=0, dropout=0.4, initscheme=initscheme, name="conv1d_17"))

	net.extend(convBlock(
		1024, nlabels, size=1, stride=1, pad=0, dropout=0.0, initscheme=initscheme, bnAct=False, name="conv1d_18"
	))

	if modelpath is not None:
		net.load(modelpath)

	return net


def unittest():
	inmaps, nlabels = 161, 29
	w2l = loadW2L(None, inmaps=inmaps, nlabels=29)

	shape = (16, inmaps, 200)

	data = gpuarray.to_gpu(np.random.randn(*shape).astype(np.float32))
	w2l(data)

	del w2l
	memPool.freeHeld()


if __name__ == "__main__":
	unittest()
