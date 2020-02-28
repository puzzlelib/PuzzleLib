import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool

from PuzzleLib.Containers.Sequential import Sequential

from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.Activation import Activation, relu, leakyRelu
from PuzzleLib.Modules.MaxPool2D import MaxPool2D
from PuzzleLib.Modules.Flatten import Flatten
from PuzzleLib.Modules.Linear import Linear
from PuzzleLib.Modules.SoftMax import SoftMax


def block(idx, inmaps, outmaps, sizeconv, strideconv, initscheme, actInPlace, sizepool=2, stridepool=2,
		  addMaxpool=True):
	assert len(inmaps) == len(outmaps) == len(sizeconv) == len(strideconv) == len(idx), "lengths must be the same size"

	seq = Sequential()

	for i in range(len(inmaps)):
		seq.append(Conv2D(
			inmaps=inmaps[i], outmaps=outmaps[i], size=sizeconv[i], pad=sizeconv[i] // 2, stride=strideconv[i],
			initscheme=initscheme, dilation=1, useBias=True, name="conv%s" % idx[i]
		))
		seq.append(Activation(leakyRelu, inplace=actInPlace, args=(0.01, )))

	if addMaxpool:
		seq.append(MaxPool2D(size=sizepool, stride=stridepool, name="conv%s_pool" % idx[-1]))

	return seq


def loadMiniYolo(modelpath, numOutput, actInplace=False, initscheme="none"):
	net = Sequential(name="YOLONet")

	block0 = block(
		idx=["1"], inmaps=[3], outmaps=[64], sizeconv=[7], strideconv=[2], initscheme=initscheme,
		actInPlace=actInplace
	)
	net.extend(block0)

	block1 = block(
		idx=["2"], inmaps=[64], outmaps=[192], sizeconv=[3], strideconv=[1], initscheme=initscheme,
		actInPlace=actInplace
	)
	net.extend(block1)

	block2 = block(
		idx=["3", "4", "5", "6"], inmaps=[192, 128, 256, 256], outmaps=[128, 256, 256, 512],
		sizeconv=[1, 3, 1, 3], strideconv=[1, 1, 1, 1], initscheme=initscheme, actInPlace=actInplace
	)
	net.extend(block2)

	block3 = block(
		idx=["7", "8", "9", "10", "11", "12", "13", "14", "15", "16"],
		inmaps=[512, 256, 512, 256, 512, 256, 512, 256, 512, 512],
		outmaps=[256, 512, 256, 512, 256, 512, 256, 512, 512, 1024],
		sizeconv=[1, 3, 1, 3, 1, 3, 1, 3, 1, 3], strideconv=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
		initscheme=initscheme, actInPlace=actInplace
	)
	net.extend(block3)

	block4 = block(
		idx=["17", "18", "19", "20", "21", "22", "23", "24"],
		inmaps=[1024, 512, 1024, 512, 1024, 1024, 1024, 1024],
		outmaps=[512, 1024, 512, 1024, 1024, 1024, 1024, 1024],
		sizeconv=[1, 3, 1, 3, 3, 3, 3, 3], strideconv=[1, 1, 1, 1, 1, 2, 1, 1],
		initscheme=initscheme, actInPlace=actInplace, addMaxpool=False
	)
	net.extend(block4)

	net.append(Flatten())
	insize = int(np.prod(net.dataShapeFrom((1, 3, 448, 448))))

	net.append(Linear(insize, 512, initscheme=initscheme, name="fc25"))
	net.append(Activation(relu, inplace=actInplace, name="fc_relu24"))

	net.append(Linear(512, 4096, initscheme=initscheme, name="fc26"))
	net.append(Activation(relu, inplace=actInplace, name="fc_relu25"))

	net.append(Linear(4096, numOutput, initscheme=initscheme, name="fc27"))
	net.append(SoftMax())

	if modelpath is not None:
		net.load(modelpath)

	return net


def unittest():
	data = gpuarray.to_gpu(np.zeros((1, 3, 448, 448), dtype=np.float32))

	yolo = loadMiniYolo(None, numOutput=1470, initscheme="gaussian")
	yolo(data)

	del yolo
	memPool.freeHeld()


if __name__ == "__main__":
	unittest()
