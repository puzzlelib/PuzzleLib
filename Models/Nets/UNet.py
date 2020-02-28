import numpy as np

from PuzzleLib.Backend.Utils import memoryPool as memPool
from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers import Parallel, Sequential
from PuzzleLib.Modules import Replicate, Conv2D, MaxPool2D, Activation, relu, sigmoid, Deconv2D, Concat, Identity, \
	Dropout


def blockA(blockId, actInplace, initscheme):
	assert isinstance(blockId, int)
	assert 1 <= blockId <= 5

	inmaps = 1 if blockId == 1 else 2**(4 + blockId)
	outmaps = 2**(5 + blockId)

	block = Sequential(name="block_%d" % blockId)
	if blockId > 1:
		block.append(MaxPool2D(size=2, stride=2, name="pool%d" % (blockId - 1, )))

	block.append(Conv2D(inmaps, outmaps, 3, pad=1, initscheme=initscheme, name="conv_%d_1" % blockId))
	block.append(Activation(relu, inplace=actInplace, name="relu%d" % (2 * blockId - 1, )))

	block.append(Conv2D(outmaps, outmaps, 3, pad=1, initscheme=initscheme, name="conv_%d_2" % blockId))
	block.append(Activation(relu, inplace=actInplace, name="relu%d" % (2 * blockId, )))

	if blockId >= 4:
		block.append(Dropout(name="drop%d" % blockId))

	if blockId == 5:
		block.append(Deconv2D(1024, 512, size=2, stride=2, useBias=False, initscheme=initscheme, name="upscore1"))
		block.append(Activation(relu, inplace=actInplace, name="relu11"))

	return block


def shortcut(blockId):
	assert isinstance(blockId, int)
	assert blockId < 6

	return Sequential(name="shortcut_%d" % blockId).append(Identity())


def blockB(blockId, actInplace, initscheme):
	assert type(blockId) is int
	assert 6 <= blockId <= 9

	inmaps = 2 ** (16 - blockId)
	outmaps = inmaps // 2
	reluId = 12 + (blockId - 6) * 3

	block = Sequential(name="block_%d" % blockId)

	block.append(Conv2D(inmaps, outmaps, 3, pad=1, initscheme=initscheme, name="conv_%d_1" % blockId))
	block.append(Activation(relu, inplace=actInplace, name="relu%d" % reluId))

	block.append(Conv2D(outmaps, outmaps, 3, pad=1, initscheme=initscheme, name="conv_%d_2" % blockId))
	block.append(Activation(relu, inplace=actInplace, name="relu%d" % (reluId + 1, )))

	if blockId < 9:
		block.append(Deconv2D(
			outmaps, outmaps // 2, 2, stride=2, useBias=False, initscheme=initscheme, name="upscore%d" % (blockId - 4)
		))

		block.append(Conv2D(
			outmaps // 2, outmaps // 2, size=3, pad=1, initscheme=initscheme, name="conv_%d_3" % blockId
		))
		block.append(Activation(relu, inplace=actInplace, name="relu%d" % (reluId + 2)))

	else:
		block.append(Conv2D(64, 1, 1, initscheme=initscheme, name="score"))
		block.append(Activation(sigmoid, inplace=actInplace))

	return block


def loadUNet(modelpath, actInplace=False, initscheme="none"):
	net = Sequential(name="unet")

	blocksA, blocksB, shortcuts = [None], [None] * 6, [None]

	for blockId in range(1, 6):
		blocksA.append(blockA(blockId, actInplace, initscheme))
		shortcuts.append(shortcut(blockId))

	for blockId in range(6, 10):
		blocksB.append(blockB(blockId, actInplace, initscheme))

	for blockId in range(1, 5):
		blocksA[blockId].append(Replicate(2))
		blocksA[blockId].append(
			Parallel(name="fork_%d" % blockId).append(blocksA[blockId + 1]).append(shortcuts[blockId + 1])
		)

	for blockId in range(4, 0, -1):
		blocksA[blockId].append(Concat(axis=1, name="concat%d" % (5 - blockId, )))
		blocksA[blockId].extend(blocksB[10 - blockId])

	net.extend(blocksA[1])

	if modelpath is not None:
		net.load(modelpath)

	return net


def unittest():
	inshape = (1, 1, 256, 256)
	unet = loadUNet(None, initscheme="gaussian")

	data = gpuarray.to_gpu(np.random.randn(*inshape).astype(np.float32))
	assert unet(data).shape == inshape

	del unet
	memPool.freeHeld()


if __name__ == "__main__":
	unittest()
