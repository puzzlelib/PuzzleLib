import string

import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers.Sequential import Sequential
from PuzzleLib.Containers.Parallel import Parallel

from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.BatchNorm2D import BatchNorm2D
from PuzzleLib.Modules.Activation import Activation, relu
from PuzzleLib.Modules.Identity import Identity
from PuzzleLib.Modules.Replicate import Replicate
from PuzzleLib.Modules.Add import Add
from PuzzleLib.Modules.MaxPool2D import MaxPool2D
from PuzzleLib.Modules.AvgPool2D import AvgPool2D
from PuzzleLib.Modules.Flatten import Flatten
from PuzzleLib.Modules.Linear import Linear
from PuzzleLib.Modules.SoftMax import SoftMax


def residMiniBlock(inmaps, outmaps, size, stride, pad, blockname, mininame, addAct, actInplace, bnInplace, initscheme):
	block = Sequential()

	block.append(Conv2D(
		inmaps, outmaps, size, stride=stride, pad=pad, useBias=False, initscheme=initscheme,
		name="res%s_branch%s" % (blockname, mininame)
	))
	block.append(BatchNorm2D(outmaps, name="bn%s_branch%s" % (blockname, mininame), inplace=bnInplace))

	if addAct:
		block.append(Activation(relu, inplace=actInplace, name="res%s_branch%s_relu" % (blockname, mininame)))

	return block


def residBlock(inmaps, hmaps, stride, blockname, convShortcut, actInplace, bnInplace, initscheme):
	block = Sequential()

	branch = Sequential()
	branch.extend(residMiniBlock(inmaps, hmaps, 1, stride, 0, blockname, "2a", True, actInplace, bnInplace, initscheme))
	branch.extend(residMiniBlock(hmaps, hmaps, 3, 1, 1, blockname, "2b", True, actInplace, bnInplace, initscheme))
	branch.extend(residMiniBlock(hmaps, 4 * hmaps, 1, 1, 0, blockname, "2c", False, actInplace, bnInplace, initscheme))

	shortcut = Sequential()
	if convShortcut:
		shortcut.extend(residMiniBlock(
			inmaps, 4 * hmaps, 1, stride, 0, blockname, "1", False, actInplace, bnInplace, initscheme
		))
	else:
		shortcut.append(Identity())

	block.append(Replicate(2))
	block.append(Parallel().append(branch).append(shortcut))

	block.append(Add())
	block.append(Activation(relu, inplace=actInplace))

	return block


def loadResNet(modelpath, layers, actInplace=False, bnInplace=False, initscheme="none", name=None):
	if layers == "50":
		if name is None:
			name = "ResNet-50"

		level3names = ["3%s" % alpha for alpha in string.ascii_lowercase[1:4]]
		level4names = ["4%s" % alpha for alpha in string.ascii_lowercase[1:6]]

	elif layers == "101":
		if name is None:
			name = "ResNet-101"

		level3names = ["3b%s" % num for num in range(1, 4)]
		level4names = ["4b%s" % num for num in range(1, 23)]

	elif layers == "152":
		if name is None:
			name = "ResNet-152"

		level3names = ["3b%s" % num for num in range(1, 8)]
		level4names = ["4b%s" % num for num in range(1, 36)]

	else:
		raise ValueError("Unsupported ResNet layers mode")

	net = Sequential(name=name)

	net.append(Conv2D(3, 64, 7, stride=2, pad=3, name="conv1", initscheme=initscheme, useBias=False))
	net.append(BatchNorm2D(64, name="bn_conv1", inplace=bnInplace))
	net.append(Activation(relu, inplace=actInplace, name="conv1_relu"))
	net.append(MaxPool2D(3, 2, name="pool1"))

	net.extend(residBlock(64, 64, 1, "2a", True, actInplace, bnInplace, initscheme))
	net.extend(residBlock(256, 64, 1, "2b", False, actInplace, bnInplace, initscheme))
	net.extend(residBlock(256, 64, 1, "2c", False, actInplace, bnInplace, initscheme))

	net.extend(residBlock(256, 128, 2, "3a", True, actInplace, bnInplace, initscheme))

	for name in level3names:
		net.extend(residBlock(512, 128, 1, name, False, actInplace, bnInplace, initscheme))

	net.extend(residBlock(512, 256, 2, "4a", True, actInplace, bnInplace, initscheme))

	for name in level4names:
		net.extend(residBlock(1024, 256, 1, name, False, actInplace, bnInplace, initscheme))

	net.extend(residBlock(1024, 512, 2, "5a", True, actInplace, bnInplace, initscheme))
	net.extend(residBlock(2048, 512, 1, "5b", False, actInplace, bnInplace, initscheme))
	net.extend(residBlock(2048, 512, 1, "5c", False, actInplace, bnInplace, initscheme))

	net.append(AvgPool2D(7, 1))
	net.append(Flatten())
	net.append(Linear(2048, 1000, initscheme=initscheme, name="fc1000"))
	net.append(SoftMax())

	if modelpath is not None:
		net.load(modelpath, assumeUniqueNames=True)

	return net


def unittest():
	data = gpuarray.to_gpu(np.random.randn(1, 3, 224, 224).astype(np.float32))

	res = loadResNet(None, layers="50", initscheme="gaussian")
	res(data)

	del res
	gpuarray.memoryPool.freeHeld()

	res = loadResNet(None, layers="101", initscheme="gaussian")
	res(data)

	del res
	gpuarray.memoryPool.freeHeld()

	res = loadResNet(None, layers="152", initscheme="gaussian")
	res(data)

	del res
	gpuarray.memoryPool.freeHeld()


if __name__ == "__main__":
	unittest()
