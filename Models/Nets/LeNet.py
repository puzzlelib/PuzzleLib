import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Containers.Sequential import Sequential

from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.MaxPool2D import MaxPool2D
from PuzzleLib.Modules.Activation import Activation, relu
from PuzzleLib.Modules.Flatten import Flatten
from PuzzleLib.Modules.Linear import Linear


def loadLeNet(modelpath, initscheme="none", name="lenet-5-like"):
	net = Sequential(name=name)

	net.append(Conv2D(1, 16, 3, initscheme=initscheme))
	net.append(MaxPool2D())
	net.append(Activation(relu))

	net.append(Conv2D(16, 32, 4, initscheme=initscheme))
	net.append(MaxPool2D())
	net.append(Activation(relu))

	net.append(Flatten())
	net.append(Linear(32 * 5 * 5, 1024, initscheme=initscheme))
	net.append(Activation(relu))

	net.append(Linear(1024, 10, initscheme=initscheme))

	if modelpath is not None:
		net.load(modelpath)

	return net


def unittest():
	data = gpuarray.to_gpu(np.random.randn(1, 1, 28, 28).astype(np.float32))

	net = loadLeNet(None)
	net(data)


if __name__ == "__main__":
	unittest()
