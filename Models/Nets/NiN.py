import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool

from PuzzleLib.Containers.Sequential import Sequential

from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.Activation import Activation, relu
from PuzzleLib.Modules.MaxPool2D import MaxPool2D
from PuzzleLib.Modules.AvgPool2D import AvgPool2D
from PuzzleLib.Modules.Flatten import Flatten
from PuzzleLib.Modules.SoftMax import SoftMax


def loadNiNImageNet(modelpath, poolmode="max", actInplace=False, initscheme="none", name="CaffeNet"):
	if poolmode == "avg":
		pool = AvgPool2D
	elif poolmode == "max":
		pool = MaxPool2D
	else:
		raise ValueError("Unsupported pool mode")

	net = Sequential(name=name)

	net.append(Conv2D(3, 96, 11, stride=4, initscheme=initscheme, name="conv1"))
	net.append(Activation(relu, inplace=actInplace, name="relu0"))
	net.append(Conv2D(96, 96, 1, stride=1, initscheme=initscheme, name="cccp1"))
	net.append(Activation(relu, inplace=actInplace, name="relu1"))
	net.append(Conv2D(96, 96, 1, stride=1, initscheme=initscheme, name="cccp2"))
	net.append(Activation(relu, inplace=actInplace, name="relu2"))
	net.append(pool(3, 2, name="pool1"))

	net.append(Conv2D(96, 256, 5, stride=1, pad=2, initscheme=initscheme, name="conv2"))
	net.append(Activation(relu, inplace=actInplace, name="relu3"))
	net.append(Conv2D(256, 256, 1, stride=1, initscheme=initscheme, name="cccp3"))
	net.append(Activation(relu, inplace=actInplace, name="relu4"))
	net.append(Conv2D(256, 256, 1, stride=1, initscheme=initscheme, name="cccp4"))
	net.append(Activation(relu, inplace=actInplace, name="relu5"))
	net.append(pool(3, 2, name="pool2"))

	net.append(Conv2D(256, 384, 3, stride=1, pad=1, initscheme=initscheme, name="conv3"))
	net.append(Activation(relu, inplace=actInplace, name="relu6"))
	net.append(Conv2D(384, 384, 1, stride=1, initscheme=initscheme, name="cccp5"))
	net.append(Activation(relu, inplace=actInplace, name="relu7"))
	net.append(Conv2D(384, 384, 1, stride=1, initscheme=initscheme, name="cccp6"))
	net.append(Activation(relu, inplace=actInplace, name="relu8"))
	net.append(pool(3, 2, name="pool3"))

	net.append(Conv2D(384, 1024, 3, stride=1, pad=1, initscheme=initscheme, name="conv4-1024"))
	net.append(Activation(relu, inplace=actInplace, name="relu9"))
	net.append(Conv2D(1024, 1024, 1, stride=1, initscheme=initscheme, name="cccp7-1024"))
	net.append(Activation(relu, inplace=actInplace, name="relu10"))
	net.append(Conv2D(1024, 1000, 1, stride=1, initscheme=initscheme, name="cccp8-1024"))
	net.append(Activation(relu, inplace=actInplace, name="relu11"))
	net.append(AvgPool2D(5, 1, name="pool4"))

	net.append(Flatten())
	net.append(SoftMax())

	if modelpath is not None:
		net.load(modelpath)

	return net


def unittest():
	nin = loadNiNImageNet(None, initscheme="gaussian")

	data = gpuarray.to_gpu(np.random.randn(1, 3, 224, 224).astype(np.float32))
	nin(data)

	del nin
	memPool.freeHeld()


if __name__ == "__main__":
	unittest()
