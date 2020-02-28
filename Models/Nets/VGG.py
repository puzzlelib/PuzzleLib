import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool

from PuzzleLib.Containers.Sequential import Sequential

from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.Activation import Activation, relu
from PuzzleLib.Modules.AvgPool2D import AvgPool2D
from PuzzleLib.Modules.MaxPool2D import MaxPool2D
from PuzzleLib.Modules.Flatten import Flatten
from PuzzleLib.Modules.Linear import Linear
from PuzzleLib.Modules.SoftMax import SoftMax


def loadVGG(modelpath, layers, poolmode="max", initscheme="none", withLinear=True, actInplace=False, name=None):
	if poolmode == "avg":
		pool = AvgPool2D
	elif poolmode == "max":
		pool = MaxPool2D
	else:
		raise ValueError("Unsupported pool mode")

	if layers not in {"11", "16", "19"}:
		raise ValueError("Unsupported VGG layers mode")

	if name is None and layers == "11":
		name = "VGG_ILSVRC_11_layers"
	elif name is None and layers == "16":
		name = "VGG_ILSVRC_16_layers"
	elif name is None and layers == "19":
		name = "VGG_ILSVRC_19_layers"

	layers = int(layers)

	net = Sequential(name=name)

	net.append(Conv2D(3, 64, 3, pad=1, initscheme=initscheme, name="conv1_1"))
	net.append(Activation(relu, inplace=actInplace, name="relu1_1"))

	if layers > 11:
		net.append(Conv2D(64, 64, 3, pad=1, initscheme=initscheme, name="conv1_2"))
		net.append(Activation(relu, inplace=actInplace, name="relu1_2"))

	net.append(pool(2, 2, name="pool1"))

	net.append(Conv2D(64, 128, 3, pad=1, initscheme=initscheme, name="conv2_1"))
	net.append(Activation(relu, inplace=actInplace, name="relu2_1"))

	if layers > 11:
		net.append(Conv2D(128, 128, 3, pad=1, initscheme=initscheme, name="conv2_2"))
		net.append(Activation(relu, inplace=actInplace, name="relu2_2"))

	net.append(pool(2, 2, name="pool2"))

	net.append(Conv2D(128, 256, 3, pad=1, initscheme=initscheme, name="conv3_1"))
	net.append(Activation(relu, inplace=actInplace, name="relu3_1"))
	net.append(Conv2D(256, 256, 3, pad=1, initscheme=initscheme, name="conv3_2"))
	net.append(Activation(relu, inplace=actInplace, name="relu3_2"))

	if layers > 11:
		net.append(Conv2D(256, 256, 3, pad=1, initscheme=initscheme, name="conv3_3"))
		net.append(Activation(relu, inplace=actInplace, name="relu3_3"))

	if layers > 16:
		net.append(Conv2D(256, 256, 3, pad=1, initscheme=initscheme, name="conv3_4"))
		net.append(Activation(relu, inplace=actInplace, name="relu3_4"))

	net.append(pool(2, 2, name="pool3"))

	net.append(Conv2D(256, 512, 3, pad=1, initscheme=initscheme, name="conv4_1"))
	net.append(Activation(relu, inplace=actInplace, name="relu4_1"))
	net.append(Conv2D(512, 512, 3, pad=1, initscheme=initscheme, name="conv4_2"))
	net.append(Activation(relu, inplace=actInplace, name="relu4_2"))

	if layers > 11:
		net.append(Conv2D(512, 512, 3, pad=1, initscheme=initscheme, name="conv4_3"))
		net.append(Activation(relu, inplace=actInplace, name="relu4_3"))

	if layers > 16:
		net.append(Conv2D(512, 512, 3, pad=1, initscheme=initscheme, name="conv4_4"))
		net.append(Activation(relu, inplace=actInplace, name="relu4_4"))

	net.append(pool(2, 2, name="pool4"))

	net.append(Conv2D(512, 512, 3, pad=1, initscheme=initscheme, name="conv5_1"))
	net.append(Activation(relu, inplace=actInplace, name="relu5_1"))
	net.append(Conv2D(512, 512, 3, pad=1, initscheme=initscheme, name="conv5_2"))
	net.append(Activation(relu, inplace=actInplace, name="relu5_2"))

	if layers > 11:
		net.append(Conv2D(512, 512, 3, pad=1, initscheme=initscheme, name="conv5_3"))
		net.append(Activation(relu, inplace=actInplace, name="relu5_3"))

	if layers > 16:
		net.append(Conv2D(512, 512, 3, pad=1, initscheme=initscheme, name="conv5_4"))
		net.append(Activation(relu, inplace=actInplace, name="relu5_4"))

	net.append(pool(2, 2, name="pool5"))

	if withLinear:
		net.append(Flatten())

		insize = int(np.prod(net.dataShapeFrom((1, 3, 224, 224))))

		net.append(Linear(insize, 4096, initscheme=initscheme, name="fc6"))
		net.append(Activation(relu, inplace=actInplace, name="relu6"))

		net.append(Linear(4096, 4096, initscheme=initscheme, name="fc7"))
		net.append(Activation(relu, inplace=actInplace, name="relu7"))

		net.append(Linear(4096, 1000, initscheme=initscheme, name="fc8"))
		net.append(SoftMax())

	if modelpath is not None:
		net.load(modelpath)

	return net


def unittest():
	data = gpuarray.to_gpu(np.random.randn(1, 3, 224, 224).astype(np.float32))

	vgg11 = loadVGG(None, layers="11", initscheme="gaussian")
	vgg11(data)

	del vgg11
	memPool.freeHeld()

	vgg16 = loadVGG(None, layers="16", initscheme="gaussian")
	vgg16(data)

	del vgg16
	memPool.freeHeld()

	vgg19 = loadVGG(None, layers="19", initscheme="gaussian")
	vgg19(data)

	del vgg19
	memPool.freeHeld()


if __name__ == "__main__":
	unittest()
