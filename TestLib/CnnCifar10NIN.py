import numpy as np

from PuzzleLib.Containers import Sequential
from PuzzleLib.Modules import Conv2D, Activation, relu, MaxPool2D, Dropout, AvgPool2D, Flatten

from PuzzleLib.Datasets import Cifar10Loader
from PuzzleLib.Visual import showImageBasedFilters, showFilters
from PuzzleLib.Optimizers import Hooks, MomentumSGD
from PuzzleLib.Handlers import Trainer, Validator
from PuzzleLib.Cost import CrossEntropy


def buildNet():
	seq = Sequential(name="cifar")

	seq.append(Conv2D(3, 192, 5, pad=2, initscheme="gaussian", wscale=0.05, name="conv1"))
	seq.append(Activation(relu, name="relu1"))

	seq.append(Conv2D(192, 160, 1, initscheme="gaussian", wscale=0.05, name="cccp1"))
	seq.append(Activation(relu, name="relu_cccp1"))
	seq.append(Conv2D(160, 96, 1, initscheme="gaussian", wscale=0.05, name="cccp2"))
	seq.append(Activation(relu, name="relu_cccp2"))

	seq.append(MaxPool2D(3, 2, pad=1, name="pool1"))
	seq.append(Dropout(name="drop3"))

	seq.append(Conv2D(96, 192, 5, pad=2, initscheme="gaussian", wscale=0.05, name="conv2"))
	seq.append(Activation(relu, name="relu2"))

	seq.append(Conv2D(192, 192, 1, initscheme="gaussian", wscale=0.05, name="cccp3"))
	seq.append(Activation(relu, name="relu_cccp3"))
	seq.append(Conv2D(192, 192, 1, initscheme="gaussian", wscale=0.05, name="cccp4"))
	seq.append(Activation(relu, name="relu_cccp4"))

	seq.append(AvgPool2D(3, 2, pad=1, name="pool2"))
	seq.append(Dropout(name="drop6"))

	seq.append(Conv2D(192, 192, 3, pad=1, initscheme="gaussian", wscale=0.05, name="conv3"))
	seq.append(Activation(relu, name="relu3"))

	seq.append(Conv2D(192, 192, 1, initscheme="gaussian", wscale=0.05, name="cccp5"))
	seq.append(Activation(relu, name="relu_cccp5"))
	seq.append(Conv2D(192, 10, 1, initscheme="gaussian", wscale=0.05, name="cccp6"))
	seq.append(Activation(relu, name="relu_cccp6"))

	seq.append(AvgPool2D(8, 1, name="pool3"))
	seq.append(Flatten())

	return seq


def main():
	cifar10 = Cifar10Loader()
	data, labels = cifar10.load(path="../TestData/")
	data, labels = data[:], labels[:]
	print("Loaded cifar10")

	data = data.reshape(data.shape[0], -1)
	data -= np.mean(data, axis=0, keepdims=True) + 1e-8
	data /= np.std(data, axis=0, keepdims=True) + 1e-5

	data = data.reshape(data.shape[0], 3, 32, 32)

	np.random.seed(1234)
	net = buildNet()

	optimizer = MomentumSGD(learnRate=0.1, momRate=0.9)
	optimizer.addHook(Hooks.WeightDecay(0.0001))
	optimizer.setupOn(net, useGlobalState=True)

	cost = CrossEntropy(maxlabels=10)
	trainer = Trainer(
		net, cost, optimizer,
		onBatchFinish=lambda train: print("Processed batch %d out of %d" % (train.currBatch, train.totalBatches))
	)

	validator = Validator(net, cost)
	epochs = 100

	for i in range(epochs):
		trainer.trainFromHost(
			data[:50000], labels[:50000], macroBatchSize=25000,
			onMacroBatchFinish=lambda train: print("Train error: %s" % train.cost.getMeanError())
		)
		valerror = validator.validateFromHost(data[50000:], labels[50000:], macroBatchSize=10000)
		print("Finished epoch %d out of %d. Val error: %s" % (i + 1, epochs, valerror))

		if i + 1 == 60 or i + 1 == 80:
			optimizer.learnRate *= 0.1
			print("Lowered learn rate: %s" % optimizer.learnRate)

		showImageBasedFilters(net["conv1"].W.get(), "../TestData/ninconv1.png")
		showFilters(net["conv2"].W.get(), "../TestData/ninconv2.png")
		showFilters(net["conv3"].W.get(), "../TestData/ninconv3.png")


if __name__ == "__main__":
	main()
