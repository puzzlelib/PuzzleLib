import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Datasets import MnistLoader

from PuzzleLib.Containers import *
from PuzzleLib.Modules import *
from PuzzleLib.Handlers import *
from PuzzleLib.Optimizers import MomentumSGD
from PuzzleLib.Cost import CrossEntropy

from PuzzleLib.Converter.TensorRT.Tests.Common import benchModels
from PuzzleLib.Converter.TensorRT.BuildRTEngine import buildRTEngine, DataType
from PuzzleLib.Converter.TensorRT.DataCalibrator import DataCalibrator


def buildNet():
	seq = Sequential(name="lenet-5-like")
	seq.append(Conv2D(1, 16, 3))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Conv2D(16, 32, 4))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Flatten())
	seq.append(Linear(32 * 5 * 5, 1024))
	seq.append(Activation(relu))

	seq.append(Linear(1024, 10))

	return seq


def trainNet(net, data, labels, epochs):
	optimizer = MomentumSGD()
	optimizer.setupOn(net, useGlobalState=True)
	optimizer.learnRate = 0.1
	optimizer.momRate = 0.9

	cost = CrossEntropy(maxlabels=10)
	trainer = Trainer(net, cost, optimizer)
	validator = Validator(net, cost)

	for i in range(epochs):
		trainer.trainFromHost(
			data[:60000], labels[:60000], macroBatchSize=60000,
			onMacroBatchFinish=lambda train: print("Train error: %s" % train.cost.getMeanError())
		)
		print("Accuracy: %s" % (1.0 - validator.validateFromHost(data[60000:], labels[60000:], macroBatchSize=10000)))

		optimizer.learnRate *= 0.9


def validate(net, data, labels, batchsize=1):
	cost = CrossEntropy(maxlabels=10)
	validator = Validator(net, cost, batchsize=batchsize)

	return 1.0 - validator.validateFromHost(data[60000:], labels[60000:], macroBatchSize=10000)


def main():
	mnist = MnistLoader()
	data, labels = mnist.load(path="../TestData/")
	data, labels = data[:], labels[:]
	print("Loaded mnist")

	np.random.seed(1234)

	net = buildNet()
	trainNet(net, data, labels, 15)

	calibrator = DataCalibrator(data[:60000])
	net.evalMode()

	engine = buildRTEngine(
		net, inshape=data[:1].shape, savepath="../TestData", dtype=DataType.int8, calibrator=calibrator
	)

	benchModels(net, engine, gpuarray.to_gpu(data[:1]))

	print("Net    accuracy: %s" % validate(net, data, labels))
	print("Engine accuracy: %s" % validate(engine, data, labels, batchsize=1))


if __name__ == "__main__":
	main()
