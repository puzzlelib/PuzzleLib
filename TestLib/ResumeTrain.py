import os

import numpy as np

from PuzzleLib.Datasets import MnistLoader
from PuzzleLib.Handlers import Trainer, Validator
from PuzzleLib.Optimizers import MomentumSGD
from PuzzleLib.Cost import CrossEntropy

from PuzzleLib.Models.Nets.LeNet import loadLeNet


def train(net, optimizer, data, labels, epochs):
	cost = CrossEntropy(maxlabels=10)
	trainer = Trainer(net, cost, optimizer)
	validator = Validator(net, cost)

	for i in range(epochs):
		trainer.trainFromHost(
			data[:60000], labels[:60000], macroBatchSize=60000,
			onMacroBatchFinish=lambda tr: print("Train error: %s" % tr.cost.getMeanError())
		)
		print("Accuracy: %s" % (1.0 - validator.validateFromHost(data[60000:], labels[60000:], macroBatchSize=10000)))

		optimizer.learnRate *= 0.9
		print("Reduced optimizer learn rate to %s" % optimizer.learnRate)


def main():
	mnist = MnistLoader()
	data, labels = mnist.load(path="../TestData/")
	data, labels = data[:], labels[:]
	print("Loaded mnist")

	np.random.seed(1234)
	net = loadLeNet(None, initscheme=None)

	optimizer = MomentumSGD()
	optimizer.setupOn(net, useGlobalState=True)
	optimizer.learnRate = 0.1
	optimizer.momRate = 0.9

	epochs = 10
	print("Training for %s epochs ..." % epochs)
	train(net, optimizer, data, labels, epochs)

	print("Saving net and optimizer ...")
	net.save("../TestData/net.hdf")
	optimizer.save("../TestData/optimizer.hdf")

	print("Reloading net and optimizer ...")
	net.load("../TestData/net.hdf")
	optimizer.load("../TestData/optimizer.hdf")

	print("Continuing training for %s epochs ..." % epochs)
	train(net, optimizer, data, labels, epochs)

	os.remove("../TestData/net.hdf")
	os.remove("../TestData/optimizer.hdf")


if __name__ == "__main__":
	main()
