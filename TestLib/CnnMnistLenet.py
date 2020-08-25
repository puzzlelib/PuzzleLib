import numpy as np

from PuzzleLib.Datasets import MnistLoader
from PuzzleLib.Visual import showFilters
from PuzzleLib.Handlers import Trainer, Validator
from PuzzleLib.Optimizers import MomentumSGD
from PuzzleLib.Cost import CrossEntropy

from PuzzleLib.Models.Nets.LeNet import loadLeNet


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

	cost = CrossEntropy(maxlabels=10)
	trainer = Trainer(net, cost, optimizer)
	validator = Validator(net, cost)

	for i in range(15):
		trainer.trainFromHost(
			data[:60000], labels[:60000], macroBatchSize=60000,
			onMacroBatchFinish=lambda train: print("Train error: %s" % train.cost.getMeanError())
		)
		print("Accuracy: %s" % (1.0 - validator.validateFromHost(data[60000:], labels[60000:], macroBatchSize=10000)))

		optimizer.learnRate *= 0.9

		showFilters(net[0].W.get(), "../TestData/conv1.png")
		showFilters(net[3].W.get(), "../TestData/conv2.png")


if __name__ == "__main__":
	main()
