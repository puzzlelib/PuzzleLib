import tempfile, os

import numpy as np

from PuzzleLib.Models.Nets.SentiNet import buildNet
from PuzzleLib.Cost.CrossEntropy import CrossEntropy
from PuzzleLib.Optimizers.AdaDelta import AdaDelta

from PuzzleLib.Handlers.Trainer import Trainer
from PuzzleLib.Handlers.Validator import Validator

from PuzzleLib.Datasets.Utils import validate, getDim, splitData, replicateData


def train(net, trainData, trainLabels, valData, valLabels, dim=0, epochs=50, epochsBeforeSaving=0, saving=True,
		  printing=True, macroBatchSize=30000, optimizeNet=True):
	if dim == 0:
		dim = getDim(trainLabels)

	numOfChunks = 1
	batchsize = 64

	if printing:
		print("Batchsize: %d" % batchsize)
		print("Num of chunks: %d" % numOfChunks)

	macroBatchSize = min(len(trainLabels), macroBatchSize)

	optimizer = AdaDelta()
	optimizer.setupOn(net)

	cost = CrossEntropy(dim)

	trainer = Trainer(net, cost, optimizer, batchsize=batchsize)
	validator = Validator(net, cost)

	if optimizeNet:
		net.optimizeForShape((batchsize, *trainData.shape[1:]))

	lowestValerror = np.inf
	valerror = np.inf

	for epoch in range(epochs):
		trainSize = trainData.shape[0]
		chunkSize = trainSize // numOfChunks

		for j in range(numOfChunks + 1):
			start = j * chunkSize
			end = min((j + 1) * chunkSize, trainSize)

			if start == end:
				continue

			trainer.trainFromHost(trainData[start:end], trainLabels[start:end], macroBatchSize=macroBatchSize)
			valerror = validator.validateFromHost(valData, valLabels, macroBatchSize=macroBatchSize)

			if printing:
				trainerror = trainer.cost.getMeanError()

				print("Epoch #%d/%d. Chunk #%d/%d. Train error: %s. Val error: %s" % (
					epoch + 1, epochs, j + 1, numOfChunks, trainerror, valerror))

			if lowestValerror >= valerror and epoch >= epochsBeforeSaving:
				lowestValerror = valerror

				if saving:
					net.save(os.path.join(tempfile.gettempdir(), net.name + ".hdf"))

					if printing:
						print("Net saved for %d epoch. Validation accuracy: %-6f%%" % (
							epoch + 1, 100.0 * (1.0 - valerror)))

		if printing:
			print("Finished epoch #%02d. Total mean error: %8f. Validation accuracy: %-6f%%\n" % (
				epoch + 1, cost.getMeanError(), 100.0 * (1.0 - valerror)))

	bestPrecision = 1.0 - lowestValerror

	if printing:
		print("Highest accuracy: %-6f%%\n" % (100.0 * bestPrecision))

	if saving:
		net.load(os.path.join(tempfile.gettempdir(), net.name + ".hdf"))
		return net, bestPrecision
	else:
		return None, bestPrecision


def buildTrainValidate(data, labels, vocabulary=None, w2v=None, wscale=0.25, embsize=300, padding=4, dim=2,
					   sentlength=100, epochs=5, epochsBeforeSaving=0, branches=(3, 4, 5), saving=True, printing=True):
	data = np.asarray(data.copy())
	labels = np.asarray(labels.copy())

	# data = np.asarray(data)
	# labels = np.asarray(labels)

	trainData, valData, trainLabels, valLabels = splitData(data, labels, validation=0.1, dim=dim)
	trainData, trainLabels = replicateData(trainData, trainLabels, dim=dim)

	if printing:
		print("Train data amount: %d" % trainData.shape[0])
		print("Validation data amount: %d\n" % valData.shape[0])

	net = buildNet(vocabulary, branches, w2v, sentlength + 2 * padding, embsize, wscale, dim=dim)

	net.setAttr("sentlength", sentlength)
	net.setAttr("padding", padding)

	if printing:
		print("Starting training ...")

	net, accuracy = train(
		net, trainData, trainLabels, valData, valLabels, dim, epochs, epochsBeforeSaving, saving, printing
	)

	if net:
		_, _, accuracy = validate(net, valData, valLabels, dim, log=printing)

	return accuracy, net, trainData, valData, trainLabels, valLabels


def unittest():
	vocabsize= 1000
	sentlength = 100

	data = np.random.randint(0, vocabsize, (10000, sentlength), dtype=np.int32)
	labels = np.random.randint(0, 2, (10000, ), dtype=np.int32)

	buildTrainValidate(data, labels, vocabsize, padding=0, embsize=64, epochs=15, saving=False)


if __name__ == "__main__":
	unittest()
