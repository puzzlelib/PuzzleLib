import numpy as np

from PuzzleLib.Grid import runGrid


def train(nodeinfo, verbose):
	from PuzzleLib.Datasets import Cifar10Loader
	cifar10 = Cifar10Loader(cachename="cifar10-%s.hdf" % nodeinfo.index)
	data, labels = cifar10.load(path="../TestData/")

	data, labels = data[:], labels[:]
	print("[%s]: Loaded cifar10" % nodeinfo.index)

	np.random.seed(1234)

	from PuzzleLib.TestLib.CnnCifar10Simple import buildNet
	net = buildNet()

	from PuzzleLib.Optimizers import MomentumSGD
	optimizer = MomentumSGD(learnRate=0.01, momRate=0.9, nodeinfo=nodeinfo)
	optimizer.setupOn(net, useGlobalState=True)

	from PuzzleLib.Cost import CrossEntropy
	cost = CrossEntropy(maxlabels=10)

	from PuzzleLib.Handlers import Trainer, Validator
	trainer = Trainer(net, cost, optimizer, batchsize=128 // nodeinfo.gridsize)
	validator = Validator(net, cost)

	import math
	currerror = math.inf

	valsize = 10000
	trainsize = data.shape[0] - valsize

	trainpart = trainsize // nodeinfo.gridsize
	valpart = valsize // nodeinfo.gridsize

	for i in range(25):
		start, end = nodeinfo.index * trainpart, (nodeinfo.index + 1) * trainpart

		trainer.trainFromHost(data[start:end], labels[start:end], macroBatchSize=trainpart)
		trerr = cost.getMeanError()

		if verbose:
			print("[%s]: Epoch %s local train error: %s" % (nodeinfo.index, i + 1, trerr))

		trerr = nodeinfo.meanValue(trerr)

		if nodeinfo.index == 0:
			print("Epoch %s global train error: %s" % (i + 1, trerr))

		start, end = trainsize + nodeinfo.index * valpart, trainsize + (nodeinfo.index + 1) * valpart
		valerr = validator.validateFromHost(data[start:end], labels[start:end], macroBatchSize=valpart)

		if verbose:
			print("[%s]: Epoch %s local accuracy: %s" % (nodeinfo.index, i + 1, 1.0 - valerr))

		valerr = nodeinfo.meanValue(valerr)

		if nodeinfo.index == 0:
			print("Epoch %s global accuracy: %s" % (i + 1, 1.0 - valerr))

		if valerr >= currerror:
			optimizer.learnRate *= 0.5
			print("[%s]: Lowered learn rate: %s" % (nodeinfo.index, optimizer.learnRate))

		currerror = valerr


def main():
	runGrid(target=train, size=2, verbose=True)


if __name__ == "__main__":
	main()
