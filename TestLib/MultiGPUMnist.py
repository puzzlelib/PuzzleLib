import numpy as np

from PuzzleLib.Grid import runGrid


def train(nodeinfo, verbose):
	from PuzzleLib.Datasets import MnistLoader
	mnist = MnistLoader(cachename="mnist-%s.hdf" % nodeinfo.index)
	data, labels = mnist.load(path="../TestData/")

	data, labels = data[:], labels[:]
	print("[%s]: Loaded mnist" % nodeinfo.index)

	np.random.seed(1234)

	from PuzzleLib.Models.Nets.LeNet import loadLeNet
	net = loadLeNet(None, initscheme=None)

	from PuzzleLib.Optimizers import MomentumSGD
	optimizer = MomentumSGD(learnRate=0.1, momRate=0.9, nodeinfo=nodeinfo)
	optimizer.setupOn(net, useGlobalState=True)

	from PuzzleLib.Cost import CrossEntropy
	cost = CrossEntropy(maxlabels=10)

	from PuzzleLib.Handlers import Trainer, Validator
	trainer = Trainer(net, cost, optimizer, batchsize=128 // nodeinfo.gridsize)
	validator = Validator(net, cost)

	valsize = 10000
	trainsize = data.shape[0] - valsize

	trainpart = trainsize // nodeinfo.gridsize
	valpart = valsize // nodeinfo.gridsize

	for i in range(15):
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

		optimizer.learnRate *= 0.9


def main():
	runGrid(target=train, size=2, verbose=True)


if __name__ == "__main__":
	main()
