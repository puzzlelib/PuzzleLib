from PuzzleLib.Backend import Dnn

from PuzzleLib.Containers import Sequential
from PuzzleLib.Modules import Embedder, SwapAxes, RNN, Linear

from PuzzleLib.Datasets import IMDBLoader
from PuzzleLib.Handlers import Trainer, Validator
from PuzzleLib.Optimizers import Adam
from PuzzleLib.Cost import BCE


def buildNet(numwords, maxlen, hintBatchsize):
	seq = Sequential()
	seq.append(Embedder(numwords, maxlen, 128, initscheme="uniform", wscale=0.05, learnable=True))

	seq.append(SwapAxes(0, 1))
	seq.append(RNN(128, 128, mode="lstm", dropout=0.2, hintBatchSize=hintBatchsize))

	seq.append(Linear(128, 1))
	return seq


def main():
	hintBatchsize, batchsize = (40, 40) if Dnn.deviceSupportsBatchHint() else (None, 32)
	numwords, maxlen = 20000, 80

	imdb = IMDBLoader(numwords=numwords, maxlen=maxlen)
	data, labels, _ = imdb.load(path="../TestData/")
	data, labels = data[:], labels[:]
	print("Loaded IMDB")

	net = buildNet(numwords, maxlen, hintBatchsize)

	optimizer = Adam(alpha=1e-3)
	optimizer.setupOn(net, useGlobalState=True)

	cost = BCE()
	trainer = Trainer(net, cost, optimizer, batchsize=batchsize)
	validator = Validator(net, cost, batchsize=batchsize)

	print("Started training ...")

	for i in range(15):
		trainer.trainFromHost(
			data[:25000], labels[:25000], macroBatchSize=25000,
			onMacroBatchFinish=lambda train: print("Train error: %s" % train.cost.getMeanError())
		)
		print("Accuracy: %s" % (1.0 - validator.validateFromHost(data[25000:], labels[25000:], macroBatchSize=25000)))


if __name__ == "__main__":
	main()
