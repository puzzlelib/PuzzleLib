from PuzzleLib.Datasets import IMDBLoader

from PuzzleLib.Containers import *
from PuzzleLib.Modules import *
from PuzzleLib.Handlers import *
from PuzzleLib.Optimizers import Adam
from PuzzleLib.Cost import BCE


def main():
	numwords, maxlen, embsize = 5000, 250, 50

	imdb = IMDBLoader(numwords=numwords, maxlen=maxlen)
	data, labels, _ = imdb.load(path="../TestData/")
	data, labels = data[:], labels[:]
	print("Loaded IMDB")

	seq = Sequential()

	seq.append(Embedder(numwords, maxlen, embsize, initscheme="uniform", wscale=0.05, learnable=True))

	seq.append(Dropout(p=0.2))
	seq.append(SwapAxes(1, 2))

	seq.append(Conv1D(embsize, embsize, 3))
	seq.append(Activation(relu))

	seq.append(MaxPool1D(maxlen - 2, 1))
	seq.append(Flatten())

	seq.append(Linear(embsize, 250))
	seq.append(Dropout(p=0.2))
	seq.append(Activation(relu))

	seq.append(Linear(250, 1))

	optimizer = Adam(alpha=1e-3)
	optimizer.setupOn(seq, useGlobalState=True)

	cost = BCE()
	trainer = Trainer(seq, cost, optimizer, batchsize=32)
	validator = Validator(seq, cost, batchsize=32)

	for i in range(15):
		trainer.trainFromHost(data[:25000], labels[:25000], macroBatchSize=25000,
							  onMacroBatchFinish=lambda train: print("Train error: %s" % train.cost.getMeanError()))
		print("Accuracy: %s" % (1.0 - validator.validateFromHost(data[25000:], labels[25000:], macroBatchSize=25000)))


if __name__ == "__main__":
	main()
