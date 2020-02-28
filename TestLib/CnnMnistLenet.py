from PuzzleLib.Datasets import MnistLoader

from PuzzleLib.Containers import *
from PuzzleLib.Modules import *
from PuzzleLib.Handlers import *
from PuzzleLib.Optimizers import MomentumSGD
from PuzzleLib.Cost import CrossEntropy

from PuzzleLib.Visual import *


def main():
	mnist = MnistLoader()
	data, labels = mnist.load(path="../TestData/")
	data, labels = data[:], labels[:]
	print("Loaded mnist")

	np.random.seed(1234)

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

	optimizer = MomentumSGD()
	optimizer.setupOn(seq, useGlobalState=True)
	optimizer.learnRate = 0.1
	optimizer.momRate = 0.9

	cost = CrossEntropy(maxlabels=10)
	trainer = Trainer(seq, cost, optimizer)
	validator = Validator(seq, cost)

	for i in range(15):
		trainer.trainFromHost(data[:60000], labels[:60000], macroBatchSize=60000,
							  onMacroBatchFinish=lambda train: print("Train error: %s" % train.cost.getMeanError()))
		print("Accuracy: %s" % (1.0 - validator.validateFromHost(data[60000:], labels[60000:], macroBatchSize=10000)))

		optimizer.learnRate *= 0.9

		showFilters(seq[0].W.get(), "../TestData/conv1.png")
		showFilters(seq[3].W.get(), "../TestData/conv2.png")


if __name__ == "__main__":
	main()
