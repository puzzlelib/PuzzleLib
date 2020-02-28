import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Handlers.Handler import Handler


class Trainer(Handler):
	def __init__(self, mod, cost, optimizer, onBatchFinish=None, batchsize=128):
		super().__init__(mod, onBatchFinish, batchsize)
		self.cost = cost
		self.optimizer = optimizer


	def trainFromHost(self, data, target, macroBatchSize=10000, onMacroBatchFinish=None, random=True):
		self.cost.resetAccumulator()

		self.module.trainMode()
		self.handleFromHost([data, target], None, macroBatchSize, onMacroBatchFinish, random=random)


	def train(self, data, target, random=True):
		self.cost.resetAccumulator()

		self.module.trainMode()
		self.handle([data, target], None, random=random)


	def handleBatch(self, batch, idx, state):
		data, target = batch

		grad = self.cost(self.module(data), target, queryError=False)

		self.optimizer.zeroGradParams()
		self.module.backward(grad, updGrad=False)
		self.optimizer.update()


def unittest():
	onDeviceTest()
	onHostTest()


def onDeviceTest():
	from PuzzleLib.Containers import Sequential
	from PuzzleLib.Modules import Conv2D, MaxPool2D, Activation, relu, Flatten, Linear

	from PuzzleLib.Cost.CrossEntropy import CrossEntropy
	from PuzzleLib.Optimizers.NesterovSGD import NesterovSGD

	data = gpuarray.to_gpu(np.random.randn(10000, 3, 28, 28).astype(np.float32))
	dataTarget = gpuarray.to_gpu(np.random.randint(low=0, high=10, size=(10000, )).astype(np.int32))

	seq = Sequential()
	seq.append(Conv2D(3, 16, 9))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Conv2D(16, 32, 5))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Flatten())
	seq.append(Linear(3 * 3 * 32, 10))

	entr = CrossEntropy()

	opt = NesterovSGD()
	opt.setupOn(seq)

	def onBatchFinish(train):
		print("Finished batch #%d, error=%s" % (train.currBatch, train.cost.getError()))

	trainer = Trainer(seq, entr, opt, onBatchFinish=onBatchFinish)
	trainer.train(data, dataTarget)


def onHostTest():
	from PuzzleLib.Containers import Sequential
	from PuzzleLib.Modules import Conv2D, MaxPool2D, Activation, relu, Flatten, Linear

	from PuzzleLib.Cost.CrossEntropy import CrossEntropy
	from PuzzleLib.Optimizers.NesterovSGD import NesterovSGD

	data = np.random.randn(50000, 3, 28, 28).astype(np.float32)
	dataTarget = np.random.randint(low=0, high=10, size=(50000, )).astype(np.int32)

	seq = Sequential()
	seq.append(Conv2D(3, 16, 9))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Conv2D(16, 32, 5))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Flatten())
	seq.append(Linear(3 * 3 * 32, 10))

	entr = CrossEntropy()

	opt = NesterovSGD()
	opt.setupOn(seq)

	def onMacroBatchFinish(train):
		print("Finished mb #%d, error=%s" % (train.currMacroBatch, train.cost.getMeanError()))

	trainer = Trainer(seq, entr, opt)
	trainer.trainFromHost(data, dataTarget, onMacroBatchFinish=onMacroBatchFinish)


if __name__ == "__main__":
	unittest()
