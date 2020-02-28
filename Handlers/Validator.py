import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Handlers.Handler import Handler


class Validator(Handler):
	def __init__(self, mod, cost, onBatchFinish=None, batchsize=128):
		super().__init__(mod, onBatchFinish, batchsize)
		self.error = 0.0
		self.cost = cost


	def validateFromHost(self, data, target, macroBatchSize=10000, onMacroBatchFinish=None):
		nstates = len(target) if isinstance(target, list) else 1
		state = {"error": [0.0] * nstates}

		self.module.evalMode()
		self.handleFromHost([data, target], state, macroBatchSize, onMacroBatchFinish, random=False)

		error = [error / self.getDataSize(target) for error in state["error"]]
		self.error = error if isinstance(target, list) else error[0]

		return self.error


	def validate(self, data, target):
		nstates = len(target) if isinstance(target, list) else 1
		state = {"error": [0.0] * nstates}

		self.module.evalMode()
		self.handle([data, target], state, random=False)

		error = [error / self.getDataSize(target) for error in state["error"]]
		self.error = error if isinstance(target, list) else error[0]

		return self.error


	def handleBatch(self, batch, idx, state):
		data, target = batch
		error = state["error"]

		batchError = self.cost.validate(self.module(data), target)
		batchError = batchError if isinstance(batchError, list) else [batchError]

		for i in range(len(error)):
			error[i] += self.getDataSize(data) * batchError[i]


def unittest():
	onDeviceTest()
	onHostTest()


def onDeviceTest():
	from PuzzleLib.Containers import Sequential
	from PuzzleLib.Modules import Conv2D, MaxPool2D, Activation, relu, Flatten, Linear

	from PuzzleLib.Cost.CrossEntropy import CrossEntropy

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

	val = Validator(seq, entr)
	print("Validation error on small data: %s" % val.validate(data, dataTarget))


def onHostTest():
	from PuzzleLib.Containers import Sequential
	from PuzzleLib.Modules import Conv2D, MaxPool2D, Activation, relu, Flatten, Linear

	from PuzzleLib.Cost.CrossEntropy import CrossEntropy

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

	val = Validator(seq, entr)
	val.validateFromHost(data, dataTarget)
	print("Validation error on big data: %s" % val.error)


if __name__ == "__main__":
	unittest()
