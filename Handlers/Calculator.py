import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import copy

from PuzzleLib.Handlers.Handler import Handler


class Calculator(Handler):
	def calcFromHost(self, data, macroBatchSize=10000, onMacroBatchFinish=None):
		state = {"hostSize": self.getDataSize(data)}

		self.module.evalMode()
		self.handleFromHost(data, state, macroBatchSize, onMacroBatchFinish, random=False)
		return state["hostData"]


	def calc(self, data):
		state = {"devSize": self.getDataSize(data)}

		self.module.evalMode()
		self.handle(data, state, random=False)
		return state["devData"]


	def onMacroBatchStart(self, idx, macroBatchSize, state):
		state["devSize"] = macroBatchSize


	def onMacroBatchFinish(self, idx, macroBatchSize, state):
		if not "hostData" in state:
			def reserveHostData(data):
				return np.empty((state["hostSize"], ) + data.shape[1:], dtype=data.dtype)

			state["hostData"] = self.parseShapeTree(state["devData"], onData=reserveHostData)

		def copyHostData(indata, outdata):
			outdata[idx * macroBatchSize:(idx + 1) * macroBatchSize] = indata.get()

		self.parseShapeTree(state["devData"], copyHostData, state["hostData"])
		del state["devData"]


	def handleBatch(self, batch, idx, state):
		outBatch = self.module(batch)

		if not "devData" in state:
			def reserveDevData(data):
				return gpuarray.empty((state["devSize"], ) + data.shape[1:], dtype=data.dtype)

			state["devData"] = self.parseShapeTree(outBatch, onData=reserveDevData)

		def copyDevData(indata, outdata):
			copy(outdata[idx * self.batchsize:(idx + 1) * self.batchsize], indata)

		self.parseShapeTree(outBatch, copyDevData, state["devData"])


def unittest():
	onDeviceTest()
	onHostTest()


def onDeviceTest():
	from PuzzleLib.Containers import Sequential
	from PuzzleLib.Modules import Conv2D, MaxPool2D, Activation, relu, Flatten, Linear

	data = gpuarray.to_gpu(np.random.randn(10000, 3, 28, 28).astype(np.float32))

	seq = Sequential()
	seq.append(Conv2D(3, 16, 9))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Conv2D(16, 32, 5))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Flatten())
	seq.append(Linear(3 * 3 * 32, 10))

	calc = Calculator(seq)
	calc.onBatchFinish = lambda calculator: print("Finished batch #%d" % calculator.currBatch)
	calc.calc(data)


def onHostTest():
	from PuzzleLib.Containers import Sequential
	from PuzzleLib.Modules import Conv2D, MaxPool2D, Activation, relu, Flatten, Linear

	data = np.random.randn(50000, 3, 28, 28).astype(np.float32)

	seq = Sequential()
	seq.append(Conv2D(3, 16, 9))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Conv2D(16, 32, 5))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Flatten())
	seq.append(Linear(3 * 3 * 32, 10))

	calc = Calculator(seq)
	calc.onBatchFinish = lambda calculator: print("Finished batch #%d" % calculator.currBatch)
	calc.calcFromHost(data, onMacroBatchFinish=lambda calculator: print("Finished mb #%d" % calculator.currMacroBatch))


if __name__ == "__main__":
	unittest()
