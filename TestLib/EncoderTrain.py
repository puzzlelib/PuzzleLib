import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.gpuarray import memoryPool as memPool

from PuzzleLib.Containers import Sequential
from PuzzleLib.Modules import Linear, Activation, relu, Dropout

from PuzzleLib.Datasets import MnistLoader
from PuzzleLib.Visual import showFilters
from PuzzleLib.Optimizers import MomentumSGD
from PuzzleLib.Cost import MSE
from PuzzleLib.Variable import Variable


def buildEncoder():
	seq = Sequential()

	seq.append(Linear(784, 256))
	seq.append(Activation(relu, inplace=True))
	seq.append(Dropout())
	seq.append(Linear(256, 784, empty=True, transpose=True))

	seq[-1].setVar("W", seq[0].vars["W"])
	seq[-1].setVar("b", Variable(gpuarray.zeros((784,), dtype=np.float32, allocator=memPool)))

	return seq


def main():
	mnist = MnistLoader()
	data, _ = mnist.load(path="../TestData")
	data = data[:].reshape(data.shape[0], -1)
	print("Loaded mnist")

	np.random.seed(1234)
	net = buildEncoder()

	optimizer = MomentumSGD()
	optimizer.setupOn(net, useGlobalState=True)
	optimizer.learnRate = 10.0
	optimizer.momRate = 0.5

	data = gpuarray.to_gpu(data)
	batchsize = 100

	mse = MSE()

	for epoch in range(40):
		for i in range(data.shape[0] // batchsize):
			batch = data[i * batchsize:(i + 1) * batchsize]

			net(batch)
			_, grad = mse(net.data, batch)

			net.zeroGradParams()
			net.backward(grad)
			optimizer.update()

		optimizer.learnRate *= 0.8
		print("Finished epoch %d" % (epoch + 1))

		print("Error: %s" % (mse.getMeanError()))
		mse.resetAccumulator()

		if (epoch + 1) % 5 == 0:
			filters = net[0].W.get().T
			showFilters(filters.reshape(16, 16, 28, 28), "../TestData/encoder.png")


if __name__ == "__main__":
	main()
