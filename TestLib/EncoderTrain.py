from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool

from PuzzleLib.Datasets import MnistLoader

from PuzzleLib.Containers import *
from PuzzleLib.Modules import *
from PuzzleLib.Optimizers import MomentumSGD
from PuzzleLib.Cost import MSE
from PuzzleLib.Variable import Variable

from PuzzleLib.Visual import *


def main():
	mnist = MnistLoader()
	data, _ = mnist.load(path="../TestData")
	data = data[:].reshape(data.shape[0], np.prod(data.shape[1:]))
	print("Loaded mnist")

	np.random.seed(1234)

	seq = Sequential()
	seq.append(Linear(784, 256))
	seq.append(Activation(relu, inplace=True))
	seq.append(Dropout())
	seq.append(Linear(256, 784, empty=True, transpose=True))

	seq[-1].setVar("W", seq[0].vars["W"])
	seq[-1].setVar("b", Variable(gpuarray.zeros((784, ), dtype=np.float32, allocator=memPool)))

	optimizer = MomentumSGD()
	optimizer.setupOn(seq, useGlobalState=True)
	optimizer.learnRate = 10.0
	optimizer.momRate = 0.5

	data = gpuarray.to_gpu(data)
	batchsize = 100

	mse = MSE()

	for epoch in range(40):
		for i in range(data.shape[0] // batchsize):
			batch = data[i * batchsize:(i + 1) * batchsize]

			seq(batch)
			_, grad = mse(seq.data, batch)

			seq.zeroGradParams()
			seq.backward(grad)
			optimizer.update()

		optimizer.learnRate *= 0.8
		print("Finished epoch %d" % (epoch + 1))

		print("Error: %s" % (mse.getMeanError()))
		mse.resetAccumulator()

		if (epoch + 1) % 5 == 0:
			filters = seq[0].W.get().T
			showImageBatchInFolder(filters.reshape(256, 1, 28, 28), "../TestData/encoder/", "filter")


if __name__ == "__main__":
	main()
