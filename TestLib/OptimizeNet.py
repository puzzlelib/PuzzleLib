import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Benchmarks import timeKernel

from PuzzleLib.Models.Nets.VGG import loadVGG

from PuzzleLib.Optimizers import SGD
from PuzzleLib.Cost import CrossEntropy
from PuzzleLib.Handlers import Trainer


def main():
	net = loadVGG(None, "16")

	batchsize = 16
	size = (batchsize, 3, 224, 224)

	batch = np.random.normal(size=size).astype(dtype=np.float32)
	batch = gpuarray.to_gpu(batch)

	labels = np.random.randint(low=0, high=1000, size=(batchsize, ), dtype=np.int32)
	labels = gpuarray.to_gpu(labels)

	optimizer = SGD()
	optimizer.setupOn(net)

	cost = CrossEntropy(maxlabels=1000)
	trainer = Trainer(net, cost, optimizer)

	print("Started benchmarking %s ..." % net.name)
	timeKernel(
		trainer.train, args=(batch, labels), looplength=100, logname="Before optimizing %s" % net.name, normalize=True
	)

	net.optimizeForShape(size)
	timeKernel(
		trainer.train, args=(batch, labels), looplength=100, logname="After optimizing %s" % net.name, normalize=True
	)


if __name__ == "__main__":
	main()
