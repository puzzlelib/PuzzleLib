import numpy as np

from PuzzleLib.Backend.Benchmarks import timeKernel


def scoreModels(net, engine, data, labels):
	hostNetData = net(data).get()
	hostEngineData = engine(data).get()

	assert np.allclose(hostNetData, hostEngineData, atol=1e-6)

	printResults(hostNetData, labels, "Net")
	printResults(hostEngineData, labels, "Engine")


def printResults(probs, labels, name):
	probs = probs.flatten()

	idx = (-probs).argsort()[:5]
	print("%s top-5 predictions: " % name)

	for i in range(5):
		print("#%s %s (prob=%s)" % (i, labels[idx[i]], probs[idx[i]]))


def benchModels(net, engine, data):
	net.optimizeForShape(data.shape)

	nettime = timeKernel(net, args=(data, ), looplength=100, log=False, normalize=True)
	enginetime = timeKernel(engine, args=(data, ), looplength=100, log=False, normalize=True)

	print("Net    time: host=%.10f" % nettime)
	print("Engine time: host=%.10f" % enginetime)
