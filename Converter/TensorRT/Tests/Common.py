import numpy as np

from PuzzleLib.Backend.Benchmarks import timeKernel


def scoreModels(net, engine, data, labels):
	hostNetData = net(data).get()
	hostEngineData = engine(data).get()

	assert np.allclose(hostNetData, hostEngineData)

	printResults(hostNetData, labels, "Net")
	printResults(hostEngineData, labels, "Engine")


def printResults(probs, labels, name):
	probs = probs.flatten()

	idx = (-probs).argsort()[:5]
	print("%s top-5 predictions: " % name)

	for i in range(5):
		print("#%s %s (prob=%s)" % (i, labels[idx[i]], probs[idx[i]]))


def benchModels(net, engine, data, lognames=None):
	netlabel, enginelabel = ("Net   ", "Engine") if lognames is None else lognames

	net.optimizeForShape(data.shape)

	nettime = timeKernel(net, args=(data, ), looplength=100, log=False, normalize=True)
	enginetime = timeKernel(engine, args=(data, ), looplength=100, log=False, normalize=True)

	print("%s time: device=%.10f host=%.10f" % (netlabel, nettime[0], nettime[1]))
	print("%s time: device=%.10f host=%.10f" % (enginelabel, enginetime[0], enginetime[1]))
