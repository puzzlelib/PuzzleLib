import numpy as np

from PuzzleLib import Visual


def loadV3Labels(filename):
	with open(filename) as f:
		synsets = f.readlines()
		synsets = [line.strip() for line in synsets]

	labels = {}
	for i, synset in enumerate(synsets):
		labels[i] = synset

	return labels


def loadLabels(synpath, wordpath):
	with open(synpath) as f:
		synsets = f.readlines()
		synsets = [line.strip() for line in synsets]

	with open(wordpath) as f:
		lines = f.readlines()
		lines = [line.strip() for line in lines]

	words = {}
	for line in lines:
		tags = line.split(sep=" ", maxsplit=1)
		words[tags[0]] = tags[1]

	labels = {}
	for i, synset in enumerate(synsets):
		labels[i] = words[synset]

	return labels


def showLabelResults(res, labels, limit=5, header=""):
	idx = (-res).argsort()[:limit]

	print("%sTop-%s predictions:" % ("%s " % header if len(header) > 0 else "", limit))
	for i in range(limit):
		print("#%s %s (prob=%s)" % (i + 1, labels[idx[i]], res[idx[i]]))


def loadVGGSample(filename, shape=None, normalize=False):
	meanPixel = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((1, 3, 1, 1))
	sample = loadSample(filename, shape) - meanPixel

	return sample * (2.0 / 255.0) - 1.0 if normalize else sample


def loadResNetSample(net, filename, shape=None):
	mean = net.getAttr("mean")
	return loadSample(filename, shape) - mean


def loadSample(filename, shape=None):
	return Visual.loadImage(filename, shape, normalize=False)[:, ::-1, :, :].astype(np.float32)
