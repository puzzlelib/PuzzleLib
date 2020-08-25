import time

import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Containers.Sequential import Sequential
from PuzzleLib.Containers.Parallel import Parallel

from PuzzleLib.Modules.Embedder import Embedder
from PuzzleLib.Modules.Reshape import Reshape
from PuzzleLib.Modules.Replicate import Replicate
from PuzzleLib.Modules.Conv2D import Conv2D
from PuzzleLib.Modules.MaxPool2D import MaxPool2D
from PuzzleLib.Modules.Concat import Concat
from PuzzleLib.Modules.Activation import Activation, relu
from PuzzleLib.Modules.Dropout import Dropout
from PuzzleLib.Modules.Linear import Linear


def buildBranch(fHeight, sentlength, branchMaps, embsize):
	seq = Sequential()

	seq.append(Conv2D(1, outmaps=branchMaps, size=(fHeight, embsize)))
	seq.append(MaxPool2D(size=(sentlength - fHeight + 1, 1)))
	seq.append(Reshape((-1, branchMaps)))

	return seq


def buildNet(vocabulary, branches, w2v, sentlength, embsize, wscale, dim=2, branchMaps=100, name="sentinet"):
	def onVocabulary(W):
		W[0] = np.zeros((1, embsize), dtype=np.float32)

		arrayPOS = [
			"", "_S", "_A", "_V", "_UNKN", "_ADJ", "_ADV", "_INTJ", "_NOUN", "_PROPN", "_VERB", "_ADP",
			"_AUX", "_CCONJ", "_DET", "_NUM", "_PART", "_PRON", "_SCONJ", "_SUM", "_X"
		]

		tmpPOS = []
		if not w2v:
			return

		for word in vocabulary:
			for pos in tmpPOS:
				if (word + pos) in w2v.vocab:
					W[vocabulary[word]] = w2v[word + pos]
					break

			for i, pos in enumerate(arrayPOS):
				if (word + pos) in w2v.vocab:
					tmpPOS.append(pos)
					W[vocabulary[word]] = w2v[word + pos]
					del arrayPOS[i]
					break

	net = Sequential(name)
	net.setAttr("timestamp", int(time.time()))

	net.append(Embedder(
		vocabulary, sentlength, embsize, wscale=wscale, onVocabulary=onVocabulary, learnable=True, name="embedder"
	))

	net.append(Reshape((-1, 1, sentlength, embsize)))

	branchNum = len(branches)
	net.append(Replicate(times=branchNum))

	par = Parallel()

	for branchFilterSize in branches:
		par.append(buildBranch(branchFilterSize, sentlength, branchMaps, embsize))

	net.append(par)
	net.append(Concat(axis=1))
	net.append(Activation(relu))
	net.append(Dropout(p=0.5))

	net.append(Linear(branchNum * branchMaps, dim))

	return net


def unittest():
	vocabsize = 1000
	sentlength, embsize = 100, 128

	data = gpuarray.to_gpu(np.random.randint(0, vocabsize, (1, sentlength), dtype=np.int32))

	senti = buildNet(vocabsize, (3, 5, 7), None, sentlength, embsize, 1.0)
	senti(data)

	del senti
	gpuarray.memoryPool.freeHeld()


if __name__ == "__main__":
	unittest()
