import os, json

import numpy as np
import h5py

from PuzzleLib.Datasets.DataLoader import DataLoader


class IMDBLoader(DataLoader):
	def __init__(self, numwords=None, skiptop=0, maxlen=None, padchar=0, startchar=1, oovchar=2, indexFrom=3):
		super().__init__(("data", "labels", "vocabulary"), "imdb.hdf")

		self.numwords = numwords
		self.skiptop = skiptop

		self.maxlen = maxlen

		self.padchar = padchar
		self.startchar = startchar
		self.oovchar = oovchar
		self.indexFrom = indexFrom

		self.datafile = "imdb.npz"
		self.indexfile = "imdb_word_index.json"


	def checkCacheParams(self, log=True):
		if os.path.exists(self.cachename):
			with h5py.File(self.cachename, "r") as hdf:
				params = json.loads(str(np.array(hdf["params"])))

				for paramName in ["numwords", "skiptop", "maxlen", "padchar", "startchar", "oovchar", "indexFrom"]:
					if params[paramName] != getattr(self, paramName):
						if log:
							print("[%s] Existing cache has different param '%s', clearing ..." %
								  (self.__class__.__name__, paramName))

						return False

		return True


	def loadVocabulary(self, path):
		with open(os.path.join(path, self.indexfile)) as f:
			d = json.load(f)

		dt = h5py.special_dtype(vlen=str)
		vocab = np.empty(shape=(self.numwords, ), dtype=dt)

		for word, idx in d.items():
			if idx < self.numwords:
				vocab[int(idx)] = word

		return vocab


	def load(self, path, compress="gzip", log=True):
		self.cachename = os.path.join(path, self.cachename)

		if not self.checkCacheParams():
			self.clear()

		if not os.path.exists(self.cachename):
			if log:
				print("[%s] Started unpacking ..." % self.__class__.__name__)

			with np.load(os.path.join(path, self.datafile), allow_pickle=True) as f:
				traindata, testdata = f["x_train"], f["x_test"]
				trainlabels, testlabels = f["y_train"], f["y_test"]

			trainperm = np.random.permutation(traindata.shape[0])
			testperm = np.random.permutation(testdata.shape[0])

			traindata, trainlabels = traindata[trainperm], trainlabels[trainperm]
			testdata, testlabels = testdata[testperm], testlabels[testperm]

			data, labels = np.concatenate([traindata, testdata]), np.concatenate([trainlabels, testlabels])

			if self.startchar is not None:
				data = [[self.startchar] + [w + self.indexFrom for w in sample] for sample in data]
			elif self.indexFrom:
				data = [[w + self.indexFrom for w in sample] for sample in data]

			if self.numwords is None:
				self.numwords = max([max(sample) for sample in data])

			if log:
				print("[%s] Started truncating vocabulary (%s max) ..." % (self.__class__.__name__, self.numwords))

			if self.oovchar is not None:
				data = [[self.oovchar if (w >= self.numwords or w < self.skiptop) else w for w in sample]
						for sample in data]
			else:
				truncdata = []
				for sample in data:
					truncsample = []
					for w in sample:
						if self.skiptop <= w < self.numwords:
							truncsample.append(w)

					truncsample = [self.padchar] * (len(sample) - len(truncsample)) + truncsample
					truncdata.append(np.array(truncsample, dtype=np.int32))

				data = truncdata

			if log:
				print("[%s] Started adjusting samples length (%s max) ..." % (self.__class__.__name__, self.maxlen))

			if self.maxlen is None:
				self.maxlen = max([len(sample) for sample in data])

			adjdata = []
			for sample, label in zip(data, labels):
				if len(sample) < self.maxlen:
					adjdata.append([self.padchar] * (self.maxlen - len(sample)) + sample)
				else:
					adjdata.append(sample[-self.maxlen:])

			data = adjdata

			print("[%s] Building cache ..." % self.__class__.__name__)

			vocab = self.loadVocabulary(path)
			data, labels = np.array(data, dtype=np.int32), np.array(labels, dtype=np.int32)

			with h5py.File(self.cachename, "w") as hdf:
				dsetname, lblsetname, vocsetname = self.datanames
				hdf.create_dataset(dsetname, data=data, compression=compress)
				hdf.create_dataset(lblsetname, data=labels, compression=compress)
				hdf.create_dataset(vocsetname, data=vocab, compression=compress)

				params = json.dumps({"numwords": self.numwords, "skiptop": self.skiptop, "maxlen": self.maxlen,
									 "padchar": self.padchar, "startchar": self.startchar, "oovchar": self.oovchar,
									 "indexFrom": self.indexFrom})

				dt = h5py.special_dtype(vlen=str)
				hdf.create_dataset("params", (), dtype=dt, data=params)

		hdf = h5py.File(self.cachename, "r")
		dsetname, lblsetname, vocsetname = self.datanames
		return hdf[dsetname], hdf[lblsetname], hdf[vocsetname]


def unittest():
	imdb = IMDBLoader(numwords=20000, maxlen=80)
	imdb.load(path="../TestData/")
	imdb.clear()


if __name__ == "__main__":
	unittest()
