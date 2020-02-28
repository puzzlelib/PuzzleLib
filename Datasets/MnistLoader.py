import os, struct, array

import numpy as np
import h5py

from PuzzleLib.Datasets.DataLoader import DataLoader


class MnistLoader(DataLoader):
	def __init__(self, onSample=None, cachename="mnist.hdf"):
		super().__init__(("data", "labels"), cachename)

		if onSample:
			self.onSample = onSample
		else:
			self.onSample = lambda smp: np.array(smp, dtype=np.float32).reshape((1, 28, 28)) / 255.0

		self.testdata = "t10k-images.idx3-ubyte"
		self.testlabels = "t10k-labels.idx1-ubyte"

		self.traindata = "train-images.idx3-ubyte"
		self.trainlabels = "train-labels.idx1-ubyte"


	def load(self, path, compress="gzip", log=True):
		self.cachename = os.path.join(path, self.cachename)

		if not os.path.exists(self.cachename):
			imgs, lbls = [], []

			if log:
				print("[%s] Started unpacking ..." % self.__class__.__name__)

			for filename in [self.testlabels, self.trainlabels]:
				with open(os.path.join(path, filename), "rb") as file:
					magic, size = struct.unpack(">II", file.read(8))

					trueMagic = 2049
					if magic != trueMagic:
						raise ValueError("Bad magic number (got %s, expected %s)" % (magic, trueMagic))

					lbls += array.array("B", file.read())

			for filename in [self.testdata, self.traindata]:
				with open(os.path.join(path, filename), "rb") as file:
					magic, size, rows, cols = struct.unpack(">IIII", file.read(16))

					trueMagic = 2051
					if magic != trueMagic:
						raise ValueError("Bad magic number (got %s, expected %s)" % (magic, trueMagic))

					data = array.array("B", file.read())
					datsize = rows * cols

					for i in range(size):
						dat = data[i * datsize:(i+1) * datsize]
						imgs.append(dat)

			images = np.empty((len(imgs), 1, rows, cols), dtype=np.float32)
			labels = np.empty((len(imgs), ), dtype=np.int32)

			print("[%s] Building cache ..." % self.__class__.__name__)

			for i in range(len(lbls)):
				images[i] = self.onSample(imgs[i])
				labels[i] = lbls[i]

			with h5py.File(self.cachename, "w") as hdf:
				dsetname, lblsetname = self.datanames
				hdf.create_dataset(dsetname, data=images, compression=compress)
				hdf.create_dataset(lblsetname, data=labels, compression=compress)

		hdf = h5py.File(self.cachename, "r")
		dsetname, lblsetname = self.datanames
		return hdf[dsetname], hdf[lblsetname]


def unittest():
	mnist = MnistLoader()
	mnist.load(path="../TestData/")
	mnist.clear()


if __name__ == "__main__":
	unittest()
