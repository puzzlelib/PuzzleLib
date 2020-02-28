import os, tarfile, pickle

import numpy as np
import h5py

from PuzzleLib.Datasets.DataLoader import DataLoader


class Cifar10Loader(DataLoader):
	def __init__(self, onSample=None, onSampleBatch=None, cachename="cifar10.hdf"):
		super().__init__(("data", "labels"), cachename)

		if onSample:
			self.onSample = onSample
		else:
			self.onSample = lambda smp: smp.reshape(3, 32, 32).astype(np.float32) * 2.0 / 255.0 - 1.0

		if onSampleBatch:
			self.onSampleBatch = onSampleBatch
		else:
			self.onSampleBatch = lambda smp, b: smp.reshape(b, 3, 32, 32).astype(np.float32) * 2.0 / 255.0 - 1.0

		self.datafiles = ["cifar-10-python.tar.gz", "cifar-10-python.tar"]


	def load(self, path, compress="gzip", log=True):
		self.cachename = os.path.join(path, self.cachename)

		filename = None
		for datafile in self.datafiles:
			if tarfile.is_tarfile(os.path.join(path, datafile)):
				filename = os.path.join(path, datafile)
				break

		if filename is None:
			raise ValueError("No proper datafile found in path %s (searched for %s)" % (path, self.datafiles))

		if not os.path.exists(self.cachename):
			dicts = []

			with tarfile.open(filename) as tar:
				for name in tar.getnames():
					if "data_batch" in name or "test_batch" in name:
						f = tar.extractfile(name)
						dicts.append(pickle.load(f, encoding="latin1"))

						if log:
							print("[%s] Unpacked %s" % (self.__class__.__name__, name))

			totallen = 0
			for d in dicts:
				totallen += len(d["labels"])

			images = np.empty((totallen, 3, 32, 32), dtype=np.float32)
			labels = np.empty((totallen, ), dtype=np.int32)

			if log:
				print("[%s] Started merging ..." % self.__class__.__name__)

			idx = 0
			for i, d in enumerate(dicts):
				data = d["data"]
				lbls = d["labels"]

				images[idx:idx + data.shape[0]] = self.onSampleBatch(data, data.shape[0])
				labels[idx:idx + len(lbls)] = lbls
				idx += data.shape[0]

				if log:
					print("[%s] Merged #%d batch out of %d" % (self.__class__.__name__, i + 1, len(dicts)))

			if log:
				print("[%s] Writing in cache ..." % self.__class__.__name__)

			with h5py.File(self.cachename, "w") as hdf:
				dsetname, lblsetname = self.datanames
				hdf.create_dataset(dsetname, data=images, compression=compress)
				hdf.create_dataset(lblsetname, data=labels, compression=compress)

		hdf = h5py.File(self.cachename, "r")
		dsetname, lblsetname = self.datanames
		return hdf[dsetname], hdf[lblsetname]


def unittest():
	cifar10 = Cifar10Loader()
	cifar10.load(path="../TestData/")
	cifar10.clear()


if __name__ == "__main__":
	unittest()
