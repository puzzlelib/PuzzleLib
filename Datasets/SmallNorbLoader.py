import os, struct

import numpy as np
import h5py
from PIL import Image

from PuzzleLib.Datasets.DataLoader import DataLoader


class SmallNorbLoader(DataLoader):
	def __init__(self, onSample=None, sampleInfo=None, cachename=None):
		super().__init__(("data", "labels", "info"), "smallnorb.hdf" if cachename is None else cachename)

		self.sampleInfo = lambda: (np.float32, (28, 28)) if sampleInfo is None else sampleInfo
		self.onSample = lambda sample: np.array(Image.fromarray(sample).resize((28, 28))) \
			if onSample is None else onSample

		self.testdata = "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat"
		self.testlabels = "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat"
		self.testinfo = "smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat"

		self.traindata = "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat"
		self.trainlabels = "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat"
		self.traininfo = "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat"

		self.nlabels = 5
		self.ninstances = 10
		self.nelevs = 9
		self.nazimuths = 18
		self.nlights = 6


	def load(self, path, sort=False, compress="gzip", log=True, onlyTest=False):
		self.cachename = os.path.join(path, self.cachename)

		if not os.path.exists(self.cachename):
			if log:
				print("[%s] Started unpacking ..." % self.__class__.__name__)

			data, labels, info = None, None, None
			files = [self.testdata] if onlyTest else [self.traindata, self.testdata]

			for filename in files:
				with open(os.path.join(path, filename), "rb") as file:
					magic, ndim = struct.unpack("<ii", file.read(8))
					dims = struct.unpack("<" + "i" * max(ndim, 3), file.read(max(ndim, 3) * 4))

					trueMagic = 0x1E3D4C55
					if magic != trueMagic:
						raise ValueError("Bad magic number (got 0x%x, expected 0x%x)" % (magic, trueMagic))

					indata = np.fromfile(file, dtype=np.uint8).reshape(*dims)

					dtype, reqdims = self.sampleInfo()
					outdata = np.empty(dims[:2] + reqdims, dtype=dtype)

					for i in range(dims[0]):
						for j in range(dims[1]):
							outdata[i, j] = self.onSample(indata[i, j])

						if (i + 1) % 100 == 0 and log:
							print("[%s] Unpacked %s pairs out of %s" % (self.__class__.__name__, i + 1, dims[0]))

					data = outdata if data is None else np.vstack((data, outdata))

			for filename in [self.trainlabels, self.testlabels]:
				with open(os.path.join(path, filename), "rb") as file:
					magic, ndim = struct.unpack("<ii", file.read(8))
					struct.unpack("<" + "i" * max(ndim, 3), file.read(max(ndim, 3) * 4))

					trueMagic = 0x1E3D4C54
					if magic != trueMagic:
						raise ValueError("Bad magic number (got 0x%x, expected 0x%x)" % (magic, trueMagic))

					inlabels = np.fromfile(file, dtype=np.uint32)
					labels = inlabels if labels is None else np.concatenate((labels, inlabels))

			for filename in [self.traininfo, self.testinfo]:
				with open(os.path.join(path, filename), "rb") as file:
					magic, ndim = struct.unpack("<ii", file.read(8))
					dims = struct.unpack("<" + "i" * max(ndim, 3), file.read(max(ndim, 3) * 4))

					trueMagic = 0x1E3D4C54
					if magic != trueMagic:
						raise ValueError("Bad magic number (got 0x%x, expected 0x%x)" % (magic, trueMagic))

					ininfo = np.fromfile(file, dtype=np.uint32).reshape(dims[:2])
					info = ininfo if info is None else np.vstack((info, ininfo))

			if sort:
				data, labels, info = self.sortDataset(data, labels, info, log=log)

			print("[%s] Building cache ..." % self.__class__.__name__)

			with h5py.File(self.cachename, "w") as hdf:
				dsetname, lblsetname, infosetname = self.datanames
				hdf.create_dataset(dsetname, data=data, compression=compress)
				hdf.create_dataset(lblsetname, data=labels, compression=compress)
				hdf.create_dataset(infosetname, data=info, compression=compress)

		hdf = h5py.File(self.cachename, "r")
		dsetname, lblsetname, infosetname = self.datanames
		return hdf[dsetname], hdf[lblsetname], hdf[infosetname]


	def sortDataset(self, data, labels, info, log=True):
		shape = (self.nlabels, self.ninstances, self.nlights, self.nelevs, self.nazimuths)

		sortdata = np.empty(shape + data.shape[1:], dtype=np.float32)
		sortlabels = np.empty(shape, dtype=np.uint32)
		sortinfo = np.empty(shape + info.shape[1:], dtype=np.uint32)

		if log:
			print("[%s] Started sorting dataset ..." % self.__class__.__name__)

		for i in range(data.shape[0]):
			instance, elev, azimuth, light = info[i]
			label = labels[i]

			sortdata[label, instance, light, elev, azimuth // 2] = data[i]
			sortlabels[labels, instance, light, elev, azimuth // 2] = label
			sortinfo[labels, instance, light, elev, azimuth // 2] = info[i]

			if log and (i + 1) % 100 == 0:
				print("[%s] Sorted %s pairs out of %s" % (self.__class__.__name__, i + 1, data.shape[0]))

		return sortdata, sortlabels, sortinfo


def unittest():
	smallnorb = SmallNorbLoader()
	smallnorb.load(path="../TestData/", sort=True, onlyTest=True)
	smallnorb.clear()


if __name__ == "__main__":
	unittest()
