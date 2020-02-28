import os

import h5py
import numpy as np
from PIL import Image

from PuzzleLib.Datasets.DataLoader import DataLoader


class InputLoader(DataLoader):
	def __init__(self, onFile=None, exts=None, dataname=None, cachename=None, onFileList=None):
		super().__init__(dataname, cachename)

		if onFile is None:
			def onFile(f):
				img = np.array(Image.open(f), dtype=np.float32) * 2.0 / 255.0 - 1.0
				img = np.rollaxis(img, 2)

				return img.reshape(1, *img.shape)

		self.onFile = onFile
		self.onFileList = onFileList

		if exts is None:
			self.exts = [".png", ".jpg", ".jpeg"]
		else:
			self.exts = ["." + ext if not ext.startswith(".") else ext for ext in exts]

		self.resizeFactor = 1.5

		self.log = True

		self.hdf = None
		self.compress = None
		self.dataset = None

		self.maxsamples = 0
		self.samples = 0


	def checkNeedToLoad(self, log=True):
		if os.path.exists(self.cachename):
			with h5py.File(self.cachename, "r") as hdf:
				mtimes = hdf["timestamps"]

				for inputname, mtime in mtimes.items():
					if mtime[()] < os.path.getmtime(inputname.replace("\\", "/")):
						if log:
							print("[%s] Archive %s has newer time stamp" % (self.__class__.__name__, inputname))

						return True
		else:
			return True

		return False


	def createDataset(self, unpacked):
		dataset = self.hdf.create_dataset(self.datanames[0], shape=unpacked.shape,
										  maxshape=(None, ) + unpacked.shape[1:], dtype=unpacked.dtype,
										  compression=self.compress)

		dataset[:] = unpacked
		return dataset


	def load(self, inputnames, maxsamples=None, filepacksize=5000, compress="gzip", log=True):
		self.log = log

		if isinstance(inputnames, str):
			inputnames = [inputnames]

		if self.cachename is None:
			self.cachename = os.path.splitext(inputnames[0])[0] + ".hdf"

		needsToLoad = self.checkNeedToLoad(log)
		if needsToLoad:
			if log:
				print("[%s] Creating cache file %s ..." % (self.__class__.__name__, self.cachename))

			timestamps = {}
			for inputname in inputnames:
				timestamps[inputname] = os.path.getmtime(inputname)

			with h5py.File(self.cachename, "w") as hdf:
				timeGrp = hdf.create_group("timestamps")
				for name, attr in timestamps.items():
					timeGrp.create_dataset(os.path.normpath(name).replace("/", "\\"), data=attr)

				self.hdf = hdf
				self.compress = compress
				self.dataset = None

				self.maxsamples = maxsamples
				self.samples = 0

				for i, inputname in enumerate(inputnames):
					if log:
						print("[%s] Unpacking archive %s (%d out of %d) ..." %
							  (self.__class__.__name__, inputname, i + 1, len(inputnames)))

					self.unpack(inputname, filepacksize)
					if self.maxsamples is not None and self.samples == self.maxsamples:
						print("[%s] Reached max limit of samples (%d)" % (self.__class__.__name__, self.maxsamples))

		else:
			if log:
				print("[%s] Using cache %s ..." % (self.__class__.__name__, self.cachename))

		return h5py.File(self.cachename, "r")[self.datanames[0]]


	def unpack(self, inputname, filepacksize):
		self.checkInput(inputname)

		with self.openInput(inputname) as inp:
			files = self.getFilelist(inp)

			numofpacks = len(files) // filepacksize
			resid = (len(files) % filepacksize != 0)

			packs = []
			for idx in range(numofpacks):
				packs.append(files[idx * filepacksize:(idx + 1) * filepacksize])

			if resid:
				packs.append(files[numofpacks * filepacksize:])

			for idx, pack in enumerate(packs):
				if self.log:
					print("[%s] Started unpacking pack %d out of %d ..." %
						  (self.__class__.__name__, idx + 1, len(packs)))

				self.cacheFilepack(inp, pack)
				if self.maxsamples is not None and self.samples == self.maxsamples:
					break


	def cacheFilepack(self, inp, pack):
		data = None
		nsamples = 0

		for i, file in enumerate(pack):
			try:
				if self.log:
					print("[%s] Unpacking file %s (%d out of %d)" % (self.__class__.__name__, file, i + 1, len(pack)))

				batch = self.onFile(self.openFile(inp, file))
			except Exception as e:
				raise RuntimeError("Unpacking failure: %s" % e)

			if data is None:
				data = np.empty((len(pack)-1 + batch.shape[0], ) + batch.shape[1:], dtype=batch.dtype)

			if nsamples + batch.shape[0] > data.shape[0]:
				newShape = (int(self.resizeFactor * (data.shape[0] + batch.shape[0])), ) + data.shape[1:]
				newData = np.empty(newShape, dtype=batch.dtype)

				newData[:data.shape[0]] = data
				data = newData

			data[nsamples:nsamples + batch.shape[0]] = batch
			nsamples += batch.shape[0]

			if self.maxsamples is not None and self.samples + nsamples >= self.maxsamples - 1:
				data = data[:self.maxsamples - self.samples]
				nsamples = self.maxsamples - self.samples
				break

		data = data[:nsamples]

		if self.log:
			size = data.nbytes / 1024**2
			print("[%s] Saving unpacked data ... (shape=%s, size=%s mbytes)" %
				  (self.__class__.__name__, data.shape, size))

		if self.dataset is None:
			self.dataset = self.createDataset(data)

		else:
			if self.samples + nsamples > self.dataset.shape[0]:
				self.dataset.resize((self.samples + nsamples, ) + self.dataset.shape[1:])
				self.dataset[self.samples:] = data

			else:
				self.dataset[self.samples:self.samples + nsamples] = data

		self.samples += nsamples
		print("[%s] Samples ready: %d (max: %s)" % (self.__class__.__name__, self.samples, self.maxsamples))


	def checkInput(self, inputname):
		raise NotImplementedError()


	def openInput(self, inputname):
		raise NotImplementedError()


	def getFilelist(self, inp):
		lst = self.loadFilelist(inp)
		if self.onFileList is not None:
			lst = self.onFileList(lst)

		return lst


	def loadFilelist(self, inp):
		raise NotImplementedError()


	def openFile(self, inp, file):
		raise NotImplementedError()
