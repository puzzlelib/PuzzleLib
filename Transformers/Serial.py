import numpy as np

from PuzzleLib.Transformers.Provider import Provider


class Serial(Provider):
	def __init__(self, dataset, labels=None, numofthreads=4):
		super().__init__(numofthreads)

		self.datalen = dataset.shape[0]

		self.labels = labels
		self.dataset = dataset

		self.index = 0


	def getNextChunk(self, chunksize, **kwargs):
		if chunksize >= self.datalen:
			self.index = 0

			if self.labels is not None:
				return np.array(self.dataset), np.array(self.labels)
			else:
				return np.array(self.dataset)

		begin = self.index
		end = self.index + chunksize

		if end > self.datalen:
			chunk = np.empty((chunksize, ) + self.dataset.shape[1:], dtype=self.dataset.dtype)
			tup = chunk

			chunk[:self.datalen - begin] = self.dataset[begin:self.datalen]

			self.index = end - self.datalen
			chunk[self.datalen - begin:] = self.dataset[:self.index]

			if self.labels is not None:
				labels = np.empty((chunksize, ) + self.dataset.shape[1:], dtype=self.labels.dtype)
				tup = (chunk, labels)

				labels[:self.datalen - begin] = self.labels[begin:self.datalen]
				labels[self.datalen - begin:] = self.labels[:self.index]

		else:
			self.index = end
			chunk = np.array(self.dataset[begin:end])
			tup = chunk

			if self.labels is not None:
				labels = np.array(self.labels[begin:end])
				tup = (chunk, labels)

		return tup


def unittest():
	from PuzzleLib.Datasets.ZipLoader import ZipLoader

	zipfile = ZipLoader()
	data = zipfile.load("../TestData/test.zip")

	with Serial(data) as serial:
		for _ in range(10):
			serial.prepareData(chunksize=4)
			serial.getData()


if __name__ == "__main__":
	unittest()
