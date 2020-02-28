import random
import numpy as np

from PuzzleLib.Transformers.Provider import Provider


class Merger(Provider):
	def __init__(self, datasets, labelIds=None, numofthreads=4):
		super().__init__(numofthreads)

		self.datalens = []
		self.datasets = datasets
		self.indices = [0] * len(self.datasets)

		self.labelIds = labelIds

		for dataset in datasets:
			self.datalens.append(dataset.shape[0])

			if dataset.shape[1:] != datasets[0].shape[1:]:
				raise ValueError("Datasets must have same shapes")


	def getNextChunk(self, chunksize, **kwargs):
		ratios, randomize, permutate = kwargs["ratios"], kwargs["randomize"], kwargs["permutate"]

		if not randomize and chunksize >= sum(self.datalens):
			chunksize = sum(self.datalens)

		self.deriveChunkRatios(ratios, chunksize)

		if randomize:
			return self.getRandomChunk(chunksize, ratios, permutate)

		else:
			reviseRatios = False
			for i, dataset in enumerate(self.datasets):
				if self.datalens[i] < ratios[i]:
					ratios[i] = self.datalens[i]
					reviseRatios = True

			if reviseRatios:
				chunksize = sum(ratios)

			return self.getRationedChunk(chunksize, ratios, permutate)


	def getRandomChunk(self, chunksize, ratios, permutate):
		chunk = np.empty((chunksize, ) + self.datasets[0].shape[1:], dtype=self.datasets[0].dtype)

		labels = None
		if self.labelIds is not None:
			labels = np.empty((chunksize, ), dtype=np.int32)

		if permutate:
			order = np.random.permutation(chunksize)
		else:
			order = np.arange(chunksize)

		idx = 0
		for i, dataset in enumerate(self.datasets):
			for _ in range(ratios[i]):
				chunk[order[idx]] = dataset[random.randint(0, self.datalens[i]-1)]

				if self.labelIds is not None:
					labels[order[idx]] = self.labelIds[i]

				idx += 1

		if self.labelIds is not None:
			return chunk, labels
		else:
			return chunk


	def getRationedChunk(self, chunksize, ratios, permutate):
		chunk = np.empty((chunksize, ) + self.datasets[0].shape[1:], dtype=self.datasets[0].dtype)

		if self.labelIds is not None:
			labels = np.empty((chunksize, ), dtype=np.int32)

		if permutate:
			order = np.random.permutation(chunksize)

		idx = 0
		for i, dataset in enumerate(self.datasets):
			begin = self.indices[i]
			end = self.indices[i] + ratios[i]

			if end > self.datalens[i]:
				self.indices[i] = end - self.datalens[i]

				if permutate:
					for d in range(ratios[i]):
						if begin + d < self.datalens[i]:
							chunk[order[idx + d]] = dataset[begin + d]
						else:
							chunk[order[idx + d]] = dataset[begin + d - self.datalens[i]]

						if self.labelIds is not None:
							labels[order[idx + d]] = self.labelIds[i]

				else:
					chunk[idx:idx + self.datalens[i] - begin] = dataset[begin:self.datalens[i]]
					chunk[idx + self.datalens[i] - begin:idx + ratios[i]] = dataset[:self.indices[i]]

					if self.labelIds is not None:
						labels[idx:idx + self.datalens[i] - begin] = self.labelIds[i]
						labels[idx + self.datalens[i] - begin:idx + ratios[i]] = self.labelIds[i]
			else:
				self.indices[i] = end

				if permutate:
					for d in range(ratios[i]):
						chunk[order[idx + d]] = dataset[begin + d]

						if self.labelIds is not None:
							labels[order[idx + d]] = self.labelIds[i]
				else:
					chunk[idx:idx + ratios[i]] = dataset[begin:end]

					if self.labelIds is not None:
						labels[idx:idx + ratios[i]] = self.labelIds[i]

			idx += ratios[i]

		if self.labelIds is not None:
			return chunk, labels
		else:
			return chunk


	@staticmethod
	def deriveChunkRatios(ratios, chunksize):
		norm = sum(ratios)
		for i in range(len(ratios) - 1):
			ratios[i] = int(ratios[i] / norm * chunksize)

		ratios[-1] = chunksize - sum(ratios[:-1])


	def prepareData(self, ratios=None, chunksize=20000, randomize=False, permutate=True):
		if ratios is None:
			ratios = [1] * len(self.datasets)
		else:
			assert (len(ratios) == len(self.datasets))

		super().prepareData(chunksize, ratios=ratios, randomize=randomize, permutate=permutate)


def unittest():
	from PuzzleLib.Datasets.ZipLoader import ZipLoader

	zipfile = ZipLoader()
	data1 = zipfile.load("../TestData/test.zip")
	data2 = zipfile.load("../TestData/test.zip")

	with Merger([data1, data2], [0, 1]) as merger:
		for _ in range(10):
			merger.prepareData(chunksize=10, ratios=[6, 4], permutate=False)
			merger.getData()


if __name__ == "__main__":
	unittest()
