class Transformer:
	def __call__(self, batch, threadidx):
		return batch


def unittest():
	from PuzzleLib.Transformers.Merger import Merger
	from PuzzleLib.Datasets.ZipLoader import ZipLoader

	zipfile = ZipLoader()
	data1 = zipfile.load("../TestData/test.zip")
	data2 = zipfile.load("../TestData/test.zip")

	with Merger([data1, data2]) as merger:
		merger.addTransformer(Transformer())

		for _ in range(10):
			merger.prepareData(chunksize=4, permutate=False)
			merger.getData()


if __name__ == "__main__":
	unittest()
