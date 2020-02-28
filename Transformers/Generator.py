import numpy as np

from PuzzleLib.Transformers.Provider import Provider
from PuzzleLib.Transformers.Transformer import Transformer


class Generator(Provider):
	def getNextChunk(self, chunksize, **kwargs):
		return None


class TestGenTransformer(Transformer):
	def __call__(self, batch, threadidx):
		return np.random.randn(10, 3, 4, 4).astype(np.float32)


def unittest():
	with Generator(numofthreads=4) as generator:
		generator.addTransformer(TestGenTransformer())

		generator.prepareData()
		assert generator.getData().shape == (40, 3, 4, 4)


if __name__ == "__main__":
	unittest()
