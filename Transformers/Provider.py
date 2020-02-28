from multiprocessing import Pool

import numpy as np


class Provider:
	def __init__(self, numofthreads=4):
		self.transformers = []

		self.numofthreads = numofthreads
		self.pool = Pool(numofthreads)
		self.pool.starmap(lambda: np.random.seed(), ())
		self.poolresults = None

		self.data = None


	def __enter__(self):
		return self


	def __exit__(self, exc_type, exc_value, traceback):
		self.closePool()


	def closePool(self):
		self.pool.close()
		self.pool.join()


	def addTransformer(self, transformer):
		self.transformers.append(transformer)


	def getNextChunk(self, chunksize, **kwargs):
		raise NotImplementedError()


	def prepareData(self, chunksize=20000, **kwargs):
		result = self.getNextChunk(chunksize, **kwargs)

		if len(self.transformers) == 0:
			self.data = result
			return

		if result is not None:
			if isinstance(result, tuple) or isinstance(result, list):
				batchsize = result[0].shape[0] // self.numofthreads
			else:
				batchsize = result.shape[0] // self.numofthreads

			batches = []
			for i in range(self.numofthreads - 1):
				if isinstance(result, tuple) or isinstance(result, list):
					batches.append([res[i * batchsize:(i + 1) * batchsize] for res in result])
				else:
					batches.append(result[i * batchsize:(i + 1) * batchsize])

			if isinstance(result, tuple) or isinstance(result, list):
				batches.append([res[(self.numofthreads - 1) * batchsize:] for res in result])
			else:
				batches.append(result[(self.numofthreads - 1) * batchsize:])

			args = []
			for i, batch in enumerate(batches):
				arg = (self.transformers, batch, i)
				args.append(arg)
		else:
			args = []
			for i in range(self.numofthreads):
				args.append((self.transformers, None, i))

		self.poolresults = self.pool.starmap_async(self.worker, args)


	def getData(self):
		if self.poolresults is not None:
			self.poolresults.wait()

			results = [None] * self.numofthreads
			for data in self.poolresults.get():
				result, threadidx = data
				results[threadidx] = result

			self.poolresults = None

			length = 0
			if isinstance(results[0], tuple) or isinstance(results[0], list):
				datshape = [res.shape[1:] for res in results[0]]

				for res in results:
					length += res[0].shape[0]

				self.data = tuple(np.empty((length, )+shape, dtype=results[0][i].dtype)
								  for i, shape in enumerate(datshape))

				idx = 0
				for res in results:
					for i, dat in enumerate(res):
						self.data[i][idx:idx + dat.shape[0]] = dat

					idx += res[0].shape[0]

			else:
				datshape = results[0].shape[1:]

				for res in results:
					length += res.shape[0]

				self.data = np.empty((length, ) + datshape, dtype=np.float32)

				idx = 0
				for res in results:
					self.data[idx:idx + res.shape[0]] = res
					idx += res.shape[0]

		return self.data


	@staticmethod
	def worker(transformers, batch, threadidx):
		for transformer in transformers:
			batch = transformer(batch, threadidx)

		return batch, threadidx
