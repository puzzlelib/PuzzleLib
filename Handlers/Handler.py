import numpy as np

from PuzzleLib.Backend import gpuarray


class Handler:
	def __init__(self, mod, onBatchFinish=None, batchsize=128):
		self.module = mod

		self.batchsize = batchsize
		self.onBatchFinish = onBatchFinish

		self.currBatch = 0
		self.totalBatches = 0

		self.currMacroBatch = 0
		self.totalMacroBatches = 0


	def handleFromHost(self, data, state=None, macroBatchSize=10000, onMacroBatchFinish=None, random=True):
		datasize = self.getDataSize(data)
		self.totalMacroBatches = (datasize + macroBatchSize - 1) // macroBatchSize

		order = np.random.permutation(self.totalMacroBatches) if random else np.arange(self.totalMacroBatches)

		for i, n in enumerate(order):
			macrobatch = self.sliceData(data, n, macroBatchSize, postSlice=lambda dat: gpuarray.to_gpu(dat))

			self.currMacroBatch = i + 1

			self.onMacroBatchStart(n, macroBatchSize, state)
			self.handle(macrobatch, state, random=random)
			self.onMacroBatchFinish(n, macroBatchSize, state)

			if onMacroBatchFinish:
				onMacroBatchFinish(self)


	def handle(self, data, state=None, random=True):
		datasize = self.getDataSize(data)
		self.totalBatches = (datasize + self.batchsize - 1) // self.batchsize

		order = np.random.permutation(self.totalBatches) if random else np.arange(self.totalBatches)

		for i, n in enumerate(order):
			batch = self.sliceData(data, n, self.batchsize, postSlice=lambda dat: dat)

			self.currBatch = i + 1

			self.handleBatch(batch, n, state)
			self.module.reset()

			if self.onBatchFinish:
				self.onBatchFinish(self)


	@staticmethod
	def getDataSize(data):
		while isinstance(data, list):
			data = data[0]

		return data.shape[0]


	@classmethod
	def parseShapeTree(cls, data, onData, auxdata=None):
		if isinstance(data, list):
			outdata = [
				cls.parseShapeTree(dat, onData, auxdata[i] if auxdata is not None else None)
				for i, dat in enumerate(data)
			]

		else:
			outdata = onData(data, auxdata) if auxdata is not None else onData(data)

		return outdata


	@classmethod
	def sliceData(cls, data, idx, batchsize, postSlice):
		if isinstance(data, list):
			slicing = [cls.sliceData(dat, idx, batchsize, postSlice) for dat in data]
		else:
			slicing = postSlice(data[idx * batchsize:(idx + 1) * batchsize])

		return slicing


	def onMacroBatchStart(self, idx, macroBatchSize, state):
		pass


	def onMacroBatchFinish(self, idx, macroBatchSize, state):
		pass


	def handleBatch(self, batch, idx, state):
		raise NotImplementedError()
