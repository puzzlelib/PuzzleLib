import numpy as np

from PuzzleLib.Backend import gpuarray


class CostError(Exception):
	pass


class Cost:
	def __init__(self):
		self.accumErr = gpuarray.empty((), dtype=np.float32)
		self.devErr = gpuarray.empty((), dtype=np.float32)

		self.error = None
		self.valError = None
		self.grad = None

		self.batchsize = None
		self.numOfSamples = None

		self.dirty = True
		self.resetAccumulator()


	def resetAccumulator(self):
		self.resetDeviceAccumulator()

		self.batchsize = 0
		self.numOfSamples = 0


	def updateState(self, samples):
		self.batchsize = samples
		self.numOfSamples += samples


	def resetDeviceAccumulator(self):
		self.accumErr.fill(0.0)


	def getError(self):
		if self.dirty:
			self.error = self.devErr.get() / self.batchsize
			self.dirty = False

		return self.error


	def getMeanError(self):
		return self.accumErr.get() / self.numOfSamples


	def getValError(self):
		return self.valError


	def __call__(self, pred, target, queryError=True):
		if isinstance(target, gpuarray.GPUArray) and isinstance(pred, gpuarray.GPUArray):
			assert pred.shape[0] == target.shape[0]

		self.checkDataShape(pred, target)
		self.reset()

		self.grad = self.calcGrad(pred, target)
		self.calcError(pred, target)
		self.dirty = True

		self.updateState(self.getBatchsize(pred))

		if queryError:
			self.error = self.getError()

		if queryError:
			return self.error, self.grad
		else:
			return self.grad


	def calcError(self, pred, target):
		raise NotImplementedError()


	def calcGrad(self, pred, target):
		raise NotImplementedError()


	def validate(self, pred, target):
		if isinstance(target, gpuarray.GPUArray) and isinstance(pred, gpuarray.GPUArray):
			assert pred.shape[0] == target.shape[0]

		self.checkValDataShape(pred, target)
		self.valError = self.calcVal(pred, target)

		return self.valError


	def calcVal(self, pred, target):
		raise NotImplementedError()


	def reset(self):
		self.error = None
		self.valError = None

		self.grad = None


	def checkDataShape(self, pred, target):
		pass


	def checkValDataShape(self, pred, target):
		pass


	def getBatchsize(self, pred):
		return pred.shape[0]
