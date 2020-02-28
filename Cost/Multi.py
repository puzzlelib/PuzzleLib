import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Cost.Cost import Cost


class Multi(Cost):
	def __init__(self):
		self.costs = []
		super().__init__()

		self.devErr = None
		self.accumErr = None


	def append(self, cost):
		self.costs.append(cost)
		return self


	def resetAccumulator(self):
		for cost in self.costs:
			cost.resetAccumulator()


	def updateState(self, samples):
		for cost in self.costs:
			cost.updateState(samples)


	def resetDeviceAccumulator(self):
		for cost in self.costs:
			cost.resetDeviceAccumulator()


	def getError(self):
		if self.dirty:
			self.error = []
			for cost in self.costs:
				self.error.append(cost.getError())

			self.dirty = False

		return self.error


	def getMeanError(self):
		accumErr = []
		for cost in self.costs:
			accumErr.append(cost.getMeanError())

		return accumErr


	def calcGrad(self, preds, targets):
		grads = []

		for i, cost in enumerate(self.costs):
			cost.grad = cost.calcGrad(preds[i], targets[i])
			grads.append(cost.grad)

		return grads


	def calcError(self, preds, targets):
		for i, cost in enumerate(self.costs):
			cost.calcError(preds[i], targets[i])


	def calcVal(self, preds, targets):
		error = []

		for i, cost in enumerate(self.costs):
			error.append(cost.calcVal(preds[i], targets[i]))

		return error


	def checkDataShape(self, preds, targets):
		assert len(preds) == len(targets)
		assert [preds[i].dtype == preds[0].dtype for i in range(len(preds))]

		for i, cost in enumerate(self.costs):
			cost.checkDataShape(preds[i], targets[i])


	def checkValDataShape(self, preds, targets):
		assert len(preds) == len(targets)
		assert [preds[i].dtype == preds[0].dtype for i in range(len(preds))]

		for i, cost in enumerate(self.costs):
			cost.checkValDataShape(preds[i], targets[i])


	def getBatchsize(self, preds):
		return preds[0].shape[0]


def unittest():
	errorTest()
	valTest()


def errorTest():
	pred = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))
	target = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))

	from PuzzleLib.Cost.Abs import Abs
	from PuzzleLib.Cost.MSE import MSE

	multi = Multi().append(MSE()).append(Abs())
	multi([pred, pred], [target, target])

	hostError = [np.linalg.norm(target.get() - pred.get())**2 / (2.0 * np.prod(target.shape)),
				 np.linalg.norm((target.get() - pred.get()).ravel(), ord=1) / np.prod(target.shape)]

	assert np.isclose(multi.error[0], hostError[0])
	assert np.isclose(multi.error[1], hostError[1])


def valTest():
	pred = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))
	target = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))

	from PuzzleLib.Cost.Abs import Abs
	from PuzzleLib.Cost.MSE import MSE

	multi = Multi().append(Abs()).append(MSE())
	error = multi.validate([pred, pred], [target, target])

	hostError = [np.linalg.norm((target.get() - pred.get()).ravel(), ord=1) / np.prod(target.shape),
				 np.linalg.norm(target.get() - pred.get())**2 / (2.0 * np.prod(target.shape))]

	assert np.isclose(error[0], hostError[0])
	assert np.isclose(error[1], hostError[1])


if __name__ == "__main__":
	unittest()
