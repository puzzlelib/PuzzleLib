from PuzzleLib import Config
from PuzzleLib.Backend import gpuarray


class Variable:
	index = 0


	def __init__(self, data, name=None, withgrad=True, grad=None, updater=None, postUpdater=None):
		if name is None:
			self.name = str(type(self).index)
			type(self).index += 1
		else:
			self.name = name

		self.data = data
		self.updater = updater

		if updater is not None:
			return

		self.postUpdater = postUpdater
		self.grad = None

		if grad is not None:
			self.grad = grad

		elif withgrad and not Config.globalEvalMode:
			self.grad = gpuarray.zeros(shape=self.data.shape, dtype=self.data.dtype)

		self.learnRate, self.momRate = 1.0, 1.0
		self.wc = 0.0


	@property
	def hasUpdater(self):
		return self.updater is not None


	@property
	def hasPostUpdater(self):
		return self.postUpdater is not None


	def update(self, learnRate):
		self.updater(self, learnRate)


	def postUpdate(self):
		self.postUpdater(self)


	def set(self, variable):
		self.data.set(variable.data)

		if self.grad is not None:
			self.grad.set(variable.grad)
