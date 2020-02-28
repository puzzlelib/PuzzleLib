import numpy as np

from PuzzleLib.Backend.Kernels.ElementWise import weightDecayKer


class Hook:
	def __call__(self, var, state, stream=None):
		raise NotImplementedError()


class WeightDecay(Hook):
	def __init__(self, rate):
		self.rate = rate


	def __call__(self, var, state, stream=None):
		assert var.grad.dtype == np.float32
		if var.wc > 0.0:
			weightDecayKer(var.grad, var.data, self.rate * var.wc, stream=stream)
