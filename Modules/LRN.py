from PuzzleLib.Modules.Module import ModuleError, Module


class LRN(Module):
	def __init__(self, N=5, alpha=1e-4, beta=0.75, K=2.0, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.N = N
		self.alpha = alpha
		self.beta = beta
		self.K = K

		self.workspace = None


	def dataShapeFrom(self, shape):
		return shape


	def checkDataShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Data must be 4d tensor")


	def gradShapeFrom(self, shape):
		return shape


	def checkGradShape(self, shape):
		if len(shape) != 4:
			raise ModuleError("Grad must be 4d tensor")


	def updateData(self, data):
		raise NotImplementedError()


	def updateGrad(self, grad):
		raise NotImplementedError()


	def reset(self):
		super().reset()
		self.workspace = None
