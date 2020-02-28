import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Containers.Container import Container


class Parallel(Container):
	def __init__(self, name=None):
		super().__init__(name)
		self.graph = []


	@property
	def gradUsesOutData(self):
		if len(self.graph) == 0:
			return False

		for mod in self.graph:
			if mod.gradUsesOutData:
				return True

		return False


	@gradUsesOutData.setter
	def gradUsesOutData(self, val):
		pass


	@property
	def inplace(self):
		for mod in self.graph[:-1]:
			if getattr(mod, "inplace", False):
				return True

		return False


	def getBlueprint(self):
		blueprint = super().getBlueprint()
		blueprint["graph"] = [mod.name for mod in self.graph]

		return blueprint


	def append(self, mod, acquire=True):
		super().append(mod, acquire)
		self.graph.append(mod)

		return self


	def extend(self, container, acquire=True):
		if isinstance(container, Parallel):
			container = container.graph

		for mod in container:
			self.append(mod, acquire)


	def pop(self):
		mod = self.graph.pop()
		super().removeModule(mod)

		return mod


	def __getitem__(self, item):
		if isinstance(item, str):
			return super().__getitem__(item)

		elif isinstance(item, int):
			return self.graph[item]

		elif isinstance(item, slice):
			parallel = Parallel()
			parallel.extend(self.graph[item.start:item.stop:item.step])

			return parallel

		else:
			raise NotImplementedError(type(item).__name__)


	def getByIndex(self, index):
		return self.graph[index]


	def optimizeForShape(self, shapes, memlimit=None):
		for i, mod in enumerate(self.graph):
			mod.optimizeForShape(shapes[i], memlimit)


	def updateData(self, data):
		assert len(data) == len(self.graph)

		self.data = []
		for i, mod in enumerate(self.graph):
			try:
				mod(data[i])

			except ModuleError as e:
				raise ModuleError("%s:\nData error in module %d (%s):\n%s" % (self, i, mod, e))

			except Exception as e:
				self.handleError(mod, e)

			self.data.append(mod.data)


	def dataShapeFrom(self, shapes):
		outshapes = []

		for i, mod in enumerate(self.graph):
			outshapes.append(mod.dataShapeFrom(shapes[i]))

		return outshapes


	def backward(self, grad, updParamGrads=True, updGrad=True, scale=1.0, momentum=1.0):
		assert len(grad) == len(self.graph)

		self.grad = []
		for i, mod in enumerate(self.graph):
			try:
				mod.backward(grad[i], updParamGrads=updParamGrads, updGrad=updGrad, scale=scale, momentum=momentum)

			except ModuleError as e:
				raise ModuleError("%s:\nGrad error in module %d (%s):\n%s" % (self, i, mod, e))

			except Exception as e:
				self.handleError(mod, e)

			self.grad.append(mod.grad)


	def gradShapeFrom(self, shapes):
		inshapes = []

		for i, mod in enumerate(self.graph):
			inshapes.append(mod.gradShapeFrom(shapes[i]))

		return inshapes


	def updateGrad(self, grad):
		assert False


def unittest():
	from PuzzleLib.Containers.Sequential import Sequential
	from PuzzleLib.Modules import Linear, Activation, sigmoid, Identity, Concat

	data1 = gpuarray.to_gpu(np.random.randn(128, 128).astype(np.float32))
	data2 = gpuarray.to_gpu(np.random.randn(128, 16).astype(np.float32))
	data3 = gpuarray.to_gpu(np.random.randn(128, 32).astype(np.float32))

	seq = Sequential()
	seq.append(Linear(128, 64))
	seq.append(Activation(sigmoid))

	parallel = Parallel()
	parallel.append(seq)
	parallel.append(Identity())
	parallel.append(Identity())

	concat = Concat(axis=1)

	parallel([data1, data2, data3])
	concat(parallel.data)

	assert np.allclose(data2.get(), concat.data.get()[:, 64:64 + 16])

	grad = gpuarray.to_gpu(np.random.randn(128, 112).astype(np.float32))
	concat.backward(grad)
	parallel.backward(concat.grad)

	assert np.allclose(grad.get()[:, 64:64 + 16], parallel.grad[1].get())

	parallel = parallel[::2]
	parallel([data1, data3])


if __name__ == "__main__":
	unittest()
