import numpy as np

from PuzzleLib import Config
from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Containers.Container import ContainerError, Container


class Sequential(Container):
	def __init__(self, name=None):
		super().__init__(name)
		self.graph = []


	@property
	def gradUsesOutData(self):
		if len(self.graph) == 0:
			return False

		index = -1
		mod = self.graph[index]

		while mod.movesData:
			index -= 1
			mod = self.graph[index]

		return mod.gradUsesOutData


	@gradUsesOutData.setter
	def gradUsesOutData(self, val):
		pass


	@property
	def inplace(self):
		fwdinp = True

		for mod in self.graph:
			if mod.movesData:
				continue
			elif getattr(mod, "inplace", False):
				break
			else:
				fwdinp = False
				break

		bwdinp = True

		for mod in reversed(self.graph):
			if mod.movesGrad:
				continue
			elif getattr(mod, "inplace", False):
				break
			else:
				bwdinp = False
				break

		return fwdinp or bwdinp


	def getBlueprint(self):
		blueprint = super().getBlueprint()
		blueprint["graph"] = [mod.name for mod in self.graph]

		return blueprint


	def append(self, mod, acquire=True):
		if len(self.graph) > 0:
			self.checkModulesCompatibility(self.graph[-1], mod)

		super().append(mod, acquire)
		self.graph.append(mod)

		return self


	def extend(self, container, acquire=True):
		if isinstance(container, Sequential):
			container = container.graph

		for mod in container:
			self.append(mod, acquire)


	def pop(self):
		mod = self.graph.pop()
		super().removeModule(mod)

		return mod


	def insert(self, mod, index):
		if index > 0:
			self.checkModulesCompatibility(self.graph[index - 1], mod)

		super().append(mod)
		self.graph.insert(index, mod)


	def insertAfter(self, mod, name):
		index = self.getModuleIndex(name)
		self.checkModulesCompatibility(self.graph[index], mod)

		super().append(mod)
		self.graph.insert(index + 1, mod)


	def checkModulesCompatibility(self, mod1, mod2):
		if Config.disableModuleCompatChecks:
			return

		if not getattr(mod2, "inplace", False):
			return

		if not mod1.gradUsesOutData:
			if not mod1.movesData:
				return
			else:
				index = self.getModuleIndex(mod1.name) - 1

				while index >= 0:
					mod1 = self.getByIndex(index)
					index -= 1

					if mod1.movesData:
						continue

					if not mod1.gradUsesOutData:
						return
					else:
						break

				if index < 0:
					return

		raise ContainerError(
			"%s: Can't insert inplace module %s after module %s (gradient uses outdata)" % (self, mod2, mod1)
		)


	def __getitem__(self, item):
		if isinstance(item, str):
			return super().__getitem__(item)

		elif isinstance(item, int):
			return self.graph[item]

		elif isinstance(item, slice):
			assert item.step == 1 or item.step is None

			seq = Sequential()
			seq.extend(self.graph[item.start:item.stop:item.step])

			return seq

		else:
			raise NotImplementedError(type(item).__name__)


	def getByIndex(self, index):
		return self.graph[index]


	def getModuleIndex(self, name):
		index = None
		for i, mod in enumerate(self.graph):
			if mod.name == name:
				index = i
				break

		if index is None:
			raise ContainerError("%s: Module %s not found" % (self, name))

		return index


	def optimizeForShape(self, shape, memlimit=None):
		for mod in self.graph:
			mod.optimizeForShape(shape, memlimit)
			shape = mod.dataShapeFrom(shape)


	def updateData(self, data):
		for i, mod in enumerate(self.graph):
			try:
				mod(data)

			except ModuleError as e:
				raise ModuleError("%s:\nData error in module %d (%s):\n%s" % (self, i, mod, e))

			except Exception as e:
				self.handleError(mod, e)

			data = mod.data

		if len(self.graph) == 0:
			self.data = data
		else:
			self.data = self.graph[-1].data


	def dataShapeFrom(self, shape):
		for mod in self.graph:
			shape = mod.dataShapeFrom(shape)

		return shape


	def backward(self, grad, updParamGrads=True, updGrad=True, scale=1.0, momentum=1.0):
		for i, mod in enumerate(reversed(self.graph)):
			try:
				if i < len(self.graph):
					mod.backward(grad, updParamGrads=updParamGrads, scale=scale, momentum=momentum)
				else:
					mod.backward(grad, updParamGrads=updParamGrads, updGrad=updGrad, scale=scale, momentum=momentum)

			except ModuleError as e:
				raise ModuleError("%s:\nGrad error in module %d (%s):\n%s" % (self, len(self.graph)-1 - i, mod, e))

			except Exception as e:
				self.handleError(mod, e)

			grad = mod.grad

		if len(self.graph) == 0:
			self.grad = grad
		else:
			self.grad = self.graph[0].grad


	def gradShapeFrom(self, shape):
		for mod in reversed(self.graph):
			shape = mod.gradShapeFrom(shape)

		return shape


	def updateGrad(self, grad):
		assert False


def unittest():
	simpleNetTest()
	complexNetTest()


def simpleNetTest():
	from PuzzleLib.Modules import Linear, Activation, sigmoid

	data = gpuarray.to_gpu(np.random.randn(128, 128).astype(np.float32))

	seq = Sequential()

	seq.append(Linear(128, 64))
	seq.append(Activation(sigmoid))

	seq.append(Linear(64, 32))
	seq.append(Activation(sigmoid))

	seq(data)
	assert seq.data.shape == (128, 32)

	grad = gpuarray.to_gpu(np.random.randn(128, 32).astype(np.float32))
	seq.backward(grad)
	seq.updateParams(1e-4)
	assert seq.grad.shape == data.shape

	data = gpuarray.to_gpu(np.random.randn(64, 128).astype(np.float32))
	seq = seq[:2]
	seq(data)
	assert seq.data.shape == (64, 64)


def complexNetTest():
	from PuzzleLib.Modules import Conv2D, MaxPool2D, Activation, relu, Flatten

	data = gpuarray.to_gpu(np.random.randn(128, 3, 150, 150).astype(np.float32))

	seq = Sequential()

	seq.append(Conv2D(3, 16, 11))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Conv2D(16, 16, 5))
	seq.append(MaxPool2D())
	seq.append(Activation(relu))

	seq.append(Flatten())

	seq(data)

	grad = gpuarray.to_gpu(np.random.randn(*seq.data.shape).astype(np.float32))
	seq.backward(grad)
	seq.updateParams(1e-4)


if __name__ == "__main__":
	unittest()
