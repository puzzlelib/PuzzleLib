from enum import Enum

import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import dtypesSupported, fillUniform, fillNormal, copy, globalRng, memoryPool as memPool
from PuzzleLib.Backend.Kernels.ElementWise import mulKer, addKer

from PuzzleLib.Modules.Module import ModuleError, Module


class InjectMode(str, Enum):
	add = "add"
	mul = "mul"


class NoiseType(str, Enum):
	gaussian = "gaussian"
	uniform = "uniform"


class NoiseInjector(Module):
	def __init__(self, mode="add", noisetype="uniform", params=(0.0, 1.0), rng=globalRng, inplace=False, slicing=None,
				 name=None):
		super().__init__(name)
		self.registerBlueprint(locals(), exclude=["rng"])

		self.rng = globalRng if rng is None else rng

		self.mode = InjectMode(mode)
		self.type = NoiseType(noisetype)

		self.params = params
		self.slice = slicing

		self.rands = None

		self.inplace = inplace
		if inplace and Config.showWarnings:
			print("[%s] Warning: %s is using inplace flag" % (Config.libname, self))


	def updateData(self, data):
		if self.train:
			size = data.size if data.size % 2 == 0 else data.size + 1
			rands = gpuarray.empty((size, ), dtype=np.float32, allocator=memPool)

			if self.type == NoiseType.uniform:
				a, b = self.params
				fillUniform(rands, a, b, self.rng)

			elif self.type == NoiseType.gaussian:
				mean, sigma = self.params
				fillNormal(rands, mean, sigma, self.rng)

			else:
				raise NotImplementedError(self.type)

			self.rands = rands if data.dtype == np.float32 else rands.astype(data.dtype)
			self.rands = self.rands[:data.size].reshape(data.shape)

			if self.inplace:
				self.data = data
			else:
				if self.slice is not None:
					self.data = copy(None, data)
				else:
					self.data = gpuarray.empty(data.shape, dtype=data.dtype, allocator=memPool)

			if self.mode == InjectMode.add:
				addKer(data.dtype)(self.data, data, 1, self.rands, 1, slice=self.slice)
			elif self.mode == InjectMode.mul:
				mulKer(data.dtype)(self.data, data, self.rands, slice=self.slice)

			else:
				raise NotImplementedError(self.mode)
		else:
			self.data = data


	def updateGrad(self, grad):
		if self.mode == InjectMode.mul:
			if self.inplace:
				self.grad = grad
			else:
				if self.slice is not None:
					self.grad = copy(None, grad)
				else:
					self.grad = gpuarray.empty(grad.shape, dtype=grad.dtype, allocator=memPool)

			mulKer(grad.dtype)(self.grad, grad, self.rands, slice=self.slice)

		elif self.mode == InjectMode.add:
			if self.inplace:
				self.grad = grad
			else:
				self.grad = copy(None, grad)

		else:
			raise NotImplementedError(self.mode)


	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return shape


	def reset(self):
		super().reset()
		self.rands = None


	def calcMode(self, T):
		if Config.backend == Config.Backend.cuda:
			if T not in {np.float16, np.float32}:
				raise ModuleError("Unsupported dtype %s" % T)

		elif T != np.float32:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, _ in dtypesSupported():
		noiseInjectorTest(dtype)


def noiseInjectorTest(dtype):
	hostData = np.random.randn(10, 3, 16, 16).astype(dtype)
	data = gpuarray.to_gpu(hostData)

	injector = NoiseInjector(mode="mul", noisetype="uniform", params=(0.0, 10.0))
	injector.calcMode(dtype)

	injector(data)
	assert np.allclose(injector.data.get(), hostData * injector.rands.get())

	hostGrad = np.random.randn(*data.shape).astype(dtype)
	grad = gpuarray.to_gpu(hostGrad)

	injector.backward(grad)
	assert np.allclose(injector.grad.get(), hostGrad * injector.rands.get())

	injector = NoiseInjector(mode="add", noisetype="gaussian", params=(0.0, 1.0))
	injector.calcMode(dtype)

	injector(data)
	assert np.allclose(injector.data.get(), hostData + injector.rands.get())

	injector.backward(grad)
	assert np.allclose(injector.grad.get(), hostGrad)


if __name__ == "__main__":
	unittest()
