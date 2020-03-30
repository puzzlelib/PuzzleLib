import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import dtypesSupported, memoryPool as memPool
from PuzzleLib.Backend.Kernels.ElementWise import linearKer

from PuzzleLib.Modules.Module import ModuleError, Module


class MulAddConst(Module):
	def __init__(self, a=1.0, b=0.0, inplace=False, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.a, self.b = a, b

		self.inplace = inplace
		if inplace and Config.showWarnings:
			print("[%s] Warning: %s is using inplace flag" % (Config.libname, self))


	def updateData(self, data):
		self.data = data if self.inplace else gpuarray.empty(data.shape, dtype=data.dtype, allocator=memPool)
		linearKer(data.dtype)(self.data, data, self.a, self.b)


	def updateGrad(self, grad):
		self.grad = grad if self.inplace else gpuarray.empty(grad.shape, dtype=grad.dtype, allocator=memPool)
		linearKer(grad.dtype)(self.grad, grad, self.a, 0.0)


	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return shape


	def calcMode(self, T):
		dtypes = {dtype for dtype, _ in dtypesSupported()}

		if T not in dtypes:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	for dtype, atol in dtypesSupported():
		mulAddConstTest(dtype, atol)


def mulAddConstTest(dtype, atol):
	hostData = np.random.randn(10, 10, 4, 3).astype(dtype)
	data = gpuarray.to_gpu(hostData)

	mulAdd = MulAddConst(a=3.141592, b=42.0)
	mulAdd.calcMode(dtype)

	mulAdd(data)

	hostOutData = (hostData.astype(np.float32) * mulAdd.a + mulAdd.b).astype(dtype)
	assert np.allclose(hostOutData, mulAdd.data.get(), atol=atol)

	hostGrad = np.random.randn(*data.shape).astype(dtype)
	grad = gpuarray.to_gpu(hostGrad)

	mulAdd.backward(grad)

	hostInGrad = hostGrad * mulAdd.a
	assert np.allclose(hostInGrad, mulAdd.grad.get(), atol=atol)


if __name__ == "__main__":
	unittest()
