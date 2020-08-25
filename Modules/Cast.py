from enum import Enum

import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.gpuarray import memoryPool as memPool
from PuzzleLib.Backend.Kernels.ElementWise import castFP16toFP32, castFP32toFP16

from PuzzleLib.Modules.Module import ModuleError, Module


class DataType(str, Enum):
	float32 = "float32"
	float16 = "float16"


class Cast(Module):
	def __init__(self, intype, outtype, name=None):
		super().__init__(name)

		intype, outtype = self.dataTypeToNumpy(intype), self.dataTypeToNumpy(outtype)
		self.registerBlueprint(locals())

		self.intype, self.outtype = intype, outtype

		self.dataKer = self.getCastKernel(intype, outtype)
		self.gradKer = self.getCastKernel(outtype, intype)


	def updateData(self, data):
		if self.intype != self.outtype:
			self.data = gpuarray.empty(data.shape, dtype=self.outtype, allocator=memPool)
			self.dataKer(self.data, data)

		else:
			self.data = data


	def updateGrad(self, grad):
		if self.intype != self.outtype:
			self.grad = gpuarray.empty(grad.shape, dtype=self.intype, allocator=memPool)
			self.gradKer(self.grad, grad)

		else:
			self.grad = grad


	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return shape


	def checkDataType(self, dtype):
		if dtype != self.intype:
			raise ModuleError("Expected dtype %s, got %s" % (self.intype, dtype))


	def checkGradType(self, dtype):
		if dtype != self.outtype:
			raise ModuleError("Expected dtype %s, got %s" % (self.outtype, dtype))


	@staticmethod
	def getCastKernel(intype, outtype):
		return {
			(DataType.float16, DataType.float32): castFP16toFP32,
			(DataType.float32, DataType.float16): castFP32toFP16
		}.get((intype, outtype), None)


	@staticmethod
	def dataTypeToNumpy(T):
		return T if isinstance(T, DataType) else {
			np.float32: DataType.float32,
			np.float16: DataType.float16
		}[np.dtype(T).type]


def unittest():
	batchsize, size, maps = 5, 4, 8

	hostData = np.random.randn(batchsize, size, maps).astype(np.float16)
	hostGrad = np.random.randn(batchsize, size, maps).astype(np.float32)

	data, grad = gpuarray.to_gpu(hostData), gpuarray.to_gpu(hostGrad)

	cast = Cast(data.dtype, grad.dtype)

	cast(data)
	assert np.allclose(hostData.astype(grad.dtype), cast.data.get())

	cast.backward(grad)
	assert np.allclose(hostGrad.astype(data.dtype), cast.grad.get())


if __name__ == "__main__":
	unittest()
