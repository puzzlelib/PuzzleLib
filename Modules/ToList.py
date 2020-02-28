import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Modules.Module import ModuleError, Module


class ToList(Module):
	def __init__(self, name=None):
		super().__init__(name)

		self.movesData = True
		self.movesGrad = True


	def updateData(self, data):
		self.data = []
		self.extendDataList(self.data, data)


	def extendDataList(self, lst, data):
		if isinstance(data, gpuarray.GPUArray):
			lst.append(data)
		else:
			for dat in data:
				self.extendDataList(lst, dat)


	def updateGrad(self, grad):
		self.grad, _ = self.buildGradList(grad, self.inData, 0)


	def buildGradList(self, grad, data, i):
		if isinstance(data, gpuarray.GPUArray):
			i += 1
			return grad[i - 1], i
		else:
			lst = []
			for dat in data:
				inlst, i = self.buildGradList(grad, dat, i)
				lst.append(inlst)

			return lst, i


	def dataShapeFrom(self, shapes):
		lst = []
		self.extendDataShapeList(lst, shapes)
		return lst


	def extendDataShapeList(self, lst, shapes):
		if isinstance(shapes, tuple):
			lst.append(shapes)
		else:
			for shape in shapes:
				self.extendDataShapeList(lst, shape)


	def gradShapeFrom(self, shapes):
		inshapes, _ = self.buildGradShapeList(shapes, self.inData, 0)
		return inshapes


	def buildGradShapeList(self, shapes, data, i):
		if isinstance(data, gpuarray.GPUArray):
			i += 1
			return shapes[i - 1], i

		else:
			lst = []
			for dat in data:
				inlst, i = self.buildGradShapeList(shapes, dat, i)
				lst.append(inlst)

			return lst, i


	def checkGradShape(self, shapes):
		self.checkGradList(shapes, self.inData, 0)


	def checkGradList(self, shapes, data, i):
		if isinstance(data, gpuarray.GPUArray):
			if data.shape != shapes[i]:
				raise ModuleError("Inconsistency in data and corresponding grad shapes at index %s "
								  "(expected %s, given %s)" % (i, data.shape, shapes[i]))
			i += 1
			return i

		else:
			for dat in data:
				i = self.checkGradList(shapes, dat, i)

			return i


def unittest():
	data1 = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))
	data2 = gpuarray.to_gpu(np.random.randn(5, 5).astype(np.float32))
	data3 = gpuarray.to_gpu(np.random.randn(3, 6).astype(np.float32))

	data = [[data1, data2], data3]
	outdata = [data1, data2, data3]

	tolist = ToList()
	tolist(data)

	assert all(np.allclose(d.get(), outdata[i].get()) for i, d in enumerate(tolist.data))

	shapes = tolist.dataShapeFrom([data1.shape, [data2.shape, data3.shape]])
	assert all(outdata[i].shape == shape for i, shape in enumerate(shapes))

	grad1 = gpuarray.to_gpu(np.random.randn(10, 10).astype(np.float32))
	grad2 = gpuarray.to_gpu(np.random.randn(5, 5).astype(np.float32))
	grad3 = gpuarray.to_gpu(np.random.randn(3, 6).astype(np.float32))

	grad = [grad1, grad2, grad3]
	ingrad = [[grad1, grad2], grad3]

	tolist.backward(grad)

	assert np.allclose(ingrad[0][0].get(), grad1.get())
	assert np.allclose(ingrad[0][1].get(), grad2.get())
	assert np.allclose(ingrad[1].get(), grad3.get())

	inshapes = tolist.gradShapeFrom([gr.shape for gr in grad])

	assert inshapes[0][0] == grad1.shape
	assert inshapes[0][1] == grad2.shape
	assert inshapes[1] == grad3.shape


if __name__ == "__main__":
	unittest()
