import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Backend.Kernels import MatVec
from PuzzleLib.Backend.Utils import dtypesSupported

from PuzzleLib.Variable import Variable
from PuzzleLib.Modules.Module import ModuleError, Module


class Linear(Module):
	def __init__(self, insize, outsize, wscale=1.0, useBias=True, initscheme=None, name=None,
				 empty=False, transpose=False):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.transpose = transpose
		self.useBias = useBias

		self.W = None
		self.b = None

		if empty:
			return

		Wshape, bshape = ((outsize, insize), (insize, )) if transpose else ((insize, outsize), (outsize, ))
		W = self.createTensorWithScheme(initscheme, Wshape, wscale, self.calcNeuronsNumber(Wshape, transpose))

		self.setVar("W", Variable(gpuarray.empty(Wshape, dtype=self.calctype) if W is None else gpuarray.to_gpu(W)))

		if useBias:
			self.setVar("b", Variable(gpuarray.zeros(bshape, dtype=self.calctype)))


	def updateData(self, data):
		self.data = Blas.mulMatrixOnMatrix(data, self.W, transpB=self.transpose)

		if self.useBias:
			MatVec.addVecToMat(self.b, self.data, axis=1, out=self.data)


	def updateGrad(self, grad):
		self.grad = Blas.mulMatrixOnMatrix(grad, self.W, transpB=not self.transpose)


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		if not self.transpose:
			Blas.mulMatrixOnMatrix(self.inData, grad, out=self.vars["W"].grad, transpA=True, alpha=scale, beta=momentum)
		else:
			Blas.mulMatrixOnMatrix(grad, self.inData, out=self.vars["W"].grad, transpA=True, alpha=scale, beta=momentum)

		if self.useBias:
			Blas.sumOnMatrix(grad, out=self.vars["b"].grad, alpha=scale, beta=momentum)


	def dataShapeFrom(self, shape):
		return (shape[0], self.W.shape[1]) if not self.transpose else (shape[0], self.W.shape[0])


	def checkDataShape(self, shape):
		if len(shape) != 2:
			raise ModuleError("Data must be 2d matrix")

		if not self.transpose:
			if shape[1] != self.W.shape[0]:
				raise ModuleError("Expected %d data dimensions, %d were given" % (self.W.shape[0], shape[1]))
		else:
			if shape[1]!= self.W.shape[1]:
				raise ModuleError("Expected %d data dimensions, %d were given" % (self.W.shape[1], shape[1]))


	def gradShapeFrom(self, shape):
		return (shape[0], self.W.shape[0]) if not self.transpose else (shape[0], self.W.shape[1])


	def checkGradShape(self, shape):
		if len(shape) != 2:
			raise ModuleError("Grad must be 2d matrix")

		if not self.transpose:
			if shape[1] != self.W.shape[1]:
				raise ModuleError("Expected %d grad dimensions, %d were given" % (self.W.shape[1], shape[1]))
		else:
			if shape[1] != self.W.shape[0]:
				raise ModuleError("Expected %d grad dimensions, %d were given" % (self.W.shape[0], shape[1]))


	def calcMode(self, T):
		if Config.backend == Config.Backend.cuda:
			if self.calctype == T:
				return

			variables = self.vars
			self.vars = {}

			for varName, var in variables.items():
				self.setVar(varName, Variable(var.data.astype(T), name=var.name, grad=var.grad.astype(T)))

			self.calctype = T

		else:
			super().calcMode(T)


def unittest():
	for dtype, atol in dtypesSupported():
		calcTest(dtype, atol)
		trainTest(dtype)


def calcTest(dtype, atol):
	insize, outsize = 5, 1

	hostData = np.random.randn(5, insize).astype(dtype)
	data = gpuarray.to_gpu(hostData)

	linear = Linear(insize, outsize, initscheme=("xavier", "avg"))
	linear.calcMode(dtype)

	linear(data)

	hostGrad = np.random.randn(*linear.data.shape).astype(dtype)
	grad = gpuarray.to_gpu(hostGrad)

	linear.backward(grad)

	hostW, hostBias = linear.W.get(), linear.b.get()

	hostOutData = np.dot(hostData, hostW) + hostBias[np.newaxis, :]
	hostInGrad = np.dot(hostGrad, hostW.T)

	hostWGrad = np.dot(hostData.T, hostGrad)
	hostBiasGrad = np.sum(hostGrad, axis=0)

	assert np.allclose(hostOutData, linear.data.get(), atol=atol)
	assert np.allclose(hostInGrad, linear.grad.get(), atol=atol)

	assert np.allclose(hostWGrad, linear.vars["W"].grad.get(), atol=atol)
	assert np.allclose(hostBiasGrad, linear.vars["b"].grad.get(), atol=atol)


def trainTest(dtype):
	insize, outsize = 500, 100

	hostData = np.random.randn(32, insize).astype(dtype)
	hostTarget = np.random.randn(32, outsize).astype(np.float32)

	data, target = gpuarray.to_gpu(hostData), gpuarray.to_gpu(hostTarget)

	linear = Linear(insize, outsize)
	linear.calcMode(dtype)

	from PuzzleLib.Cost.MSE import MSE
	mse = MSE()

	learnRate = 1e-1

	for i in range(100):
		linear(data)

		outdata = linear.data if dtype == np.float32 else linear.data.astype(np.float32)
		error, grad = mse(outdata, target)

		linear.backward(grad if dtype == np.float32 else grad.astype(dtype))
		linear.updateParams(learnRate)

		if (i + 1) % 5 == 0:
			print("Iteration #%d error: %s" % (i + 1, error))


if __name__ == "__main__":
	unittest()
