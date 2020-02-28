import math

import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool, globalRng
from PuzzleLib.Backend.Kernels.ElementWise import rbmKer
from PuzzleLib.Backend.Kernels.MatVec import addVecToMat
from PuzzleLib.Backend import Blas

from PuzzleLib.Variable import Variable
from PuzzleLib.Modules.Module import Module


class RBM(Module):
	def __init__(self, vsize, hsize, wscale=1.0, rng=globalRng, useBias=True, name=None):
		super().__init__(name)
		self.rng = rng

		W = np.random.normal(0.0, wscale / math.sqrt(vsize + hsize), (vsize, hsize)).astype(np.float32)

		self.W = None
		self.setVar("W", Variable(gpuarray.to_gpu(W, allocator=memPool)))

		self.useBias = useBias

		if useBias:
			self.b = None
			self.setVar("b", Variable(gpuarray.zeros((vsize, ), dtype=np.float32, allocator=memPool)))

			self.c = None
			self.setVar("c", Variable(gpuarray.zeros((hsize, ), dtype=np.float32, allocator=memPool)))

		self.particles = None


	def hiddenFromVisible(self, visible):
		hidden = Blas.mulMatrixOnMatrix(visible, self.W)

		if self.useBias:
			addVecToMat(self.c, hidden, axis=1, out=hidden)

		self.activateNeurons(hidden)
		return hidden


	def visibleFromHidden(self, hidden):
		visible = Blas.mulMatrixOnMatrix(hidden, self.W, transpB=True)

		if self.useBias:
			addVecToMat(self.b, visible, axis=1, out=visible)

		self.activateNeurons(visible)
		return visible


	def activateNeurons(self, neurons):
		rands = gpuarray.empty(neurons.shape, dtype=np.float32, allocator=memPool)
		self.rng.fillUniform(rands)

		rbmKer(neurons, neurons, rands)


	def updateData(self, data):
		raise RuntimeError("RBM does not support full module interface")


	def updateGrad(self, grad):
		raise RuntimeError("RBM does not support full module interface")


	def calcCDGrad(self, data):
		hidden = self.posPhaseGrad(data)
		self.negPhaseGrad(hidden)


	def calcPCDGrad(self, data):
		hidden = self.posPhaseGrad(data)

		if self.particles is None:
			self.particles = gpuarray.to_gpu(np.random.binomial(1, 0.5, size=hidden.shape).astype(np.float32))

		self.particles = self.negPhaseGrad(self.particles)


	def posPhaseGrad(self, data):
		hidden = self.hiddenFromVisible(data)
		Blas.mulMatrixOnMatrix(data, hidden, out=self.vars["W"].grad, transpA=True)

		if self.useBias:
			Blas.sumOnMatrix(data, out=self.vars["b"].grad)
			Blas.sumOnMatrix(hidden, out=self.vars["c"].grad)

		return hidden


	def negPhaseGrad(self, hidden):
		visible = self.visibleFromHidden(hidden)
		hidden = self.hiddenFromVisible(visible)

		Blas.mulMatrixOnMatrix(visible, hidden, out=self.vars["W"].grad, transpA=True, alpha=-1.0, beta=1.0)

		if self.useBias:
			Blas.sumOnMatrix(visible, out=self.vars["b"].grad, alpha=-1.0, beta=1.0)
			Blas.sumOnMatrix(hidden, out=self.vars["c"].grad, alpha=-1.0, beta=1.0)

		return hidden


	def dataShapeFrom(self, shape):
		raise NotImplementedError()


	def gradShapeFrom(self, shape):
		raise NotImplementedError()


def unittest():
	from PuzzleLib.Optimizers.MomentumSGD import MomentumSGD
	from PuzzleLib.Datasets.MnistLoader import MnistLoader
	from PuzzleLib.Visual import showImageBatchInFolder

	mnist = MnistLoader()
	data, _ = mnist.load(path="../../TestData")
	data = data[:].reshape(data.shape[0], np.prod(data.shape[1:]))

	rbm = RBM(784, 500)
	optimizer = MomentumSGD(momRate=0.5)
	optimizer.setupOn(rbm, useGlobalState=True)

	data = gpuarray.to_gpu(data)
	batchsize = 100

	for epoch in range(10):
		for i in range(data.shape[0] // batchsize):
			batch = data[i * batchsize:(i+1) * batchsize]
			rbm.calcPCDGrad(batch)
			optimizer.update()

		optimizer.learnRate *= 0.9
		print("Finished epoch %d" % (epoch+1))

		if (epoch+1) % 5 == 0:
			filters = rbm.W.get().T
			showImageBatchInFolder(filters.reshape(500, 1, 28, 28), "../../TestData/rbm", "filter")


if __name__ == "__main__":
	unittest()
