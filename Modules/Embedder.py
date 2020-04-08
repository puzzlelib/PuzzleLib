import os

import h5py
import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Kernels.Embedder import embed, embedBackwardParams
from PuzzleLib.Backend.Utils import dtypesSupported

from PuzzleLib.Variable import Variable
from PuzzleLib.Modules.Module import ModuleError, Module


class Embedder(Module):
	def __init__(self, vocabulary, sentlength, embsize, onVocabulary=None, initscheme="uniform", wscale=1.0,
				 learnable=True, name=None):
		super().__init__(name)
		args = dict(locals())

		self.embsize = embsize
		self.sentlength = sentlength

		self.wgrad = None
		self.learnable = learnable
		self.outgrad = None

		dt = h5py.special_dtype(vlen=str)

		if isinstance(vocabulary, dict):
			vocabsize = len(vocabulary)
			vocab = np.empty(shape=(vocabsize, ), dtype=dt)

			for word, idx in vocabulary.items():
				vocab[int(idx)] = word

		elif isinstance(vocabulary, int):
			vocabsize = vocabulary
			vocab = np.empty(shape=(0, ), dtype=dt)

		else:
			raise ModuleError("Unrecognized vocabulary parameter type")

		self.vocab = None
		self.setAttr("vocab", vocab)

		args["vocabulary"] = vocabsize
		self.registerBlueprint(args, exclude=["onVocabulary"])

		Wshape = (vocabsize, embsize)
		W = self.createTensorWithScheme(initscheme, Wshape, wscale, (embsize, vocabsize))
		if W is None:
			W = np.empty(Wshape, dtype=np.float32)

		if onVocabulary is not None:
			onVocabulary(W)

		self.W = None
		self.setVar("W", Variable(gpuarray.to_gpu(W)))

		self.loadVarHook = self.checkVarOnLoad
		self.loadAttrHook = self.checkAttrOnLoad


	def checkVarOnLoad(self, paramName, dataset):
		if paramName == "W":
			if dataset.shape[1] != self.embsize:
				raise ModuleError("Expected embedding size %s, was given %s" % (self.embsize, dataset.shape[1]))

			self.setVar("W", Variable(gpuarray.to_gpu(dataset)))

		else:
			raise ModuleError("Unknown parameter name '%s' for embedder" % paramName)


	def checkAttrOnLoad(self, attrName, dataset):
		if attrName == "vocab":
			self.setAttr("vocab", dataset)

		else:
			raise ModuleError("Unknown attribute name '%s' for embedder" % attrName)


	def getVocabulary(self):
		voc = {}

		if self.hasAttr("vocab"):
			for i in range(self.vocab.shape[0]):
				voc[self.vocab[i]] = i

		return voc


	def verifyData(self, data):
		mn, mx = gpuarray.minimum(data).get(), gpuarray.maximum(data).get()
		if mn < -1:
			raise ModuleError("Embedder data verification failed, found index %s (< -1)" % mn)

		if mx >= self.W.shape[0]:
			raise ModuleError("Embedder data verification failed, found index %s (vocabulary size is %s)" %
							  (mx, self.W.shape[0]))


	def updateData(self, data):
		if Config.verifyData:
			self.verifyData(data)

		self.data = embed(data, self.W)


	def updateGrad(self, grad):
		self.grad = None


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		self.outgrad = grad
		self.vars["W"].grad.fill(0.0)

		if self.learnable:
			embedBackwardParams(self.inData, grad, self.vars["W"].grad, scale)


	def updateParams(self, learnRate):
		if self.learnable:
			embedBackwardParams(self.inData, self.outgrad, self.W, learnRate)


	def dataShapeFrom(self, shape):
		batchsize, sentlen = shape
		return batchsize, sentlen, self.embsize


	def gradShapeFrom(self, shape):
		raise ModuleError("Gradient propagation is undefined")


	def checkDataShape(self, shape):
		if len(shape) != 2:
			raise ModuleError("Data must be 2d matrix")

		if shape[1] != self.sentlength:
			raise ModuleError("Expected %d data sentence length, %d was given" % (self.sentlength, shape[1]))


	def checkGradShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Grad must be 3d tensor")

		batchsize, sentlen, embsize = shape
		if sentlen != self.sentlength:
			raise ModuleError("Expected %d grad sentence length, %d was given" % (self.sentlength, shape[1]))

		if embsize != self.embsize:
			raise ModuleError("Expected %d grad embedding size, %d was given" % (self.embsize, embsize))

		if batchsize != self.inData.shape[0]:
			raise ModuleError("Expected %d grad batch size, %d was given" % (self.inData.shape[0], batchsize))


	def checkDataType(self, dtype):
		if dtype != np.int32:
			raise ModuleError("Expected int32-tensor (got dtype %s)" % dtype)


	def reset(self):
		super().reset()
		self.outgrad = None


	def calcMode(self, T):
		if Config.backend in {Config.Backend.cuda, Config.Backend.hip}:
			if self.calctype == T:
				return

			variables = self.vars
			self.vars = {}

			for varName, var in variables.items():
				self.setVar(varName, Variable(
					var.data.astype(T), name=var.name, grad=var.grad.astype(T) if var.grad is not None else None
				))

			self.calctype = T

		else:
			super().calcMode(T)


def unittest():
	for dtype, atol in dtypesSupported():
		calcTest(dtype, atol)

	verifyDataTest()


def calcTest(dtype, atol):
	batchsize, sentlength, embsize = 10, 20, 40
	vocabsize = 1000

	hostData = np.random.randint(low=-1, high=vocabsize, size=(batchsize, sentlength), dtype=np.int32)
	data = gpuarray.to_gpu(hostData)

	embedder = Embedder(vocabsize, sentlength, embsize)
	embedder.calcMode(dtype)

	embedder(data)

	hostW = embedder.W.get()
	hostOutData = np.zeros(embedder.data.shape, dtype=dtype)

	for b in range(batchsize):
		for s in range(sentlength):
			wordidx = int(hostData[b, s])

			if wordidx != -1:
				hostOutData[b, s] = hostW[wordidx]

	assert embedder.getVocabulary() == {}
	assert np.allclose(hostOutData, embedder.data.get())

	hostGrad = np.random.randn(*embedder.data.shape).astype(dtype)
	grad = gpuarray.to_gpu(hostGrad)

	embedder.backward(grad)

	learnRate = 1e-1
	embedder.updateParams(learnRate)

	for b in range(batchsize):
		for s in range(sentlength):
			wordidx = int(hostData[b, s])

			if wordidx != -1:
				hostW[wordidx] += learnRate * hostGrad[b, s]

	assert np.allclose(hostW, embedder.W.get(), atol=atol)

	embedder.save("../TestData/embedder.hdf")
	embedder = Embedder(vocabsize, sentlength, embsize)
	embedder.load("../TestData/embedder.hdf")

	assert np.allclose(hostW, embedder.W.get(), atol=atol)
	os.remove("../TestData/embedder.hdf")


def verifyDataTest():
	batchsize, sentlength, embsize = 10, 20, 40
	vocabsize = 1000

	hostData = np.random.randint(low=-1, high=vocabsize, size=(batchsize, sentlength), dtype=np.int32)
	hostData[-1, -1] = vocabsize

	data = gpuarray.to_gpu(hostData)
	embedder = Embedder(vocabsize, sentlength, embsize)

	Config.verifyData = True

	try:
		embedder(data)
	except ModuleError as e:
		print("Caught data verification error: %s" % e)


if __name__ == "__main__":
	unittest()
