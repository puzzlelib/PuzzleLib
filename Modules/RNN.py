from enum import Enum

import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend import Blas
from PuzzleLib.Backend.Dnn.Rnn import RNNMode as BackendRNNMode, DirectionMode as BackendDirectionMode, createRnn
from PuzzleLib.Backend.Dnn.Rnn import updateRnnParams, acquireRnnParams, forwardRnn, backwardDataRnn, backwardParamsRnn
from PuzzleLib.Backend.Utils import split, memoryPool as memPool
from PuzzleLib.Modules.Module import ModuleError, Module
from PuzzleLib.Variable import Variable


class RNNMode(str, Enum):
	relu = "relu"
	tanh = "tanh"
	lstm = "lstm"
	gru = "gru"


class DirectionMode(str, Enum):
	uni = "uni"
	bi = "bi"


class WeightModifier(str, Enum):
	orthogonal = "orthogonal"
	identity = "identity"


class RNN(Module):
	def __init__(self, insize, hsize, layers=1, mode="relu", direction="uni", dropout=0.0, getSequences=False,
				 initscheme=None, modifier="orthogonal", wscale=1.0, hintBatchSize=None, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.gradUsesOutData = True

		self.insize = insize
		self.hsize = hsize
		self.layers = layers

		self.mode = RNNMode(mode)
		self.direction = DirectionMode(direction)

		self.dropout = dropout
		self.getSequences = getSequences

		self.hintBatchSize = hintBatchSize

		mode = {
			RNNMode.relu: BackendRNNMode.relu,
			RNNMode.tanh: BackendRNNMode.tanh,
			RNNMode.lstm: BackendRNNMode.lstm,
			RNNMode.gru: BackendRNNMode.gru
		}[self.mode]

		direction = {
			DirectionMode.uni: BackendDirectionMode.uni,
			DirectionMode.bi: BackendDirectionMode.bi
		}[self.direction]

		self.descRnn, W, params = createRnn(
			insize, hsize, layers, mode, direction, dropout, seed=np.random.randint(1 << 31), batchsize=hintBatchSize
		)

		self.W = None
		self.setVar("W", Variable(W))

		self.params = params
		self.initParams(initscheme, wscale, modifier)

		self.reserve, self.fulldata, self.dw = None, None, None


	def initParams(self, initscheme, wscale, modifier):
		modifier = WeightModifier(modifier)
		layers = (self.params[key] for key in sorted(self.params.keys()))

		for layer in layers:
			for paramName, param in sorted(layer.items()):
				if paramName.startswith("b"):
					param.fill(0.0)

				else:
					if paramName.startswith("r"):
						if modifier == WeightModifier.orthogonal:
							a = np.random.normal(0.0, 1.0, param.shape)
							u, _, v = np.linalg.svd(a, full_matrices=False)

							W = u if u.shape == param.shape else v
							W = W[:param.shape[0], :param.shape[1]].astype(np.float32)

						elif modifier == WeightModifier.identity:
							W = np.identity(param.shape[0], dtype=np.float32)

						else:
							raise NotImplementedError(modifier)

					else:
						W = self.createTensorWithScheme(initscheme, param.shape, wscale)

						if W is None:
							continue

					param.set(W)

		self.updateDeviceMemory()


	def updateDeviceMemory(self):
		updateRnnParams(self.descRnn, self.W, self.params)


	def setVar(self, name, var):
		if name == "W" and hasattr(self, "params"):
			_, self.params = acquireRnnParams(self.descRnn, w=var.data)

		super().setVar(name, var)


	def updateData(self, data):
		if self.train:
			self.fulldata, self.reserve = forwardRnn(data, self.W, self.descRnn)
		else:
			self.fulldata = forwardRnn(data, self.W, self.descRnn, test=True)

		if self.direction == DirectionMode.uni:
			self.data = self.fulldata if self.getSequences else self.fulldata[-1]

		else:
			if self.getSequences:
				self.data = self.fulldata
			else:
				fwddata, bwddata = self.fulldata[-1], self.fulldata[0]
				sections = (self.hsize, self.hsize)

				self.data = [split(fwddata, sections, axis=1)[0], split(bwddata, sections, axis=1)[1]]


	def updateGrad(self, grad):
		if self.getSequences:
			fullgrad = grad
		else:
			seqlen = self.fulldata.shape[0]

			if self.direction == DirectionMode.uni:
				fullgrad = gpuarray.empty((seqlen, ) + grad.shape, dtype=grad.dtype, allocator=memPool)
				fullgrad[:seqlen - 1].fill(0.0)
				fullgrad[seqlen - 1].set(grad)

			else:
				fwdgrad, bwdgrad = grad
				batchsize, hsize = fwdgrad.shape[0], 2 * self.hsize

				fullgrad = gpuarray.zeros((seqlen, batchsize, hsize), dtype=fwdgrad.dtype, allocator=memPool)

				fullgrad[0, :, bwdgrad.shape[1]:].set(bwdgrad)
				fullgrad[-1, :, :fwdgrad.shape[1]].set(fwdgrad)

		self.grad, self.reserve = backwardDataRnn(fullgrad, self.fulldata, self.W, self.reserve, self.descRnn)


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		self.dw = backwardParamsRnn(self.inData, self.fulldata, self.W, self.reserve, self.descRnn)
		Blas.addVectorToVector(self.dw, self.getVar("W").grad, out=self.getVar("W").grad, alpha=scale, beta=momentum)


	def checkDataShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Data must be 3d tensor")

		if self.hintBatchSize is not None and shape[1] != self.hintBatchSize:
			raise ModuleError("Data batch size must be = %s (was given %s)" % (self.hintBatchSize, shape[1]))

		if shape[2] != self.insize:
			raise ModuleError("Data must have data size = %s (was given %s)" % (self.insize, shape[2]))


	def checkGradShape(self, shape):
		if self.getSequences:
			if len(shape) != 3:
				raise ModuleError("Grad must be 3d tensor")

			if self.hintBatchSize is not None and shape[1] != self.hintBatchSize:
				raise ModuleError("Grad batch size must be = %s (was given %s)" % (self.hintBatchSize, shape[1]))

		else:
			if self.direction == DirectionMode.uni:
				if len(shape) != 2:
					raise ModuleError("Grad must be 2d matrix")

				if self.hintBatchSize is not None and shape[0] != self.hintBatchSize:
					raise ModuleError("Grad batch size must be = %s (was given %s)" % (self.hintBatchSize, shape[0]))

				if shape[-1] != self.hsize:
					raise ModuleError("Grad must have data size = %s (was given %s)" % (self.hsize, shape[2]))

			else:
				fwdshape, bwdshape = shape

				if len(fwdshape) != 2 or len(bwdshape) != 2:
					raise ModuleError("Grads must be 2d matrices")

				if self.hintBatchSize is not None and \
						(fwdshape[0] != self.hintBatchSize or bwdshape[0] != self.hintBatchSize):
					raise ModuleError("Grads batch size must be = %s (was given %s and %s)" %
									  (self.hintBatchSize, fwdshape[0], bwdshape[0]))

				if fwdshape[-1] != self.hsize or bwdshape[-1] != self.hsize:
					raise ModuleError("Grads must have data size = %s (was given %s and %s)" %
									  (self.hsize, fwdshape[1], bwdshape[1]))


	def dataShapeFrom(self, shape):
		hsize = self.hsize if self.direction == DirectionMode.uni else 2 * self.hsize

		if self.getSequences:
			return shape[:2] + (hsize, )
		else:
			return (shape[1], hsize) if self.direction == DirectionMode.uni else [(shape[1], hsize), (shape[1], hsize)]


	def gradShapeFrom(self, shape):
		seqlen = self.inData.shape[0]

		if self.getSequences:
			batchsize = shape[1]
		else:
			batchsize = shape[0] if self.direction == DirectionMode.uni else shape[0][0]

		return seqlen, batchsize, self.insize


	def reset(self):
		super().reset()
		self.reserve = None
		self.fulldata = None
		self.dw = None


def unittest():
	sequencesTest()
	lastStateTest()
	bidirectionalTest()
	trainTest()


def sequencesTest():
	seqlen, batchsize, insize, hsize = 4, 3, 4, 5

	hostData = np.random.randn(seqlen, batchsize, insize).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	rnn = RNN(insize, hsize, mode="relu", getSequences=True)
	rnn(data)

	hostOutData = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)
	params = {name: param.get() for name, param in rnn.params[0].items()}

	for d in range(seqlen):
		res = np.dot(hostData[d], params["wi"].T) + np.dot(hostOutData[d], params["ri"].T) + \
			  params["bwi"] + params["bri"]
		hostOutData[d + 1] = (res > 0.0) * res

	assert np.allclose(hostOutData[1:], rnn.data.get())

	hostGrad = np.random.randn(*rnn.data.shape).astype(np.float32)
	grad = gpuarray.to_gpu(hostGrad)

	rnn.backward(grad)

	hostAccGrad = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)
	hostInGrad = np.zeros((seqlen, batchsize, insize), dtype=np.float32)

	for d in range(seqlen):
		acc = (hostGrad[seqlen - d - 1] + np.dot(hostAccGrad[seqlen - d], params["ri"])) * (hostOutData[seqlen - d] > 0)

		hostAccGrad[seqlen - d - 1] = acc
		hostInGrad[seqlen - d - 1] = np.dot(acc, params["wi"])

	assert np.allclose(hostInGrad, rnn.grad.get())

	hostRiGrad = np.zeros(params["ri"].shape, dtype=np.float32)
	hostWiGrad = np.zeros(params["wi"].shape, dtype=np.float32)
	hostBiGrad = np.zeros(params["bwi"].shape, dtype=np.float32)

	for d in range(seqlen):
		hostRiGrad += np.dot(hostAccGrad[seqlen - d - 1].T, hostOutData[seqlen - d - 1])
		hostWiGrad += np.dot(hostAccGrad[seqlen - d - 1].T, hostData[seqlen - d - 1])
		hostBiGrad += np.sum(hostAccGrad[seqlen - d - 1], axis=0)

	_, dwparams = acquireRnnParams(rnn.descRnn, w=rnn.dw)
	dwparams = dwparams[0]

	assert np.allclose(hostRiGrad, dwparams["ri"].get())
	assert np.allclose(hostWiGrad, dwparams["wi"].get())
	assert np.allclose(hostBiGrad, dwparams["bwi"].get())
	assert np.allclose(hostBiGrad, dwparams["bri"].get())


def lastStateTest():
	seqlen, batchsize, insize, hsize = 5, 3, 6, 5

	hostData = np.random.randn(seqlen, batchsize, insize).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	rnn = RNN(insize, hsize, mode="relu", getSequences=False)
	rnn(data)

	hostOutData = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)
	params = {name: param.get() for name, param in rnn.params[0].items()}

	for d in range(seqlen):
		res = np.dot(hostData[d], params["wi"].T) + np.dot(hostOutData[d], params["ri"].T) + \
			  params["bwi"] + params["bri"]
		hostOutData[d + 1] = (res > 0.0) * res

	assert np.allclose(hostOutData[-1], rnn.data.get())

	hostGrad = np.random.randn(*rnn.data.shape).astype(np.float32)
	grad = gpuarray.to_gpu(hostGrad)

	rnn.backward(grad)

	hostGrad = np.zeros((seqlen, batchsize, hsize), dtype=np.float32)
	hostGrad[-1] = grad.get()

	hostAccGrad = np.zeros((seqlen + 1, batchsize, hsize), dtype=np.float32)
	hostInGrad = np.zeros((seqlen, batchsize, insize), dtype=np.float32)

	for d in range(seqlen):
		acc = (hostGrad[seqlen - d - 1] + np.dot(hostAccGrad[seqlen - d], params["ri"])) * \
			  (hostOutData[seqlen - d] > 0)

		hostAccGrad[seqlen - d - 1] = acc
		hostInGrad[seqlen - d - 1] = np.dot(acc, params["wi"])

	assert np.allclose(hostInGrad, rnn.grad.get())

	hostRiGrad = np.zeros(params["ri"].shape, dtype=np.float32)
	hostWiGrad = np.zeros(params["wi"].shape, dtype=np.float32)
	hostBiGrad = np.zeros(params["bwi"].shape, dtype=np.float32)

	for d in range(seqlen):
		hostRiGrad += np.dot(hostAccGrad[seqlen - d - 1].T, hostOutData[seqlen - d - 1])
		hostWiGrad += np.dot(hostAccGrad[seqlen - d - 1].T, hostData[seqlen - d - 1])
		hostBiGrad += np.sum(hostAccGrad[seqlen - d - 1], axis=0)

	_, dwparams = acquireRnnParams(rnn.descRnn, w=rnn.dw)
	dwparams = dwparams[0]

	assert np.allclose(hostRiGrad, dwparams["ri"].get())
	assert np.allclose(hostWiGrad, dwparams["wi"].get())
	assert np.allclose(hostBiGrad, dwparams["bwi"].get())
	assert np.allclose(hostBiGrad, dwparams["bri"].get())


def bidirectionalTest():
	seqlen, batchsize, insize, hsize = 4, 3, 4, 5

	hostData = np.random.randn(seqlen, batchsize, insize).astype(np.float32)
	data = gpuarray.to_gpu(hostData)

	rnn = RNN(insize, hsize, mode="relu", direction="bi", getSequences=False)
	rnn(data)

	hostOutData = np.zeros((seqlen + 2, batchsize, 2 * hsize), dtype=np.float32)
	params = {layernm: {name: param.get() for name, param in layer.items()} for layernm, layer in rnn.params.items()}

	for d in range(seqlen):
		res = np.dot(hostData[d], params[0]["wi"].T) + np.dot(hostOutData[d, :, :hsize], params[0]["ri"].T) + \
			  params[0]["bwi"] + params[0]["bri"]
		hostOutData[d + 1, :, :hsize] = (res > 0.0) * res

		res = np.dot(hostData[seqlen - d - 1], params[1]["wi"].T) + \
			  np.dot(hostOutData[seqlen + 1 - d, :, hsize:], params[1]["ri"].T) + params[1]["bwi"] + params[1]["bri"]
		hostOutData[seqlen - d, :, hsize:] = (res > 0.0) * res

	hostFwdOutData, hostBwdOutData = hostOutData[-2, :, :hsize], hostOutData[1, :, hsize:]

	assert np.allclose(hostFwdOutData, rnn.data[0].get())
	assert np.allclose(hostBwdOutData, rnn.data[1].get())

	hostEndGrad = [
		np.random.randn(*rnn.data[0].shape).astype(np.float32),
		np.random.randn(*rnn.data[1].shape).astype(np.float32)
	]

	grad = [gpuarray.to_gpu(gr) for gr in hostEndGrad]
	rnn.backward(grad)

	hostGrad = np.zeros((seqlen, batchsize, 2 * hsize), dtype=np.float32)
	hostGrad[-1, :, :hsize], hostGrad[0, :, hsize:] = hostEndGrad[0], hostEndGrad[1]

	hostAccGrad = np.zeros((seqlen + 2, batchsize, 2 * hsize), dtype=np.float32)
	hostInGrad = np.zeros((seqlen, batchsize, insize), dtype=np.float32)

	for d in range(seqlen):
		acc = (hostGrad[seqlen - d - 1, :, :hsize] +
			   np.dot(hostAccGrad[seqlen + 1 - d, :, :hsize], params[0]["ri"])) * \
			  (hostOutData[seqlen - d, :, :hsize] > 0)

		hostAccGrad[seqlen - d, :, :hsize] = acc
		hostInGrad[seqlen - d - 1] += np.dot(acc, params[0]["wi"])

		acc = (hostGrad[d, :, hsize:] + np.dot(hostAccGrad[d, :, hsize:], params[1]["ri"])) * \
			  (hostOutData[d + 1, :, hsize:] > 0)

		hostAccGrad[d + 1, :, hsize:] = acc
		hostInGrad[d] += np.dot(acc, params[1]["wi"])

	assert np.allclose(hostInGrad, rnn.grad.get())

	hostRi0Grad = np.zeros(params[0]["ri"].shape, dtype=np.float32)
	hostRi1Grad = np.zeros(params[1]["ri"].shape, dtype=np.float32)
	hostWi0Grad = np.zeros(params[0]["wi"].shape, dtype=np.float32)
	hostWi1Grad = np.zeros(params[1]["wi"].shape, dtype=np.float32)

	hostBi0Grad = np.zeros(params[0]["bwi"].shape, dtype=np.float32)
	hostBi1Grad = np.zeros(params[1]["bwi"].shape, dtype=np.float32)

	for d in range(seqlen):
		hostRi0Grad += np.dot(hostAccGrad[seqlen - d + 1, :, :hsize].T, hostOutData[seqlen - d, :, :hsize])
		hostWi0Grad += np.dot(hostAccGrad[seqlen - d, :, :hsize].T, hostData[seqlen - d - 1])
		hostRi1Grad += np.dot(hostAccGrad[d, :, hsize:].T, hostOutData[d + 1, :, hsize:])
		hostWi1Grad += np.dot(hostAccGrad[d + 1, :, hsize:].T, hostData[d])

		hostBi0Grad += np.sum(hostAccGrad[seqlen - d, :, :hsize], axis=0)
		hostBi1Grad += np.sum(hostAccGrad[d + 1, :, hsize:], axis=0)

	_, dwparams = acquireRnnParams(rnn.descRnn, w=rnn.dw)

	assert np.allclose(hostRi0Grad, dwparams[0]["ri"].get())
	assert np.allclose(hostWi0Grad, dwparams[0]["wi"].get())
	assert np.allclose(hostRi1Grad, dwparams[1]["ri"].get())
	assert np.allclose(hostWi1Grad, dwparams[1]["wi"].get())

	assert np.allclose(hostBi0Grad, dwparams[0]["bwi"].get())
	assert np.allclose(hostBi0Grad, dwparams[0]["bri"].get())

	assert np.allclose(hostBi1Grad, dwparams[1]["bwi"].get())
	assert np.allclose(hostBi1Grad, dwparams[1]["bri"].get())


def trainTest():
	seqlen, batchsize, insize, hsize = 10, 32, 64, 32

	data = gpuarray.to_gpu(np.random.randn(seqlen, batchsize, insize).astype(np.float32))
	target = gpuarray.to_gpu(np.random.normal(0.0, 1.0, (seqlen, batchsize, hsize)).astype(np.float32))

	rnn = RNN(insize, hsize, mode="relu", getSequences=True)
	rnn(data)

	from PuzzleLib.Cost.MSE import MSE
	mse = MSE()

	for i in range(200):
		learnRate = 1e-1

		rnn(data)
		error, grad = mse(rnn.data, target)

		rnn.backward(grad)
		rnn.updateParams(learnRate)

		if (i + 1) % 5 == 0:
			print("Iteration #%d error: %s" % (i + 1, error))


if __name__ == "__main__":
	unittest()
