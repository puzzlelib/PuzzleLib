from enum import Enum

import numpy as np

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import dtypesSupported, memoryPool as memPool

from PuzzleLib.Backend.Kernels.ElementWise import sigmoidKer, sigmoidDerKer, tanhKer, tanhDerKer, reluKer, reluDerKer
from PuzzleLib.Backend.Kernels.ElementWise import leakyReluKer, leakyReluDerKer, eluKer, eluDerKer
from PuzzleLib.Backend.Kernels.ElementWise import softPlusKer, softPlusDerKer, clipKer, clipDerKer

from PuzzleLib.Modules.Module import ModuleError, Module


class ActivationType(str, Enum):
	sigmoid = "sigmoid"
	tanh = "tanh"
	relu = "relu"
	leakyRelu = "leakyRelu"
	elu = "elu"
	softPlus = "softPlus"
	clip = "clip"


sigmoid = ActivationType.sigmoid
tanh = ActivationType.tanh
relu = ActivationType.relu
leakyRelu = ActivationType.leakyRelu
elu = ActivationType.elu
softPlus = ActivationType.softPlus
clip = ActivationType.clip


class Activation(Module):
	def __init__(self, activation, slc=None, inplace=False, name=None, args=()):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.gradUsesOutData = True

		self.inplace = inplace
		if inplace and Config.showWarnings:
			print("[%s] Warning: %s is using inplace flag" % (Config.libname, self))

		activation = ActivationType(activation)

		self.actFunc, self.actFuncDer = {
			ActivationType.sigmoid: (sigmoidKer, sigmoidDerKer),
			ActivationType.tanh: (tanhKer, tanhDerKer),
			ActivationType.relu: (reluKer, reluDerKer),
			ActivationType.leakyRelu: (leakyReluKer, leakyReluDerKer),
			ActivationType.elu: (eluKer, eluDerKer),
			ActivationType.softPlus: (softPlusKer, softPlusDerKer),
			ActivationType.clip: (clipKer, clipDerKer)
		}[activation]

		self.activation = activation
		self.slc = slc

		self.actArgs = args if len(args) > 0 else {
			ActivationType.leakyRelu: (0.01, ),
			ActivationType.elu: (1.0, ),
			ActivationType.clip: (0.0, 6.0)
		}.get(activation, ())


	def updateData(self, data):
		self.data = data if self.inplace else gpuarray.empty(data.shape, dtype=data.dtype, allocator=memPool)
		self.actFunc(data.dtype)(self.data, data, *self.actArgs, slice=self.slc)


	def updateGrad(self, grad):
		self.grad = grad if self.inplace else gpuarray.empty(grad.shape, dtype=grad.dtype, allocator=memPool)
		self.actFuncDer(grad.dtype)(self.grad, grad, self.data, *self.actArgs, slice=self.slc)


	def dataShapeFrom(self, shape):
		return shape


	def gradShapeFrom(self, shape):
		return shape


	def calcMode(self, T):
		if Config.backend == Config.Backend.cuda:
			if T not in {np.float16, np.float32}:
				raise ModuleError("Unsupported dtype %s" % T)

		elif T != np.float32:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


def unittest():
	actFuncs = {
		ActivationType.sigmoid: (
			lambda data: 1.0 / (1.0 + np.exp(-data)),
			lambda grad, outdata: grad * outdata * (1.0 - outdata)
		),

		ActivationType.tanh: (
			np.tanh,
			lambda grad, outdata: grad * (1.0 - outdata ** 2)
		),

		ActivationType.relu: (
			lambda data: (data > 0.0) * data,
			lambda grad, outdata: (outdata > 0.0) * grad
		),

		ActivationType.leakyRelu: (
			lambda data: data * ((data > 0.0) + (data <= 0.0) * 0.01),
			lambda grad, outdata: grad * ((outdata > 0.0) + (outdata <= 0.0) * 0.01)
		),

		ActivationType.elu: (
			lambda data: (data > 0.0) * data + (data <= 0.0) * (np.exp(data) - 1.0),
			lambda grad, outdata: grad * ((outdata > 0.0) + (outdata <= 0.0) * (outdata + 1.0))
		),

		ActivationType.softPlus: (
			lambda data: np.log(1.0 + np.exp(data)),
			lambda grad, outdata: grad * (1.0 - np.exp(-outdata))
		),

		ActivationType.clip: (
			lambda data: np.clip(data, 0.0, 6.0),
			lambda grad, outdata: grad * ((0.0 < outdata) & (outdata < 6.0))
		)
	}

	for dtype, atol in dtypesSupported():
		for acttype, hostAct in actFuncs.items():
			actTest(acttype, hostAct, dtype, atol)


def actTest(devAct, hostAct, dtype, atol):
	act = Activation(devAct)
	act.calcMode(dtype)

	hostData = np.random.randn(11, 51).astype(dtype)

	data = gpuarray.to_gpu(hostData)
	act(data)

	hostGrad = np.random.randn(*act.data.shape).astype(dtype)

	grad = gpuarray.to_gpu(hostGrad)
	act.backward(grad)

	hostActFwd, hostActBwd = hostAct

	hostOutData = hostActFwd(hostData)
	hostInGrad = hostActBwd(hostGrad, hostOutData)

	assert np.allclose(hostOutData, act.data.get(), atol=atol)
	assert np.allclose(hostInGrad, act.grad.get(), atol=atol)


if __name__ == "__main__":
	unittest()
