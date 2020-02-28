import sys

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Dnn.Basic import ConvFwdAlgo, ConvBwdDataAlgo, ConvBwdFilterAlgo
from PuzzleLib.Backend.Dnn.Basic import convNdbenchmark, convNd, convNdBackwardData, convNdBackwardParams

from PuzzleLib.Variable import Variable
from PuzzleLib.Modules.Module import ModuleError, Module


class ConvND(Module):
	def __init__(self, nd, inmaps, outmaps, size, stride=1, pad=0, dilation=1, wscale=1.0, useBias=True, name=None,
				 initscheme=None, empty=False, groups=1):
		super().__init__(name)

		self.stride = self.repeat(stride, nd)
		self.pad = self.repeat(pad, nd)
		self.dilation = self.repeat(dilation, nd)

		self.useBias = useBias
		self.groups = groups

		self.fwdAlgo, self.bwdFilterAlgo, self.bwdDataAlgo = None, None, None
		self.installDefaultAlgos()

		if inmaps % groups != 0 or outmaps % groups != 0:
			raise ModuleError(
				"Number of input and output maps must be divisible by number of groups "
				"(%d inmaps, %d outmaps, %d groups)" % (inmaps, outmaps, groups)
			)

		inmaps //= groups

		self.W = None
		self.b = None

		if empty:
			return

		Wshape = (outmaps, inmaps, *self.repeat(size, nd))
		W = self.createTensorWithScheme(initscheme, Wshape, wscale, self.calcNeuronsNumber(Wshape))

		self.setVar("W", Variable(gpuarray.empty(Wshape, dtype=self.calctype) if W is None else gpuarray.to_gpu(W)))

		if useBias:
			bshape = (1, outmaps) + self.repeat(1, nd)
			self.setVar("b", Variable(gpuarray.zeros(bshape, dtype=self.calctype)))


	def optimizeForShape(self, shape, memlimit=None):
		fwdRes, bwdFilterRes, bwdDataRes = convNdbenchmark(
			shape, self.W.shape, self.stride, self.pad, self.dilation, self.groups, transpose=False
		)

		memlimit = sys.maxsize if memlimit is None else memlimit

		self.fwdAlgo = next(ConvFwdAlgo(res.algo) for res in fwdRes if res.memory <= memlimit)
		self.bwdFilterAlgo = next(ConvBwdFilterAlgo(res.algo) for res in bwdFilterRes if res.memory <= memlimit)
		self.bwdDataAlgo = next(ConvBwdDataAlgo(res.algo) for res in bwdDataRes if res.memory <= memlimit)


	def installDefaultAlgos(self):
		if Config.backend == Config.Backend.cuda:
			self.fwdAlgo = ConvFwdAlgo.implicitGemm
			self.bwdFilterAlgo = ConvBwdFilterAlgo.algo0
			self.bwdDataAlgo = ConvBwdDataAlgo.algo0

		elif Config.backend == Config.Backend.intel:
			self.fwdAlgo = ConvFwdAlgo.auto
			self.bwdFilterAlgo = ConvBwdFilterAlgo.auto
			self.bwdDataAlgo = ConvBwdDataAlgo.auto


	def updateData(self, data):
		self.data = convNd(
			data, self.W, self.b, stride=self.stride, pad=self.pad, dilation=self.dilation,
			groups=self.groups, algo=self.fwdAlgo
		)


	def updateGrad(self, grad):
		self.grad = convNdBackwardData(
			grad, self.W, data=self.inData, stride=self.stride, pad=self.pad, dilation=self.dilation,
			groups=self.groups, algo=self.bwdDataAlgo
		)


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		convNdBackwardParams(
			self.inData, grad, self.W, self.b, stride=self.stride, pad=self.pad, dilation=self.dilation,
			groups=self.groups, wgrad=self.vars["W"].grad, bgrad=self.vars["b"].grad if self.b is not None else None,
			scale=scale, momentum=momentum, algo=self.bwdFilterAlgo
		)


	def dataShapeFrom(self, shape):
		raise NotImplementedError()


	def gradShapeFrom(self, shape):
		raise NotImplementedError()


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
