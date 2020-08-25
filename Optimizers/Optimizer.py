from collections import OrderedDict

import numpy as np
import h5py

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Variable import Variable


class Optimizer:
	def __init__(self, nodeinfo=None):
		self.t = 0
		self.learnRate = 0.0

		self.attrs = {"t", "learnRate"}

		self.module = None
		self.states = {}
		self.hooks = []

		self.shParams, self.shGrads = {}, {}

		self.globalState = False
		self.globalVar = OrderedDict()

		self.customVars = []
		self.nodeinfo = nodeinfo


	def setAttr(self, name, attr):
		setattr(self, name, attr)
		self.attrs.add(name)


	def getAttrDict(self):
		return {attrName: getattr(self, attrName) for attrName in self.attrs}


	def addHook(self, hook):
		if self.globalState and Config.showWarnings:
			Config.getLogger().info("Warning: adding hook to optimizer in global state mode")

		self.hooks.append(hook)


	def setupOn(self, mod, useGlobalState=False):
		if self.nodeinfo is not None:
			assert useGlobalState

		self.module = mod
		vartable = self.module.getVarTable()

		if useGlobalState:
			self.globalState = True
			self.setupGlobalState(vartable)

		else:
			self.setupLocalStates(vartable)

		if self.nodeinfo is not None:
			assert len(self.customVars) == 0


	def setupGlobalState(self, vartable):
		variables = [(names, var) for var, names in vartable.items()]
		variables = sorted(variables, key=lambda elem: elem[0][0])

		for names, var in variables:
			if var.hasUpdater:
				assert self.nodeinfo is None

				self.customVars.append(names[0])
				continue

			shape, dtype = var.data.shape, var.data.dtype.type

			shParams = self.shParams.get(dtype, gpuarray.SharedArray(dtype))
			shGrads = self.shGrads.get(dtype, gpuarray.SharedArray(dtype))

			shParams.register(var.data.shape, var.data.dtype.type, names[0])
			shGrads.register(var.grad.shape, var.grad.dtype.type, names[0])

			self.shParams[dtype] = shParams
			self.shGrads[dtype] = shGrads

		for shParams, shGrads in zip(self.shParams.values(), self.shGrads.values()):
			shParams.build()
			shGrads.build()

			self.globalVar[shParams.dtype] = Variable(shParams.ary, grad=shGrads.ary)

		for names, var in variables:
			if var.hasUpdater:
				continue

			dtype = var.data.dtype.type
			data, grad = self.shParams[dtype][names[0]], self.shGrads[dtype][names[0]]

			data.set(var.data)
			grad.set(var.grad)

			for name in names:
				self.module.setVar(name, Variable(data, grad=grad))

		for dtype, globalVar in self.globalVar.items():
			if self.nodeinfo is not None:
				self.nodeinfo.broadcastBuffer("data", globalVar.data.gpudata)

			self.states[dtype] = self.setupState(globalVar)


	def setupLocalStates(self, vartable):
		for var, names in vartable.items():
			if var.hasUpdater:
				self.customVars.append(names[0])
				continue

			self.states[names[0]] = self.setupState(var)


	def zeroGradParams(self):
		self.zeroGradGlobalParams() if self.globalState else self.zeroGradLocalParams()


	def zeroGradGlobalParams(self):
		for globalVar in self.globalVar.values():
			globalVar.grad.fill(0)


	def zeroGradLocalParams(self):
		for i, (name, state) in enumerate(self.states.items()):
			var = self.module.getVar(name)

			if var.hasUpdater:
				continue

			var.grad.fill(0)


	def setupState(self, var):
		return {}


	def update(self, useStreams=False, sync=True):
		self.t += 1

		if self.globalState:
			self.updateGlobalState()
		else:
			self.updateLocalStates(useStreams, sync)

		for name in self.customVars:
			var = self.module.getVar(name)
			var.update(self.learnRate)


	def updateGlobalState(self):
		for dtype, globalVar in self.globalVar.items():
			state = self.states[dtype]

			for hook in self.hooks:
				hook(globalVar, state)

			if self.nodeinfo is not None:
				self.nodeinfo.sumTensor("grad", globalVar.grad)

			if globalVar.learnRate > 0.0:
				self.updateVar(globalVar, state)


	def updateLocalStates(self, useStreams, sync):
		streams = gpuarray.streamManager.borrow(len(self.states)) if useStreams else None

		for i, (name, state) in enumerate(self.states.items()):
			var = self.module.getVar(name)

			assert var.grad is not None
			assert var.data.shape == var.grad.shape

			stream = streams[i] if useStreams else None

			for hook in self.hooks:
				hook(var, state, stream)

			if var.learnRate > 0.0:
				self.updateVar(var, state, stream)

		if useStreams:
			if sync:
				for stream in streams:
					stream.synchronize()

			gpuarray.streamManager.give(streams)


	def updateVar(self, var, state, stream=None):
		raise NotImplementedError()


	def save(self, hdf, name=None):
		hdf = self.ensureHdf(hdf, "w")

		if name is None:
			name = str()

		if len(self.attrs) > 0:
			attrGrp = hdf.create_group(name + ".attrs")

			for attrName, attr in self.getAttrDict().items():
				attrGrp.create_dataset(attrName, data=attr)

		if len(self.states) > 0:
			stateGrp = hdf.create_group(name + ".states")

			for stateName, state in self.states.items():
				for entityName, entity in state.items():
					stateGrp.create_dataset("%s.%s" % (stateName, entityName), data=entity.get())


	def load(self, hdf, name=None):
		hdf = self.ensureHdf(hdf, "r")

		if name is None:
			name = str()

		attrGrpName = name + ".attrs"

		if attrGrpName in hdf:
			attrGrp = hdf[attrGrpName]
			for attrName, attr in attrGrp.items():
				T = type(getattr(self, attrName))
				self.setAttr(attrName, T(np.array(attr)))

		if len(self.states) > 0:
			stateGrp = hdf[name + ".states"]

			for stateName, state in self.states.items():
				for entityName, entity in state.items():
					entity.set(np.array(stateGrp["%s.%s" % (stateName, entityName)]))


	@staticmethod
	def ensureHdf(file, mode):
		return h5py.File(file, mode) if isinstance(file, str) else file


def trainSimpleTest(optCls, dtype, *args, **kwargs):
	from PuzzleLib.Containers.Sequential import Sequential

	from PuzzleLib.Modules.Linear import Linear
	from PuzzleLib.Modules.Activation import Activation, relu
	from PuzzleLib.Modules.Cast import Cast

	from PuzzleLib.Cost.MSE import MSE

	seq = Sequential()

	seq.append(Linear(128, 64, useBias=False))
	seq.append(Activation(relu))
	seq.append(Linear(64, 32, useBias=False))
	seq.append(Activation(relu))
	seq.append(Linear(32, 16))

	seq.calcMode(dtype)
	seq.append(Cast(intype=dtype, outtype=np.float32))

	optimizer = optCls(*args, **kwargs)
	optimizer.setupOn(seq, useGlobalState=True)

	mse = MSE()

	data = gpuarray.to_gpu(np.random.randn(16, 128).astype(dtype))
	target = gpuarray.to_gpu(np.random.randn(16, 16).astype(np.float32))

	for i in range(200):
		error, grad = mse(seq(data), target)

		optimizer.zeroGradParams()
		seq.backward(grad)
		optimizer.update()

		if (i + 1) % 5 == 0:
			print("Iteration #%d error: %s" % (i + 1, error))


def trainHardTest(optCls, dtype, *args, **kwargs):
	from PuzzleLib.Containers.Sequential import Sequential

	from PuzzleLib.Modules.Conv2D import Conv2D
	from PuzzleLib.Modules.BatchNorm2D import BatchNorm2D
	from PuzzleLib.Modules.Activation import Activation, relu
	from PuzzleLib.Modules.Cast import Cast

	from PuzzleLib.Cost.MSE import MSE

	seq = Sequential()

	seq.append(Conv2D(4, 8, 5, pad=1))
	seq.append(BatchNorm2D(8))
	seq.append(Activation(relu))

	seq.append(Conv2D(8, 16, 5, pad=1))

	seq.calcMode(dtype)
	seq.append(Cast(intype=dtype, outtype=np.float32))

	optimizer = optCls(*args, **kwargs)
	optimizer.setupOn(seq, useGlobalState=True)

	mse = MSE()

	data = gpuarray.to_gpu(np.random.randn(4, 4, 5, 5).astype(dtype))
	target = gpuarray.to_gpu(np.random.randn(4, 16, 1, 1).astype(np.float32))

	for i in range(200):
		error, grad = mse(seq(data), target)

		optimizer.zeroGradParams()
		seq.backward(grad)
		optimizer.update()

		if (i + 1) % 5 == 0:
			print("Iteration #%d error: %s" % (i + 1, error))
