import warnings, json, tempfile, math, os
from enum import Enum

import numpy as np
import h5py
from h5py import h5p, h5f

from PuzzleLib import Config

from PuzzleLib.Backend import gpuarray, Blas
from PuzzleLib.Variable import Variable


class ModuleError(Exception):
	pass


class InitScheme(str, Enum):
	none = "none"
	xavier = "xavier"
	xavierUniform = "xavier_uniform"
	xavierNormal = "xavier_normal"
	he = "he"
	gaussian = "gaussian"
	uniform = "uniform"


class FactorType(str, Enum):
	in_ = "in"
	out = "out"
	avg = "avg"


class MemoryUnit(str, Enum):
	mb = "mb"
	kb = "kb"


class Module:
	__slots__ = [
		"name", "blueprint",
		"vars", "attrs",
		"gradUsesOutData", "movesData", "movesGrad",
		"grad", "inData", "data",
		"train", "calctype",
		"varLoader", "attrLoader"
	]


	def __init__(self, name=None):
		self.name = name

		self.blueprint = None
		self.registerBlueprint(locals())

		self.vars = {}
		self.attrs = {}

		self.gradUsesOutData = False
		self.movesData = False
		self.movesGrad = False

		self.grad = None

		self.inData = None
		self.data = None

		self.train = False if Config.globalEvalMode else True
		self.calctype = np.float32

		self.varLoader = None
		self.attrLoader = None


	def registerBlueprint(self, args, exclude=None):
		exclude = set() if exclude is None else exclude
		ignore = {"self", "__class__"}

		self.blueprint = {key: None if key in exclude else arg for key, arg in args.items() if key not in ignore}


	def getBlueprint(self):
		return {"classname": self.__class__.__name__, "scheme": self.blueprint}


	def setVar(self, name, var):
		setattr(self, name, var.data)
		self.vars[name] = var


	def getVar(self, name):
		return self.vars[name]


	def getVarTable(self, vartable=None, name=None, root=True):
		if root and name is None:
			name = self.name if self.name is not None else ""

		vartable = {} if vartable is None else vartable

		for paramName, var in self.vars.items():
			if var not in vartable:
				vartable[var] = []

			vartable[var].append("%s%s" % (name, paramName))

		return vartable


	def setAttr(self, name, attr):
		setattr(self, name, attr)
		self.attrs[name] = attr


	def hasAttr(self, name):
		return name in self.attrs


	def node(self, *nodes):
		from PuzzleLib.Containers.Node import Node
		return Node(self, parents=None if len(nodes) == 0 else list(nodes))


	def __call__(self, data):
		if not Config.disableDtypeShapeChecks:
			self.checkDataShape(self.acquireShapesFrom(data))
			self.checkDataType(self.acquireDtypesFrom(data))

		self.data = None
		self.inData = data

		self.updateData(data)
		return self.data


	def backward(self, grad, updParamGrads=True, updGrad=True, scale=1.0, momentum=0.0):
		if not Config.disableDtypeShapeChecks:
			self.checkGradShape(self.acquireShapesFrom(grad))
			self.checkGradType(self.acquireDtypesFrom(grad))

		self.grad = None

		if updGrad:
			self.updateGrad(grad)

		if updParamGrads and self.train:
			self.accGradParams(grad, scale=scale, momentum=momentum)


	def updateData(self, data):
		raise NotImplementedError()


	def updateGrad(self, grad):
		raise NotImplementedError()


	def zeroGradParams(self):
		for var in self.vars.values():
			if var.hasUpdater:
				continue

			var.grad.fill(0)


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		pass


	def updateParams(self, learnRate):
		for var in self.vars.values():
			Blas.toVectorAddVector(var.data.ravel(), var.grad.ravel(), alpha=learnRate)


	def optimizeForShape(self, shape, memlimit=None):
		pass


	def save(self, hdf=None, varlinks=None, name=None, compress="gzip", assumeUniqueNames=False,
			 withBlueprint=False, isRoot=True):
		serialize = True if hdf is None else False

		hdf = self.ensureHdf(hdf, "w")
		varlinks = {} if varlinks is None else varlinks

		if name is None:
			name = self.name if self.name is not None else ""

		if assumeUniqueNames and len(name) > 0:
			tokens = name.split(sep=".")
			name = "%s.%s" % (tokens[0], tokens[-1])

		try:
			paramGrp, linkGrp = hdf.require_group("params"), hdf.require_group("links")

			for paramName, var in self.vars.items():
				if var in varlinks:
					idx = varlinks[var]
				else:
					idx = len(varlinks)
					paramGrp.create_dataset(str(idx), data=var.data.get(), compression=compress)
					varlinks[var] = idx

				linkGrp["%s.%s" % (name, paramName)] = idx

			if len(self.attrs) > 0:
				attrGrp = hdf.require_group("attrs")

				for attrName, attr in self.attrs.items():
					attrGrp.create_dataset(
						"%s.%s" % (name, attrName),
						data=attr.get() if isinstance(attr, gpuarray.GPUArray) else attr, compression=compress
					)

			if withBlueprint:
				hdf.create_dataset(
					"blueprint", (), dtype=h5py.special_dtype(vlen=str),
					data=json.dumps(self.getBlueprint(), indent=4, sort_keys=True)
				)

			buffer = None
			if isRoot and serialize:
				hdf.flush()
				buffer = hdf.id.get_file_image()

		except Exception as e:
			raise ModuleError("Module %s save error: %s" % (name, e))

		finally:
			if isRoot:
				hdf.close()

		return buffer


	def load(self, hdf, initvars=None, name=None, assumeUniqueNames=False, isRoot=True):
		hdf = self.ensureHdf(hdf, "r")
		initvars = {} if initvars is None else initvars

		if name is None:
			name = self.name if self.name is not None else ""

		if assumeUniqueNames and len(name) > 0:
			tokens = name.split(sep=".")
			name = "%s.%s" % (tokens[0], tokens[-1])

		with warnings.catch_warnings():
			warnings.filterwarnings("error")

			try:
				paramGrp, linkGrp = hdf["params"], hdf["links"]

				for paramName, var in self.vars.items():
					if var not in initvars:
						idx = str(linkGrp["%s.%s" % (name, paramName)][()])
						param = np.array(paramGrp[idx])

						if self.varLoader is not None:
							self.varLoader(paramName, param)
						else:
							var.data.set(param.astype(var.data.dtype, casting="safe", copy=False))

						initvars[var] = True

				if len(self.attrs) > 0:
					attrGrp = hdf["attrs"]

					for attrName, attr in self.attrs.items():
						attrVal = np.array(attrGrp["%s.%s" % (name, attrName)])

						if self.attrLoader is not None:
							self.attrLoader(attrName, attrVal)
						elif isinstance(attr, gpuarray.GPUArray):
							attr.set(attrVal.astype(attr.dtype, casting="safe", copy=False))
						else:
							np.copyto(attr, attrVal.astype(attr.dtype, casting="safe", copy=False))

			except Exception as e:
				raise ModuleError("Module %s load error: %s" % (name, e))

			finally:
				if isRoot:
					hdf.close()


	def trainMode(self):
		self.train = True
		self.reset()


	def evalMode(self):
		self.train = False
		self.reset()


	def calcMode(self, T):
		if T != np.float32:
			raise ModuleError("Unsupported dtype %s" % T)

		self.calctype = T


	def reset(self):
		self.inData, self.data, self.grad = None, None, None


	def checkDataShape(self, shape):
		pass


	def dataShapeFrom(self, shape):
		raise NotImplementedError()


	def checkDataType(self, dtype):
		self.genericCheckDataType(dtype)


	def checkGradShape(self, shape):
		pass


	def gradShapeFrom(self, shape):
		raise NotImplementedError()


	def checkGradType(self, dtype):
		self.genericCheckDataType(dtype)


	def genericCheckDataType(self, dtype):
		if isinstance(dtype, (tuple, list)):
			for d in dtype:
				self.genericCheckDataType(d)
		else:
			if dtype != self.calctype:
				raise ModuleError("Expected dtype %s, got %s" % (self.calctype, dtype))


	def __str__(self):
		return "Module %s (name: %s)" % (self.__class__.__name__, self.name)


	def numOfParams(self):
		nParams = sum(var.data.size for var in self.vars.values())
		return nParams


	def paramSize(self, unit=None):
		size = sum(var.data.nbytes for var in self.vars.values())
		return self.convertUnit(size, unit=unit) if unit is not None else size


	@staticmethod
	def convertUnit(val, unit):
		divider = {
			MemoryUnit.kb: 1024,
			MemoryUnit.mb: 1024**2
		}[unit]

		return val / divider


	@staticmethod
	def repeat(val, ntimes):
		return (val, ) * ntimes if isinstance(val, int) else tuple(val)


	@staticmethod
	def ensureHdf(file, mode):
		if isinstance(file, str) or file is None:
			driver, driverKwds = None, {}

			if file is None:
				file = tempfile.mktemp(suffix=".hdf")
				driver, driverKwds = "core", {"backing_store": False}

			dirname = os.path.dirname(os.path.abspath(file))
			if not os.path.exists(dirname):
				os.makedirs(dirname)

			return h5py.File(file, mode, libver="earliest", driver=driver, **driverKwds)

		elif isinstance(file, bytes):
			fapl = h5p.create(h5p.FILE_ACCESS)
			fapl.set_fapl_core()
			fapl.set_file_image(file)

			fid = h5f.open(tempfile.mktemp(suffix=".hdf").encode(), h5f.ACC_RDONLY, fapl=fapl)
			return h5py.File(fid)

		else:
			return file


	@classmethod
	def acquireShapesFrom(cls, data):
		return [cls.acquireShapesFrom(d) for d in data] if isinstance(data, (tuple, list)) else data.shape


	@classmethod
	def acquireDtypesFrom(cls, data):
		return [cls.acquireDtypesFrom(d) for d in data] if isinstance(data, (tuple, list)) else data.dtype


	@staticmethod
	def createTensorWithScheme(scheme, shape, wscale, factorShape=None, factorTranspose=False, dtype=np.float32):
		factorType = FactorType.in_

		if isinstance(scheme, (tuple, list)):
			if len(scheme) != 2:
				raise ValueError("Scheme tuple has %s length, expected 2" % len(scheme))

			scheme, factorType = scheme

		scheme = InitScheme(scheme) if scheme is not None else scheme
		factorType = FactorType(factorType)

		outs, ins = Module.inferNeuronsNumber(shape if factorShape is None else factorShape, factorTranspose)

		if factorType == FactorType.avg:
			factor = (outs + ins) / 2
		elif factorType == FactorType.in_:
			factor = ins
		elif factorType == FactorType.out:
			factor = outs
		else:
			raise NotImplementedError(factorType.value)

		if scheme == InitScheme.none:
			return None

		elif scheme == InitScheme.xavierUniform or scheme is None:
			nwscale = math.sqrt(3.0 / factor)
			return np.random.uniform(-nwscale, nwscale, shape).astype(dtype)

		elif scheme == InitScheme.xavierNormal or scheme == InitScheme.xavier:
			nwscale = math.sqrt(1.0 / factor)
			return np.random.normal(0, nwscale, shape).astype(dtype)

		elif scheme == InitScheme.he:
			nwscale = math.sqrt(2.0 / factor)
			return np.random.normal(0.0, nwscale, shape).astype(dtype)

		elif scheme == InitScheme.gaussian:
			return np.random.normal(0.0, wscale, shape).astype(dtype)

		elif scheme == InitScheme.uniform:
			return np.random.uniform(-wscale, wscale, shape).astype(dtype)

		else:
			raise NotImplementedError(scheme.value)


	@staticmethod
	def inferNeuronsNumber(shape, transpose):
		ndim = len(shape)

		if ndim == 1:
			return shape[0], shape[0]

		elif ndim == 2:
			neuronsIn, neuronsOut = shape

		else:
			outmaps, inmaps = shape[:2]
			receptiveFieldSize = int(np.prod(shape[2:]))

			neuronsOut, neuronsIn = outmaps * receptiveFieldSize, inmaps * receptiveFieldSize

		return (neuronsIn, neuronsOut) if transpose else (neuronsOut, neuronsIn)


def unittest():
	class TestModule(Module):
		def __init__(self, name=None):
			super().__init__(name)
			self.setVar("var", Variable(gpuarray.to_gpu(np.random.randn(10).astype(self.calctype)), withgrad=False))

		def updateData(self, data):
			raise NotImplementedError()

		def updateGrad(self, grad):
			raise NotImplementedError()

		def dataShapeFrom(self, shape):
			raise NotImplementedError()

		def gradShapeFrom(self, shape):
			raise NotImplementedError()

	module = TestModule(name="module")
	assert module.paramSize() == module.getVar("var").data.nbytes


if __name__ == "__main__":
	unittest()
