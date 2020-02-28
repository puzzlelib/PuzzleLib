import json

import numpy as np
import h5py

from PuzzleLib.Modules.Module import Module, ModuleError


class ContainerError(ModuleError):
	pass


class Container(Module):
	__slots__ = ["modules"]


	def __init__(self, name=None):
		super().__init__(name)
		self.modules = {}


	def getBlueprint(self):
		blueprint = super().getBlueprint()
		blueprint["modules"] = {name: mod.getBlueprint() for name, mod in self.modules.items()}

		return blueprint


	def append(self, mod, acquire=True):
		mod.name = str(len(self.modules)) if mod.name is None else mod.name

		if mod.name in self.modules:
			if acquire:
				mod.name = str(len(self.modules))
			else:
				raise ContainerError("Module with name '%s' is already in container" % mod.name)

		self.modules[mod.name] = mod
		return self


	def removeModule(self, mod):
		self.modules.pop(mod.name)
		return mod


	def getByName(self, name):
		mod = None

		if name in self.modules:
			mod = self.modules[name]
		else:
			for m in self.modules.values():
				if isinstance(m, Container):
					mod = m.getByName(name)

					if mod is not None:
						break

		return mod


	def getAllByType(self, typ):
		lst = []

		for mod in self.modules.values():
			if isinstance(mod, typ):
				lst.append(mod)

			elif isinstance(mod, Container):
				lst.extend(mod.getAllByType(typ))

		return lst


	def __getitem__(self, item):
		if isinstance(item, str):
			return self.modules[item]
		else:
			raise NotImplementedError(type(item).__name__)


	def setVar(self, name, var):
		sep = name.index(".")
		if sep == -1:
			raise ContainerError("Cannot find dot-delimiter in variable name: %s" % name)

		self.modules[name[:sep]].setVar(name[sep+1:], var)


	def getVar(self, name):
		sep = name.index(".")
		if sep == -1:
			raise ContainerError("Cannot find dot-delimiter in variable name: %s" % name)

		return self.modules[name[:sep]].getVar(name[sep+1:])


	def getVarTable(self, vartable=None, name=None, root=True):
		name = "" if root else name
		vartable = {} if vartable is None else vartable

		for mod in self.modules.values():
			mod.getVarTable(vartable, "%s%s." % (name, mod.name), root=False)

		return vartable


	def setAttr(self, name, attr):
		ctrName = self.name if self.name else ""
		self.attrs["%s.%s" % (ctrName, name)] = attr


	def getAttr(self, name):
		ctrName = self.name if self.name else ""
		return self.attrs["%s.%s" % (ctrName, name)]


	def hasAttr(self, name):
		ctrName = self.name if self.name else ""
		return ("%s.%s" % (ctrName, name)) in self.attrs


	def zeroGradParams(self):
		for mod in self.modules.values():
			mod.zeroGradParams()


	def updateParams(self, learnRate):
		for mod in self.modules.values():
			mod.updateParams(learnRate)


	def genericCheckDataType(self, dtype):
		pass


	def save(self, hdf=None, varlinks=None, name=None, compress="gzip", assumeUniqueNames=False, withBlueprint=False,
			 isRoot=True):
		serialize = True if hdf is None else False

		hdf = self.ensureHdf(hdf, "w")
		varlinks = {} if varlinks is None else varlinks

		if name is None:
			name = self.name if self.name is not None else ""

		try:
			for mod in self.modules.values():
				mod.save(
					hdf, varlinks, "%s.%s" % (name, mod.name), compress=compress,
					assumeUniqueNames=assumeUniqueNames, isRoot=False
				)

			attrGrp = hdf.require_group("attrs.%s" % name)
			for attrName, attr in self.attrs.items():
				attrGrp.create_dataset(attrName, data=attr)

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
			raise ContainerError("Container %s save error: %s" % (name, e))

		finally:
			if isRoot:
				hdf.close()

		return buffer


	def load(self, hdf, initvars=None, name=None, assumeUniqueNames=False, isRoot=True):
		hdf = self.ensureHdf(hdf, "r")
		initvars = {} if initvars is None else initvars

		if name is None:
			name = self.name if self.name is not None else ""

		try:
			for mod in self.modules.values():
				mod.load(hdf, initvars, "%s.%s" % (name, mod.name), assumeUniqueNames=assumeUniqueNames, isRoot=False)

			grpName = "attrs.%s" % name

			if grpName in hdf:
				attrGrp = hdf[grpName]
				self.attrs.update((attrName, np.array(attr)) for attrName, attr in attrGrp.items())

		except Exception as e:
			raise ContainerError("Container %s load error: %s" % (name, e))

		finally:
			if isRoot:
				hdf.close()


	def trainMode(self):
		super().trainMode()
		for mod in self.modules.values():
			mod.trainMode()


	def evalMode(self):
		super().evalMode()
		for mod in self.modules.values():
			mod.evalMode()


	def calcMode(self, T):
		for mod in self.modules.values():
			try:
				mod.calcMode(T)

			except Exception as e:
				self.handleError(mod, e)


	def reset(self):
		super().reset()
		for mod in self.modules.values():
			mod.reset()


	def __str__(self):
		return "Container %s (name: %s)" % (self.__class__.__name__, self.name)


	def handleError(self, mod, e):
		msg = str(e)
		msg = ": %s" % msg if len(msg) > 0 else ""

		raise ContainerError("%s:\nModule (%s) error:\n%s%s" % (self, mod, type(e), msg))


	def numOfParams(self):
		return sum(mod.numOfParams() for mod in self.modules.values())


	def paramSize(self, unit=None):
		size = sum(mod.paramSize(unit=None) for mod in self.modules.values())
		return self.convertUnit(size, unit=unit) if unit is not None else size


	def updateData(self, data):
		raise NotImplementedError()


	def updateGrad(self, grad):
		raise NotImplementedError()


	def dataShapeFrom(self, shape):
		raise NotImplementedError()


	def gradShapeFrom(self, shape):
		raise NotImplementedError()
