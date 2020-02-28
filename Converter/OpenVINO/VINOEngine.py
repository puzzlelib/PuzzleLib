import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool

from PuzzleLib.Modules.Module import Module
from PuzzleLib.Converter.OpenVINO import Driver


def genEngineName(name, inshape, outshape):
	inshape = ",".join("-".join(str(s) for s in sh) for sh in inshape)
	outshape = ",".join("-".join(str(s) for s in sh) for sh in outshape)

	fullname = "%s.%s.%s" % (name, inshape, outshape)
	return fullname


def parseEngineShape(enginename):
	subnames = enginename.split(sep=".")
	inshape, outshape = subnames[1], subnames[2]

	inshape = [tuple(int(v) for v in sh.split(sep="-")) for sh in inshape.split(sep=",")]
	inshape = inshape[0] if len(inshape) == 1 else inshape

	outshape = [tuple(int(v) for v in sh.split(sep="-")) for sh in outshape.split(sep=",")]
	outshape = outshape[0] if len(outshape) == 1 else outshape

	return inshape, outshape


class VINOEngine(Module):
	def __init__(self, batchsize, xmlpath, binpath, inshape=None, outshape=None, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		shapes = [inshape, outshape]

		if inshape is None or outshape is None:
			parsedInshape, parsedOutshape = parseEngineShape(xmlpath)
			shapes = [parsedInshape if inshape is None else inshape, parsedOutshape if outshape is None else outshape]

		shapes = [sh if isinstance(sh, list) else [sh] for sh in shapes]
		shapes = [[(batchsize, ) + s for s in sh] for sh in shapes]
		shapes = [sh[0] if len(sh) == 1 else sh for sh in shapes]

		self.inshape, self.outshape = shapes
		self.engine = Driver.VINOEngine(batchsize, xmlpath, binpath, "CPU")


	def updateData(self, data):
		if isinstance(self.outshape, list):
			self.data = [gpuarray.empty(outshape, dtype=np.float32, allocator=memPool) for outshape in self.outshape]
		else:
			self.data = gpuarray.empty(self.outshape, dtype=np.float32, allocator=memPool)

		data = data if isinstance(data, list) else [data]
		inputs = {"data_%s" % i: (dat.ptr, dat.nbytes) for i, dat in enumerate(data)}

		outdata = self.data if isinstance(self.data, list) else [self.data]
		outputs = {"outdata_%s" % i: (data.ptr, data.nbytes) for i, data in enumerate(outdata)}

		self.engine.infer(outputs, inputs)


	def updateGrad(self, grad):
		assert False


	def dataShapeFrom(self, shape):
		return self.outshape


	def checkDataShape(self, shape):
		if isinstance(shape, list):
			for i, sh in enumerate(shape):
				if sh != self.inshape[i]:
					raise ValueError("Shape %s is not equal to shape %s on index %s" % (sh, self.inshape[i], i))

		elif shape != self.inshape:
			raise ValueError("Data shape must be equal %s (was given %s)" % (self.inshape, shape))


	def gradShapeFrom(self, shape):
		assert False


	def checkGradShape(self, shape):
		assert False
