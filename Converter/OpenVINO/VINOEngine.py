import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.gpuarray import memoryPool as memPool

from PuzzleLib.Modules.Module import Module
from PuzzleLib.Converter.OpenVINO import Driver


def genEngineName(name):
	return "%s.xml" % name, "%s.bin" % name


class VINOEngine(Module):
	def __init__(self, enginepath, batchsize, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		xmlpath, binpath = enginepath
		self.engine = Driver.VINOEngine(batchsize, xmlpath, binpath, "CPU")

		inshape, outshape = self.engine.inshape, self.engine.outshape

		inshape = [tuple(inshape[key]) for key in sorted(inshape.keys(), key=lambda nm: nm.split(sep="_")[-1])]
		outshape = [tuple(outshape[key]) for key in sorted(outshape.keys(), key=lambda nm: nm.split(sep="_")[-1])]

		self.inshape = inshape[0] if len(inshape) == 1 else inshape
		self.outshape = outshape[0] if len(outshape) == 1 else outshape


	def updateData(self, data):
		data = data if isinstance(data, list) else [data]
		inputs = {"data_%s" % i: (d.ptr, d.nbytes) for i, d in enumerate(data)}

		outshape = [self.outshape] if not isinstance(self.outshape, list) else self.outshape

		outdata = [gpuarray.empty(shape, dtype=np.float32, allocator=memPool) for shape in outshape]
		outputs = {"outdata_%s" % i: (data.ptr, data.nbytes) for i, data in enumerate(outdata)}

		self.engine.infer(outputs, inputs)
		self.data = outdata if isinstance(self.outshape, list) else outdata[0]


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
