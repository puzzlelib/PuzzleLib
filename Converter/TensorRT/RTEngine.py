import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.gpuarray import getDeviceName, memoryPool as memPool

from PuzzleLib.Modules.Module import Module
from PuzzleLib.Converter.TensorRT import Driver


class RTDataType:
	float32 = Driver.RTDataType.float
	int8 = Driver.RTDataType.int8
	float16 = Driver.RTDataType.half


def genEngineName(name, dtype=RTDataType.float32, device=None):
	dtypeToName = {
		RTDataType.float32: "float32",
		RTDataType.int8: "int8",
		RTDataType.float16: "float16"
	}

	device = getDeviceName().replace(" ", "_") if device is None else device
	return "%s.%s.%s.engine" % (name, dtypeToName[dtype], device)


class RTEngine(Module):
	def __init__(self, enginepath, log=True, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		self.engine = Driver.RTEngine(enginepath, log)

		inshape = [(self.engine.batchsize, ) + tuple(shape) for shape in self.engine.inshape]
		outshape = [(self.engine.batchsize, ) + tuple(shape) for shape in self.engine.outshape]

		self.inshape = inshape[0] if len(inshape) == 1 else inshape
		self.outshape = outshape[0] if len(outshape) == 1 else outshape


	def updateData(self, data):
		data = [data] if not isinstance(data, list) else data
		batchsize = data[0].shape[0]

		outshape = [self.outshape] if not isinstance(self.outshape, list) else self.outshape
		outdata = [gpuarray.empty((batchsize, ) + shape[1:], dtype=np.float32, allocator=memPool) for shape in outshape]

		bindings = [d.ptr for d in data] + [d.ptr for d in outdata]
		self.engine.enqueue(batchsize, bindings)

		self.data = outdata if isinstance(self.outshape, list) else outdata[0]


	def updateGrad(self, grad):
		assert False


	def dataShapeFrom(self, shape):
		return self.outshape


	def checkDataShape(self, shape):
		if isinstance(shape, list):
			for i, sh in enumerate(shape):
				if sh[1:] != self.inshape[1:]:
					raise ValueError("Shape %s is not equal to shape %s on index %s" % (sh[1:], self.inshape[i][1:], i))

				if sh[0] > self.inshape[0] or sh[0] != shape[0][0]:
					raise ValueError("Bad batch size %s on index %s (maximum=%s)" % (sh[0], i, self.inshape[0]))

		else:
			if shape[1:] != self.inshape[1:]:
				raise ValueError("Data shape must be equal %s (was given %s)" % (self.inshape[1:], shape[1:]))

			if shape[0] > self.inshape[0]:
				raise ValueError("Maximum batch size is %s (was given %s)" % (self.inshape[0], shape[0]))


	def gradShapeFrom(self, shape):
		assert False


	def checkGradShape(self, shape):
		assert False
