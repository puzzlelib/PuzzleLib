from enum import Enum

import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Utils import memoryPool as memPool

from PuzzleLib.Modules.Module import Module
from PuzzleLib.Converter.TensorRT import Driver


class DataType:
	float32 = Driver.DataType.float
	int8 = Driver.DataType.int8
	float16 = Driver.DataType.half


class RTEngineType(Enum):
	puzzle = Driver.RTEngineType.puzzle
	caffe = Driver.RTEngineType.caffe
	onnx = Driver.RTEngineType.onnx


def genEngineName(name, dtype, inshape, outshape):
	from PuzzleLib.Backend.Utils import backend
	arch = backend.device.name().replace(" ", "_")

	dtypes = {
		DataType.float32: "float32",
		DataType.int8: "int8",
		DataType.float16: "float16"
	}

	if not isinstance(inshape, list):
		inshape = [inshape]

	inshape = ",".join("-".join(str(s) for s in sh) for sh in inshape)

	if not isinstance(outshape, list):
		outshape = [outshape]

	outshape = ",".join("-".join(str(s) for s in sh) for sh in outshape)
	fullname = "%s.%s.%s.%s.%s.engine" % (name, dtypes[dtype], inshape, outshape, arch)

	return fullname


def parseEngineShape(enginename):
	subnames = enginename.split(sep=".")
	inshape, outshape = subnames[-4], subnames[-3]

	inshape = [tuple(int(v) for v in sh.split(sep="-")) for sh in inshape.split(sep=",")]
	inshape = inshape[0] if len(inshape) == 1 else inshape

	outshape = [tuple(int(v) for v in sh.split(sep="-")) for sh in outshape.split(sep=",")]
	outshape = outshape[0] if len(outshape) == 1 else outshape

	return inshape, outshape


class RTEngine(Module):
	def __init__(self, enginepath, enginetype, inshape=None, outshape=None, log=True, name=None):
		super().__init__(name)
		self.registerBlueprint(locals())

		if inshape is None or outshape is None:
			parsedInshape, parsedOutshape = parseEngineShape(enginepath)

			inshape = parsedInshape if inshape is None else inshape
			outshape = parsedOutshape if outshape is None else outshape

		self.inshape, self.outshape = inshape, outshape
		self.engine = Driver.RTEngine(enginepath, enginetype.value, log)


	def updateData(self, data):
		if isinstance(data, list):
			batchsize = data[0].shape[0]
			bindings = [dat.ptr for dat in data]

		else:
			batchsize = data.shape[0]
			bindings = [data.ptr]

		if isinstance(self.outshape, list):
			self.data = [
				gpuarray.empty((batchsize, ) + outshape[1:], dtype=np.float32, allocator=memPool)
				for outshape in self.outshape
			]

		else:
			self.data = gpuarray.empty((batchsize, ) + self.outshape[1:], dtype=np.float32, allocator=memPool)

		if isinstance(self.data, list):
			bindings.extend(data.ptr for data in self.data)
		else:
			bindings.append(self.data.ptr)

		self.engine.enqueue(batchsize, bindings)


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
