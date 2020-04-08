from enum import Enum

import numpy as np

from PuzzleLib.Cuda.Wrappers.CuDnnRnn import reluTest, tanhTest, lstmTest, gruTest

from PuzzleLib.Hip.Driver import GPUArray
from PuzzleLib.Hip.ThirdParty import libmiopen
from PuzzleLib.Hip.Wrappers.MIOpen import toDataType


class RNNMode(Enum):
	relu = libmiopen.miopenRNNMode["miopenRNNRELU"]
	tanh = libmiopen.miopenRNNMode["miopenRNNTANH"]
	lstm = libmiopen.miopenRNNMode["miopenLSTM"]
	gru = libmiopen.miopenRNNMode["miopenGRU"]


class DirectionMode(Enum):
	uni = libmiopen.miopenRNNDirectionMode["miopenRNNunidirection"]
	bi = libmiopen.miopenRNNDirectionMode["miopenRNNbidirection"]


class RNNAlgo(Enum):
	default = libmiopen.miopenRNNAlgo["miopenRNNdefault"]
	fundamental = libmiopen.miopenRNNAlgo["miopenRNNfundamental"]


class Rnn:
	def __init__(self, context, insize, hsize, dtype, layers, mode, direction):
		self.context = context

		self.insize, self.hsize, self.layers = insize, hsize, layers
		self.mode, self.direction, self.algo = mode, direction, RNNAlgo.default
		self.dtype = np.dtype(dtype).type

		self.desc = libmiopen.miopenCreateRNNDescriptor()

		dataType = toDataType[self.dtype]
		self.descData = self.context.createDescribedNdTensor(None, (1, insize), (insize, 1), dtype=dataType)

		libmiopen.miopenSetRNNDescriptor(
			self.desc, hsize, layers, libmiopen.miopenRNNInputMode["miopenRNNlinear"], direction.value, mode.value,
			libmiopen.miopenRNNBiasMode["miopenRNNwithBias"], self.algo.value, dataType.value
		)

		wsize = libmiopen.miopenGetRNNParamsSize(
			self.context.context, self.desc, self.descData.desc, dataType.value
		) // np.dtype(dtype).itemsize

		self.Wshape = (wsize, )
		self.descW = self.context.createDescribedNdTensor(None, self.Wshape, (1, ), dtype=dataType)


	def __del__(self):
		self.context.destroyDescribedTensors(self.descData, self.descW)
		libmiopen.miopenDestroyRNNDescriptor(self.desc)


	def forward(self, data, W, hidden=None, cells=None, test=False, out=None, allocator=None):
		assert data.ndim == 3 and data.shape[2] == self.insize
		assert W.shape == self.Wshape

		seqlen, batchsize, _ = data.shape
		hsize, layers = (self.hsize, self.layers) if self.direction == DirectionMode.uni else \
			(2 * self.hsize, 2 * self.layers)

		dims, strides = (layers, batchsize, self.hsize), (batchsize * self.hsize, self.hsize, 1)

		if hidden is not None:
			assert hidden.shape == dims and hidden.dtype == data.dtype
		else:
			hidden = GPUArray.zeros(dims, dtype=data.dtype, allocator=allocator)

		if cells is not None:
			assert cells.shape == dims and cells.dtype == data.dtype
		elif self.mode in {RNNMode.lstm, RNNMode.gru}:
			cells = GPUArray.zeros(dims, dtype=data.dtype, allocator=allocator)

		hptr, cptr = None if hidden is None else hidden.ptr, None if cells is None else cells.ptr

		descCells = self.context.createDescribedNdTensor(None, dims, strides, dtype=toDataType[data.dtype.type])
		out = GPUArray.empty(data.shape[:2] + (hsize, ), dtype=data.dtype, allocator=allocator) if out is None else out

		descData = self.context.createDescribedNdTensor(data[0])
		descOutData = self.context.createDescribedNdTensor(out[0])

		indescs, outdescs = [descData.desc] * seqlen, [descOutData.desc] * seqlen

		workspaceSize = libmiopen.miopenGetRNNWorkspaceSize(self.context.context, self.desc, seqlen, indescs)
		workspace = GPUArray.empty((workspaceSize, ), dtype=np.uint8, allocator=allocator)

		if test:
			libmiopen.miopenRNNForwardInference(
				self.context.context, self.desc, seqlen, indescs, data.ptr, descCells.desc, hptr, descCells.desc, cptr,
				self.descW.desc, W.ptr, outdescs, out.ptr, descCells.desc, None, descCells.desc, None,
				workspace.ptr, workspaceSize
			)

			trainReserve = None

		else:
			reserveSize = libmiopen.miopenGetRNNTrainingReserveSize(self.context.context, self.desc, seqlen, indescs)
			reserve = GPUArray.empty((reserveSize, ), dtype=np.uint8, allocator=allocator)

			libmiopen.miopenRNNForwardTraining(
				self.context.context, self.desc, seqlen, indescs, data.ptr, descCells.desc, hptr, descCells.desc, cptr,
				self.descW.desc, W.ptr, outdescs, out.ptr, descCells.desc, None, descCells.desc, None,
				workspace.ptr, workspaceSize, reserve.ptr, reserveSize
			)

			trainReserve = (workspace, reserve)

		self.context.destroyDescribedTensors(descData, descOutData, descCells)
		return out if test else (out, trainReserve)


	def backwardData(self, grad, outdata, W, trainReserve, hidden=None, cells=None, out=None, allocator=None):
		assert grad.ndim == 3 and outdata.shape == grad.shape and outdata.dtype == grad.dtype

		seqlen, batchsize, _ = grad.shape
		hsize, layers = (self.hsize, self.layers) if self.direction == DirectionMode.uni else \
			(2 * self.hsize, 2 * self.layers)

		assert W.shape == self.Wshape and grad.shape[2] == hsize
		dhidden, dcells = None, None

		dims, strides = (layers, batchsize, self.hsize), (batchsize * self.hsize, self.hsize, 1)
		hptr, cptr, dhptr, dcptr = None, None, None, None

		if hidden is not None:
			assert hidden.shape == dims and hidden.dtype == grad.dtype
			dhidden = GPUArray.empty(hidden.shape, dtype=hidden.dtype, allocator=allocator)
			hptr, dhptr = hidden.ptr, dhidden.ptr

		if cells is not None:
			assert cells.shape == dims and cells.dtype == grad.dtype
			dcells = GPUArray.empty(cells.shape, dtype=cells.dtype, allocator=allocator)
			cptr, dcptr = cells.ptr, dcells.ptr

		descCells = self.context.createDescribedNdTensor(None, dims, strides, dtype=toDataType[grad.dtype.type])
		out = GPUArray.empty(grad.shape[:2] + (self.insize, ), dtype=grad.dtype, allocator=allocator) \
			if out is None else out

		descInGrad = self.context.createDescribedNdTensor(out[0])
		descGrad = self.context.createDescribedNdTensor(grad[0])

		indescs, outdescs = [descInGrad.desc] * seqlen, [descGrad.desc] * seqlen
		workspace, reserveSpace = trainReserve

		libmiopen.miopenRNNBackwardData(
			self.context.context, self.desc, seqlen, outdescs, outdata.ptr, outdescs, grad.ptr,
			descCells.desc, None, descCells.desc, None, self.descW.desc, W.ptr,
			descCells.desc, hptr, descCells.desc, cptr, indescs, out.ptr, descCells.desc, dhptr,
			descCells.desc, dcptr, workspace.ptr, workspace.nbytes, reserveSpace.ptr, reserveSpace.nbytes
		)

		self.context.destroyDescribedTensors(descCells, descInGrad, descGrad)
		return out, dhidden, dcells


	def backwardParams(self, data, outdata, trainReserve, hidden=None, out=None, allocator=None):
		assert data.ndim == 3 and outdata.ndim == 3 and data.shape[:2] == outdata.shape[:2]

		seqlen, batchsize, _ = data.shape
		hsize, layers = (self.hsize, self.layers) if self.direction == DirectionMode.uni else \
			(2 * self.hsize, 2 * self.layers)

		assert data.shape[2] == self.insize and outdata.shape[2] == hsize
		dims, strides = (layers, batchsize, self.hsize), (batchsize * self.hsize, self.hsize, 1)

		if hidden is not None:
			assert hidden.shape == dims and hidden.dtype == data.dtype
		else:
			hidden = GPUArray.zeros(dims, dtype=data.dtype, allocator=allocator)

		hptr = None if hidden is None else hidden.ptr

		descCells = self.context.createDescribedNdTensor(None, dims, strides, dtype=toDataType[data.dtype.type])
		out = GPUArray.zeros(self.Wshape, dtype=self.dtype, allocator=allocator) if out is None else out

		descData = self.context.createDescribedNdTensor(data[0])
		descOutData = self.context.createDescribedNdTensor(outdata[0])

		indescs, outdescs = [descData.desc] * seqlen, [descOutData.desc] * seqlen
		workspace, reserveSpace = trainReserve

		libmiopen.miopenRNNBackwardWeights(
			self.context.context, self.desc, seqlen, indescs, data.ptr, descCells.desc, hptr, outdescs, outdata.ptr,
			self.descW.desc, out.ptr, workspace.ptr, workspace.nbytes, reserveSpace.ptr, reserveSpace.nbytes
		)

		self.context.destroyDescribedTensors(descCells, descData, descOutData)
		return out


def unittest():
	from PuzzleLib.Hip import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx, initmode=2)

		for dtype, atol in bnd.dtypesSupported()[:1]:
			reluTest(bnd, dtype, atol)
			tanhTest(bnd, dtype, atol)
			lstmTest(bnd, dtype, atol)
			gruTest(bnd, dtype, atol)


if __name__ == "__main__":
	unittest()
