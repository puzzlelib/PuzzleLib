from string import Template

import numpy as np

from PuzzleLib.Compiler.Codegen.Types import float_t

from PuzzleLib.Cuda.Utils import prod, roundUpDiv
from PuzzleLib.Cuda.Kernels.MatVec import MatModule


preluTmpl = Template("""

extern "C"
__global__ void prelu($T *outdata, const $T *indata, const float *slopes, int divFactor,
					  int mapsize, int maps, int size)
{
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x)
	{
		int c = (index / mapsize) % maps / divFactor;

		float data = (float)indata[index];
		outdata[index] = data * (data > 0.0f ? 1.0f : slopes[c]);
	}
}

extern "C"
__global__ void preluBackwardData($T *ingrad, const $T *outgrad, const float *slopes, const $T *indata,
								  int divFactor, int mapsize, int maps, int size)
{
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x)
	{
		int c = (index / mapsize) % maps / divFactor;

		float data = (float)indata[index];
		ingrad[index] = (float)outgrad[index] * ((data > 0.0f) + (data <= 0.0f) * slopes[c]);
	}
}

extern "C"
__global__ void preluBackwardParams(float *slopegrad, const float *outgrad, const float *indata,
									int batchsize, int stride, int size)
{
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x)
	{
		float sgrad = 0.0f;

		for (int b = 0; b < batchsize; b++)
		{
			float data = (float)indata[index + b * stride];
			sgrad += outgrad[index + b * stride] * data * (data <= 0.0f);
		}

		slopegrad[index] = sgrad;
	}
}

""")


class PReluModule:
	def __init__(self, matmod):
		backend = matmod.backend
		self.GPUArray, self.nthreads = backend.GPUArray, backend.nthreads

		self.mod = backend.SourceModule(preluTmpl.substitute(T=float_t))
		self.matmod = matmod


	def prelu(self, data, slopes, inplace=False, sharedMaps=False, allocator=None):
		assert data.dtype == slopes.dtype and slopes.dtype == np.float32
		assert slopes.shape == (1, ) if sharedMaps else data.shape[1] == slopes.shape[0]

		outdata = data if inplace else self.GPUArray.empty(data.shape, dtype=np.float32, allocator=allocator)

		mapsize = prod(data.shape[2:])
		size = prod(data.shape)

		block = (self.nthreads, 1, 1)
		grid = (roundUpDiv(size, self.nthreads), 1, 1)

		divFactor = data.shape[1] if sharedMaps else 1

		self.mod.prelu(
			outdata, data, slopes, np.int32(divFactor), np.int32(mapsize), np.int32(data.shape[1]), np.int32(size),
			block=block, grid=grid
		)

		return outdata


	def preluBackwardData(self, grad, slopes, indata, sharedMaps=False, allocator=None):
		assert grad.dtype == slopes.dtype and slopes.dtype == indata.dtype and indata.dtype == np.float32
		assert grad.shape == indata.shape
		assert slopes.shape == (1, ) if sharedMaps else grad.shape[1] == slopes.shape[0]

		ingrad = self.GPUArray.empty(grad.shape, dtype=np.float32, allocator=allocator)

		mapsize = prod(grad.shape[2:])
		size = prod(grad.shape)

		block = (self.nthreads, 1, 1)
		grid = (roundUpDiv(size, self.nthreads), 1, 1)

		divFactor = grad.shape[1] if sharedMaps else 1

		self.mod.preluBackwardData(
			ingrad, grad, slopes, indata, np.int32(divFactor), np.int32(mapsize), np.int32(grad.shape[1]),
			np.int32(size), block=block, grid=grid
		)

		return ingrad


	def preluBackwardParams(self, indata, outgrad, sharedMaps=False, allocator=None):
		assert indata.dtype == outgrad.dtype and outgrad.dtype == np.float32
		assert indata.shape == outgrad.shape

		size = prod(outgrad.shape[1:])
		stride = prod(outgrad.shape[1:])

		block = (self.nthreads, 1, 1)
		grid = (roundUpDiv(size, self.nthreads), 1, 1)

		slopegrad = self.GPUArray.empty(outgrad.shape[1:], dtype=np.float32, allocator=allocator)

		self.mod.preluBackwardParams(
			slopegrad, outgrad, indata, np.int32(outgrad.shape[0]), np.int32(stride), np.int32(size),
			block=block, grid=grid
		)

		shape = (1, prod(slopegrad.shape)) if sharedMaps else (slopegrad.shape[0], prod(slopegrad.shape[1:]))
		return self.matmod.matsum(slopegrad.reshape(shape), axis=1)


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		backend = Backend.getBackend(deviceIdx)
		module = PReluModule(MatModule(backend))

		preluTest(module)


def preluTest(module):
	batchsize, maps, h, w = 5, 4, 6, 6

	hostData = np.random.randn(batchsize, maps, h, w).astype(np.float32)
	hostSlopes = np.random.randn(maps).astype(np.float32)

	data, slopes = module.GPUArray.toGpu(hostData), module.GPUArray.toGpu(hostSlopes)
	outdata = module.prelu(data, slopes)

	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for c in range(maps):
		hostOutData[:, c] = (hostData[:, c] > 0.0) * hostData[:, c] + \
							(hostData[:, c] <= 0.0) * hostSlopes[c] * hostData[:, c]

	assert np.allclose(hostOutData, outdata.get())

	hostGrad = np.random.randn(*outdata.shape).astype(np.float32)

	grad = module.GPUArray.toGpu(hostGrad)
	ingrad = module.preluBackwardData(grad, slopes, data)

	hostInGrad = np.empty(ingrad.shape, dtype=np.float32)

	for c in range(maps):
		hostInGrad[:, c] = hostGrad[:, c] * ((hostData[:, c] > 0.0) + (hostData[:, c] <= 0.0) * hostSlopes[c])

	assert np.allclose(hostInGrad, ingrad.get())

	slopegrad = module.preluBackwardParams(data, grad)
	hostSlopeGrad = np.empty(slopegrad.shape, dtype=np.float32)

	for c in range(maps):
		hostSlopeGrad[c] = np.sum(hostGrad[:, c] * hostData[:, c] * (hostData[:, c] <= 0.0))

	assert np.allclose(hostSlopeGrad, slopegrad.get())


if __name__ == "__main__":
	unittest()
