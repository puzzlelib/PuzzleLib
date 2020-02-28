import numpy as np

from PuzzleLib.Cuda.GPUArray import GPUArray
from PuzzleLib.Cuda.SourceModule import SourceModule
from PuzzleLib.Cuda.Utils import device, prod, roundUpDiv, nthreads, memoryPool as memPool

from PuzzleLib.Cuda.Kernels.MatVec import matsum


preluTmpl = """

extern "C"
__global__ void prelu(float *outdata, const float *indata, const float *slopes, int divFactor,
					  int mapsize, int maps, int size)
{
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x)
	{
		int c = (index / mapsize) % maps / divFactor;
		outdata[index] = indata[index] > 0.0f ? indata[index] : indata[index] * slopes[c];
	}
}

extern "C"
__global__ void preluBackwardData(float *ingrad, const float *outgrad, const float *slopes, const float *indata,
								  int divFactor, int mapsize, int maps, int size)
{
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x)
	{
		int c = (index / mapsize) % maps / divFactor;
		ingrad[index] = outgrad[index] * ((indata[index] > 0.0f) + (indata[index] <= 0.0f) * slopes[c]);
	}
}

extern "C"
__global__ void preluBackwardParams(float *slopegrad, const float *outgrad, const float *indata,
									int batchsize, int stride, int size)
{
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x)
	{
		slopegrad[index] = outgrad[index] * indata[index] * (indata[index] <= 0.0f);

		for (int b = 1; b < batchsize; b++)
			slopegrad[index] += outgrad[index + b * stride] * indata[index + b * stride] *
								(indata[index + b * stride] <= 0.0f);
	}
}

"""


if device is not None:
	mod = SourceModule(preluTmpl)


def prelu(data, slopes, inplace=False, sharedMaps=False, allocator=memPool):
	assert data.dtype == slopes.dtype and slopes.dtype == np.float32
	assert slopes.shape == (1, ) if sharedMaps else data.shape[1] == slopes.shape[0]

	outdata = data if inplace else GPUArray.empty(data.shape, dtype=np.float32, allocator=allocator)

	mapsize = prod(data.shape[2:])
	size = prod(data.shape)

	block = (nthreads, 1, 1)
	grid = (roundUpDiv(size, nthreads), 1, 1)

	divFactor = data.shape[1] if sharedMaps else 1

	mod.prelu(
		outdata, data, slopes, np.int32(divFactor), np.int32(mapsize), np.int32(data.shape[1]), np.int32(size),
		block=block, grid=grid
	)

	return outdata


def preluBackwardData(grad, slopes, indata, sharedMaps=False, allocator=memPool):
	assert grad.dtype == slopes.dtype and slopes.dtype == indata.dtype and indata.dtype == np.float32
	assert grad.shape == indata.shape
	assert slopes.shape == (1, ) if sharedMaps else grad.shape[1] == slopes.shape[0]

	ingrad = GPUArray.empty(grad.shape, dtype=np.float32, allocator=allocator)

	mapsize = prod(grad.shape[2:])
	size = prod(grad.shape)

	block = (nthreads, 1, 1)
	grid = (roundUpDiv(size, nthreads), 1, 1)

	divFactor = grad.shape[1] if sharedMaps else 1

	mod.preluBackwardData(
		ingrad, grad, slopes, indata, np.int32(divFactor), np.int32(mapsize), np.int32(grad.shape[1]),
		np.int32(size), block=block, grid=grid
	)

	return ingrad


def preluBackwardParams(indata, outgrad, sharedMaps=False, allocator=memPool):
	assert indata.dtype == outgrad.dtype and outgrad.dtype == np.float32
	assert indata.shape == outgrad.shape

	size = prod(outgrad.shape[1:])
	stride = prod(outgrad.shape[1:])

	block = (nthreads, 1, 1)
	grid = (roundUpDiv(size, nthreads), 1, 1)

	slopegrad = GPUArray.empty(outgrad.shape[1:], dtype=np.float32, allocator=allocator)

	mod.preluBackwardParams(
		slopegrad, outgrad, indata, np.int32(outgrad.shape[0]), np.int32(stride), np.int32(size),
		block=block, grid=grid
	)

	shape = (1, prod(slopegrad.shape)) if sharedMaps else (slopegrad.shape[0], prod(slopegrad.shape[1:]))
	return matsum(slopegrad.reshape(shape), axis=1)


def unittest():
	batchsize, maps, h, w = 5, 4, 6, 6

	hostData = np.random.randn(batchsize, maps, h, w).astype(np.float32)
	hostSlopes = np.random.randn(maps).astype(np.float32)

	data, slopes = GPUArray.toGpu(hostData), GPUArray.toGpu(hostSlopes)
	outdata = prelu(data, slopes)

	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for c in range(maps):
		hostOutData[:, c] = (hostData[:, c] > 0.0) * hostData[:, c] + \
							(hostData[:, c] <= 0.0) * hostSlopes[c] * hostData[:, c]

	assert np.allclose(hostOutData, outdata.get())

	hostGrad = np.random.randn(*outdata.shape).astype(np.float32)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = preluBackwardData(grad, slopes, data)

	hostInGrad = np.empty(ingrad.shape, dtype=np.float32)

	for c in range(maps):
		hostInGrad[:, c] = hostGrad[:, c] * ((hostData[:, c] > 0.0) + (hostData[:, c] <= 0.0) * hostSlopes[c])

	assert np.allclose(hostInGrad, ingrad.get())

	slopegrad = preluBackwardParams(data, grad)
	hostSlopeGrad = np.empty(slopegrad.shape, dtype=np.float32)

	for c in range(maps):
		hostSlopeGrad[c] = np.sum(hostGrad[:, c] * hostData[:, c] * (hostData[:, c] <= 0.0))

	assert np.allclose(hostSlopeGrad, slopegrad.get())


if __name__ == "__main__":
	unittest()
