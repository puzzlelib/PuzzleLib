import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Utils import memoryPool as memPool, context, queue
from PuzzleLib.OpenCL.Kernels.Utils import roundUp, nthreads
from PuzzleLib.OpenCL.Wrappers import CLBlas


preluTmpl = """

__kernel void prelu(__global float *outdata, __global const float *indata, __global const float *slopes, int divFactor,
					int mapsize, int maps, int size)
{
	for (int index = get_global_id(0); index < size; index += get_global_size(0))
	{
		const int c = (index / mapsize) % maps / divFactor;
		outdata[index] = indata[index] > 0.0f ? indata[index] : indata[index] * slopes[c];
	}
}

__kernel void preluBackwardData(__global float *ingrad, __global const float *outgrad, __global const float *slopes,
								__global const float *indata, int divFactor, int mapsize, int maps, int size)
{
	for (int index = get_global_id(0); index < size; index += get_global_size(0))
	{
		const int c = (index / mapsize) % maps / divFactor;
		ingrad[index] = outgrad[index] * ((indata[index] > 0.0f) + (indata[index] <= 0.0f) * slopes[c]);
	}
}

__kernel void preluBackwardParams(__global float *slopegrad, __global const float *outgrad,
								  __global const float *indata, int batchsize, int stride, int size)
{
	for (int index = get_global_id(0); index < size; index += get_global_size(0))
	{
		slopegrad[index] = outgrad[index] * indata[index] * (indata[index] <= 0.0f);

		for (int b = 1; b < batchsize; b++)
			slopegrad[index] += outgrad[index + b * stride] * indata[index + b * stride] *
								(indata[index + b * stride] <= 0.0f);
	}
}

"""


if context:
	mod = Driver.Program(context, preluTmpl).build()


def prelu(data, slopes, inplace=False, sharedMaps=False):
	assert data.dtype == slopes.dtype and slopes.dtype == np.float32

	if sharedMaps:
		assert slopes.shape == (1, )
	else:
		assert data.shape[1] == slopes.shape[0]

	outdata = data if inplace else Driver.empty(queue, data.shape, dtype=np.float32, allocator=memPool)

	kernel = mod.prelu

	mapsize = np.prod(data.shape[2:])
	size = int(np.prod(data.shape))

	block = (nthreads, 1, 1)
	grid = (roundUp(size, nthreads), 1, 1)

	divFactor = data.shape[1] if sharedMaps else 1

	kernel(queue, grid, block, outdata.data, data.data, slopes.data,
		   np.int32(divFactor), np.int32(mapsize), np.int32(data.shape[1]), np.int32(size))

	return outdata


def preluBackwardData(grad, slopes, indata, sharedMaps=False):
	assert grad.dtype == slopes.dtype and slopes.dtype == indata.dtype and indata.dtype == np.float32
	assert grad.shape == indata.shape

	if sharedMaps:
		assert slopes.shape == (1, )
	else:
		assert grad.shape[1] == slopes.shape[0]

	ingrad = Driver.empty(queue, grad.shape, dtype=np.float32, allocator=memPool)

	kernel = mod.preluBackwardData

	mapsize = np.prod(grad.shape[2:])
	size = int(np.prod(grad.shape))

	block = (nthreads, 1, 1)
	grid = (roundUp(size, nthreads), 1, 1)

	divFactor = grad.shape[1] if sharedMaps else 1

	kernel(queue, grid, block, ingrad.data, grad.data, slopes.data, indata.data,
		   np.int32(divFactor), np.int32(mapsize), np.int32(grad.shape[1]), np.int32(size))

	return ingrad


def preluBackwardParams(indata, outgrad, sharedMaps=False):
	assert indata.dtype == outgrad.dtype and outgrad.dtype == np.float32
	assert indata.shape == outgrad.shape

	kernel = mod.preluBackwardParams

	size = int(np.prod(outgrad.shape[1:]))
	stride = np.prod(outgrad.shape[1:])

	block = (nthreads, 1, 1)
	grid = (roundUp(size, nthreads), 1, 1)

	slopegrad = Driver.empty(queue, outgrad.shape[1:], dtype=np.float32, allocator=memPool)

	kernel(queue, grid, block, slopegrad.data, outgrad.data, indata.data,
		   np.int32(outgrad.shape[0]), np.int32(stride), np.int32(size))

	if sharedMaps:
		shape = (1, int(np.prod(slopegrad.shape)))
	else:
		shape = (slopegrad.shape[0], int(np.prod(slopegrad.shape[1:])))

	slopegrad = CLBlas.sumOnMatrix(slopegrad.reshape(shape), cols=False)

	return slopegrad


def unittest():
	batchsize, maps, h, w = 5, 4, 6, 6

	data = Driver.to_device(queue, np.random.randn(batchsize, maps, h, w).astype(np.float32))
	slopes = Driver.to_device(queue, np.random.randn(maps).astype(np.float32))

	outdata = prelu(data, slopes)

	hostData, hostSlopes = data.get(), slopes.get()
	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for c in range(maps):
		hostOutData[:, c] = (hostData[:, c] > 0.0) * hostData[:, c] + \
							(hostData[:, c] <= 0.0) * hostSlopes[c] * hostData[:, c]

	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))
	ingrad = preluBackwardData(grad, slopes, data)

	hostGrad = grad.get()
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
