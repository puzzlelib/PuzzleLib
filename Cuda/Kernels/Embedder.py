import numpy as np

from PuzzleLib.Cuda.GPUArray import GPUArray
from PuzzleLib.Cuda.SourceModule import SourceModule
from PuzzleLib.Cuda.Utils import device, warpSize, roundUpDiv, memoryPool as memPool


embedTmpl = """

extern "C"
__global__ void embed(float *outdata, const int *indata, const float *vocabulary, int size, int embsize)
{
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idy >= size || idx >= embsize) return;

	int wordidx = indata[idy];
	if (wordidx == -1) return;

	outdata[embsize * idy + idx] = vocabulary[embsize * wordidx + idx];
}

extern "C"
__global__ void embedBackwardParams(float *vocabulary, const float *outgrad, const int *indata,
									float scale, int size, int embsize)
{
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idy >= size || idx >= embsize) return;

	int wordidx = indata[idy];
	if (wordidx == -1) return;

	float gr = scale * outgrad[embsize * idy + idx];
	atomicAdd(&vocabulary[embsize * wordidx + idx], gr);
}

"""


if device is not None:
	mod = SourceModule(embedTmpl)


def embed(data, W, allocator=memPool):
	assert data.dtype == np.int32 and W.dtype == np.float32

	batchsize, sentlen = data.shape
	_, embsize = W.shape

	outdata = GPUArray.zeros((batchsize, sentlen, embsize), dtype=np.float32, allocator=allocator)
	size = batchsize * sentlen

	block = (warpSize, warpSize, 1)
	grid = (roundUpDiv(embsize, warpSize), roundUpDiv(size, warpSize), 1)

	mod.embed(outdata, data, W, np.int32(size), np.int32(embsize), block=block, grid=grid)
	return outdata


def embedBackwardParams(indata, grad, W, scale):
	assert indata.shape == grad.shape[:2] and W.shape[1] == grad.shape[2]
	assert indata.dtype == np.int32 and grad.dtype == W.dtype and W.dtype == np.float32

	batchsize, sentlen = indata.shape
	_, embsize = W.shape

	size = batchsize * sentlen

	block = (warpSize, warpSize, 1)
	grid = (roundUpDiv(embsize, warpSize), roundUpDiv(size, warpSize), 1)

	mod.embedBackwardParams(
		W, grad, indata, np.float32(scale), np.int32(size), np.int32(embsize), block=block, grid=grid
	)


def unittest():
	batchsize, sentlen, embsize = 10, 5, 20
	vocabsize = 1000

	hostInData = np.random.randint(low=-1, high=vocabsize, size=(batchsize, sentlen), dtype=np.int32)
	hostW = np.random.randn(vocabsize, embsize).astype(np.float32)

	indata, W = GPUArray.toGpu(hostInData), GPUArray.toGpu(hostW)
	outdata = embed(indata, W)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	for b in range(batchsize):
		for s in range(sentlen):
			wordidx = int(hostInData[b, s])

			if wordidx != -1:
				hostOutData[b, s] = hostW[wordidx]

	assert np.allclose(hostOutData, outdata.get())

	learnRate = 0.1
	hostGrad = np.random.randn(*outdata.shape).astype(np.float32)

	grad = GPUArray.toGpu(hostGrad)
	embedBackwardParams(indata, grad, W, learnRate)

	hostGrad = grad.get()
	for b in range(batchsize):
		for s in range(sentlen):
			wordidx = int(hostInData[b, s])

			if wordidx != -1:
				hostW[wordidx] += learnRate * hostGrad[b, s]

	assert np.allclose(hostW, W.get())


if __name__ == "__main__":
	unittest()
