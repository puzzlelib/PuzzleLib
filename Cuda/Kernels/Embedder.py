import numpy as np
from PuzzleLib.Cuda.Utils import roundUpDiv


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


class EmbedModule:
	def __init__(self, backend):
		self.GPUArray, self.warpSize = backend.GPUArray, backend.warpSize

		self.mod = backend.SourceModule(embedTmpl)
		self.block = (self.warpSize, backend.nthreads // self.warpSize, 1)


	def embed(self, data, W, allocator=None):
		assert data.dtype == np.int32 and W.dtype == np.float32

		batchsize, sentlen = data.shape
		_, embsize = W.shape

		outdata = self.GPUArray.zeros((batchsize, sentlen, embsize), dtype=np.float32, allocator=allocator)
		size = batchsize * sentlen

		xblock, yblock, _ = self.block
		grid = (roundUpDiv(embsize, xblock), roundUpDiv(size, yblock), 1)

		self.mod.embed(outdata, data, W, np.int32(size), np.int32(embsize), block=self.block, grid=grid)
		return outdata


	def embedBackwardParams(self, indata, grad, W, scale):
		assert indata.shape == grad.shape[:2] and W.shape[1] == grad.shape[2]
		assert indata.dtype == np.int32 and grad.dtype == W.dtype and W.dtype == np.float32

		batchsize, sentlen = indata.shape
		_, embsize = W.shape

		size = batchsize * sentlen

		xblock, yblock, _ = self.block
		grid = (roundUpDiv(embsize, xblock), roundUpDiv(size, yblock), 1)

		self.mod.embedBackwardParams(
			W, grad, indata, np.float32(scale), np.int32(size), np.int32(embsize), block=self.block, grid=grid
		)


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		module = EmbedModule(Backend.getBackend(deviceIdx))
		embedTest(module)


def embedTest(module):
	batchsize, sentlen, embsize = 10, 5, 20
	vocabsize = 1000

	hostInData = np.random.randint(low=-1, high=vocabsize, size=(batchsize, sentlen), dtype=np.int32)
	hostW = np.random.randn(vocabsize, embsize).astype(np.float32)

	indata, W = module.GPUArray.toGpu(hostInData), module.GPUArray.toGpu(hostW)
	outdata = module.embed(indata, W)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	for b in range(batchsize):
		for s in range(sentlen):
			wordidx = int(hostInData[b, s])

			if wordidx != -1:
				hostOutData[b, s] = hostW[wordidx]

	assert np.allclose(hostOutData, outdata.get())

	learnRate = 0.1
	hostGrad = np.random.randn(*outdata.shape).astype(np.float32)

	grad = module.GPUArray.toGpu(hostGrad)
	module.embedBackwardParams(indata, grad, W, learnRate)

	hostGrad = grad.get()
	for b in range(batchsize):
		for s in range(sentlen):
			wordidx = int(hostInData[b, s])

			if wordidx != -1:
				hostW[wordidx] += learnRate * hostGrad[b, s]

	assert np.allclose(hostW, W.get())


if __name__ == "__main__":
	unittest()
