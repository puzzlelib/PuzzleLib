from string import Template

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Kernels.Utils import warpSize, roundUp, atomicAddTmpl
from PuzzleLib.OpenCL.Utils import memoryPool as memPool, context, queue


embedTmpl = Template("""

$atomicAdd


__kernel void embed(__global float *outdata, __global const int *indata, int off, __global const float *vocabulary,
					int size, int embsize)
{
	int idy = get_global_id(1);
	int idx = get_global_id(0);

	if (idy >= size || idx >= embsize) return;

	int wordidx = indata[idy + off];
	if (wordidx == -1) return;

	outdata[embsize * idy + idx] = vocabulary[embsize * wordidx + idx];
}

__kernel void embedBackwardParams(__global float *vocabulary, __global const float *outgrad, __global const int *indata,
								  int off, float scale, int size, int embsize)
{
	int idy = get_global_id(1);
	int idx = get_global_id(0);

	if (idy >= size || idx >= embsize) return;

	int wordidx = indata[idy + off];
	if (wordidx == -1) return;

	float gr = scale * outgrad[embsize * idy + idx];
	atomicAddCAS(&vocabulary[embsize * wordidx + idx], gr);
}

""")


if context:
	mod = Driver.Program(context, embedTmpl.substitute(atomicAdd=atomicAddTmpl)).build()


def embed(data, W):
	assert data.dtype == np.int32 and W.dtype == np.float32

	batchsize, sentlen = data.shape
	_, embsize = W.shape

	outdata = Driver.zeros(queue, (batchsize, sentlen, embsize), dtype=np.float32, allocator=memPool)

	size = batchsize * sentlen
	kernel = mod.embed

	block = (warpSize // 4, warpSize // 4, 1)
	grid = (roundUp(embsize, block[0]), roundUp(size, block[1]), 1)

	kernel(queue, grid, block, outdata.data, data.base_data, np.int32(data.item_offset), W.data, np.int32(size),
		   np.int32(embsize))
	return outdata


def embedBackwardParams(indata, grad, W, scale):
	assert indata.shape == grad.shape[:2] and W.shape[1] == grad.shape[2]
	assert indata.dtype == np.int32 and grad.dtype == W.dtype and W.dtype == np.float32

	batchsize, sentlen = indata.shape
	_, embsize = W.shape

	size = batchsize * sentlen
	kernel = mod.embedBackwardParams

	block = (warpSize // 4, warpSize // 4, 1)
	grid = (roundUp(embsize, block[0]), roundUp(size, block[1]), 1)

	kernel(queue, grid, block, W.data, grad.data, indata.base_data, np.int32(indata.item_offset), np.float32(scale),
		   np.int32(size), np.int32(embsize))


def unittest():
	batchsize, sentlen, embsize = 10, 5, 20
	vocabsize = 1000

	indata = Driver.to_device(queue, np.random.randint(low=-1, high=vocabsize, size=(batchsize, sentlen),
													   dtype=np.int32))
	W = Driver.to_device(queue, np.random.randn(vocabsize, embsize).astype(np.float32))

	outdata = embed(indata, W)

	hostInData, hostW = indata.get(), W.get()
	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	for b in range(batchsize):
		for s in range(sentlen):
			wordidx = int(hostInData[b, s])

			if wordidx != -1:
				hostOutData[b, s] = hostW[wordidx]

	assert np.allclose(hostOutData, outdata.get())

	learnRate = 0.1
	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))

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
