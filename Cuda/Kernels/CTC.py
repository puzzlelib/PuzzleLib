import math, random
from string import Template

import numpy as np

from PuzzleLib.Cuda.GPUArray import GPUArray
from PuzzleLib.Cuda.SourceModule import SourceModule
from PuzzleLib.Cuda.Utils import device, warpSize, memoryPool as memPool

from PuzzleLib.Cuda.Wrappers.CuDnn import context as cudnn
from PuzzleLib.Cuda.Kernels.RadixSort import scanSumTmpl, radixSortTmpl, segmentSeqTmpl


ctcTmpl = Template("""

#include <math_constants.h>


$segmentSeq


__forceinline__ __device__ float logPlus(float p1, float p2)
{
	if (p1 <= -CUDART_INF_F)
		return p2;

	if (p2 <= -CUDART_INF_F)
		return p1;

	return log1pf(expf(-fabsf(p1 - p2))) + max(p1, p2);
}


extern "C" __launch_bounds__($NT) __global__
void calcAlphas(const float * __restrict__ indata, const int * __restrict__ datalen, int T, int vocabsize,
				const int * __restrict__ labels, const int * __restrict__ offsets, float * __restrict__ alphas,
				int blank, float * __restrict__ nll, float * __restrict__ error)
{
	__shared__ int shlabels[$NV];

	int offset = offsets[blockIdx.x];
	int S = 2 * (offsets[blockIdx.x + 1] - offset) + 1;

	indata += blockIdx.x * vocabsize;
	labels += offset;

	alphas += T * (2 * offset + blockIdx.x);

	for (int i = threadIdx.x; i < S; i += $NT)
	{
		int label = (i % 2 == 0) ? blank : labels[i / 2];

		shlabels[i] = label;
		alphas[i] = (i < 2) ? logf(indata[label]) : -CUDART_INF_F;
	}

	__syncthreads();
	T = datalen[blockIdx.x];

	for (int t = 1; t < T; t++)
	{
		for (int i = threadIdx.x; i < S; i += $NT)
		{
			float prevSum = alphas[(t - 1) * S + i];

			if (i > 0)
			{
				prevSum = logPlus(prevSum, alphas[(t - 1) * S + (i - 1)]);

				if (i > 1 && shlabels[i] != blank && shlabels[i] != shlabels[i - 2])
					prevSum = logPlus(prevSum, alphas[(t - 1) * S + (i - 2)]);
			}

			alphas[t * S + i] = prevSum + logf(indata[t * gridDim.x * vocabsize + shlabels[i]]);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		float loglike = logPlus(alphas[(T - 1) * S + (S - 2)], alphas[(T - 1) * S + (S - 1)]);

		nll[blockIdx.x] = -loglike;
		atomicAdd(error, -loglike);
	}
}


extern "C" __launch_bounds__($NT) __global__
void calcBetas(const float * __restrict__ indata, const int * __restrict__ datalen, int T, int vocabsize,
			   const int * __restrict__ labels, const int * __restrict__ offsets, const float * __restrict__ alphas,
			   int blank, const float * __restrict__ nll, float * __restrict__ grad)
{
	__shared__ int indices[$NV], shlabels[$NV];

	__shared__ union
	{
		SegmentStorage segmentStorage;
		float betas[2][$NV];
	}
	cache;

	int offset = offsets[blockIdx.x];
	int S = 2 * (offsets[blockIdx.x + 1] - offset) + 1;

	indata += blockIdx.x * vocabsize;
	grad += blockIdx.x * vocabsize;

	labels += offset;
	alphas += T * (2 * offset + blockIdx.x);

	float loglike = nll[blockIdx.x];

	if (loglike >= CUDART_INF_F)
		return;

	int keys[$VT];
	for (int i = 0; i < $VT; i++)
	{
		int j = threadIdx.x + i * $NT;
		keys[i] = (j < S) ? ((j % 2 == 0) ? blank : labels[j / 2]) : 0x7FFFFFFF;

		if (j < S)
			shlabels[j] = keys[i];
	}

	SegmentResult segments = blockSegmentSeq(keys, S, indices, &cache.segmentStorage);
	T = datalen[blockIdx.x];

	int src = 0, dst = 1;

	for (int t = T - 1; t >= 0; t--)
	{
		if (t < T - 1)
		{
			for (int i = threadIdx.x; i < S; i += $NT)
			{
				float nextSum = cache.betas[src][i];

				if (i < S - 1)
				{
					nextSum = logPlus(nextSum, cache.betas[src][i + 1]);

					if (i < S - 2 && shlabels[i] != blank && shlabels[i] != shlabels[i + 2])
						nextSum = logPlus(nextSum, cache.betas[src][i + 2]);
				}

				cache.betas[dst][i] = nextSum + logf(indata[t * gridDim.x * vocabsize + shlabels[i]]);
			}

			src = (src + 1) % 2, dst = (dst + 1) % 2;
		}
		else
		{
			for (int i = threadIdx.x; i < S; i += $NT)
			{
				int offset = (T - 1) * gridDim.x * vocabsize + shlabels[i];
				cache.betas[0][i] = (i >= S - 2) ? logf(indata[offset]) : -CUDART_INF_F;
			}
		}

		__syncthreads();

		float gr[$VT];
		for (int i = 0; i < $VT; i++)
		{
			if (i >= segments.length)
				break;

			gr[i] = -CUDART_INF_F;

			for (int j = segments.start[i]; j < segments.end[i]; j++)
				gr[i] = logPlus(gr[i], alphas[t * S + indices[j]] + cache.betas[src][indices[j]]);
		}

		for (int i = threadIdx.x; i < vocabsize; i += $NT)
			grad[t * gridDim.x * vocabsize + i] = -indata[t * gridDim.x * vocabsize + i];

		__syncthreads();

		for (int i = 0; i < $VT; i++)
		{
			if (i >= segments.length)
				break;

			int offset = t * gridDim.x * vocabsize + segments.label[i];
			float data = indata[offset];

			if (data > 0.0f)
				grad[offset] += expf(gr[i] - logf(data) + loglike);
		}
	}
}

""")


def generateModule(NT, VT):
	NV = NT * VT

	scanSum = scanSumTmpl.substitute(warpSize=warpSize, NT=NT)
	radixSort = radixSortTmpl.substitute(scanSum=scanSum, warpSize=warpSize, NT=NT, VT=VT, NV=NV)
	segmentSeq = segmentSeqTmpl.substitute(radixSort=radixSort, NT=NT, VT=VT, NV=NV)

	return SourceModule(ctcTmpl.substitute(segmentSeq=segmentSeq, NT=NT, VT=VT, NV=NV))


configs = [
	(32, 1),
	(64, 1),
	(128, 1),
	(64, 3),
	(128, 2),
	(32, 9),
	(64, 6),
	(128, 4),
	(64, 9),
	(128, 6),
	(128, 9),
	(128, 10)
]


if device is not None:
	modules = [generateModule(NT, VT) for NT, VT in configs]


def ctcLoss(data, datalen, labels, lengths, blank, error=None, normalized=False, returnAlphas=False):
	T, batchsize, vocabsize = data.shape
	mx = 2 * np.max(lengths) + 1

	config = min(i for i, (NT, VT) in enumerate(configs) if mx <= NT * VT)
	mod, NT = modules[config], configs[config][0]

	if not normalized:
		data = cudnn.softmaxNd(data.reshape(T * batchsize, vocabsize, 1, 1), allocator=memPool).reshape(
			T, batchsize, vocabsize
		)

	offsets = np.cumsum(lengths, dtype=np.int32)
	extOffsets = np.empty(shape=(batchsize + 1, ), dtype=np.int32)

	extOffsets[0] = 0
	extOffsets[1:] = offsets

	alphas = GPUArray.empty((T * (2 * int(offsets[-1]) + batchsize), ), dtype=np.float32, allocator=memPool)
	offsets = GPUArray.toGpu(extOffsets, allocator=memPool)

	nll = GPUArray.empty((batchsize, ), dtype=np.float32, allocator=memPool)

	error = GPUArray.zeros((), dtype=np.float32, allocator=memPool) if error is None else error
	grad = GPUArray.zeros(data.shape, dtype=np.float32, allocator=memPool)

	mod.calcAlphas(
		data, datalen, np.int32(T), np.int32(vocabsize), labels, offsets, alphas, np.int32(blank),
		nll, error, block=(NT, 1, 1), grid=(batchsize, 1, 1)
	)

	mod.calcBetas(
		data, datalen, np.int32(T), np.int32(vocabsize), labels, offsets, alphas, np.int32(blank),
		nll, grad, block=(NT, 1, 1), grid=(batchsize, 1, 1)
	)

	return (error, grad) if not returnAlphas else (error, grad, alphas)


def unittest():
	times, batchsize, vocabsize = 20, 3, 6
	hostData, hostDataLen, hostLabels, lengths = createData(times, batchsize, vocabsize)

	data, datalen, labels = GPUArray.toGpu(hostData), GPUArray.toGpu(hostDataLen), GPUArray.toGpu(hostLabels)
	blank = 0

	error, grad, alphas = ctcLoss(data, datalen, labels, lengths, blank, returnAlphas=True)
	hostError, hostGrad, hostAlphas = ctcLossTest(hostData, hostDataLen, hostLabels, lengths, blank)

	assert np.allclose(hostAlphas, alphas.get())

	assert np.isclose(hostError, error.get())
	assert np.allclose(hostGrad, grad.get(), atol=1e-5)


def createData(times, batchsize, vocabsize):
	data = np.random.randn(times, batchsize, vocabsize).astype(np.float32)
	datalen = np.array([times] * batchsize, dtype=np.int32)

	lengths = np.array([random.randint(a=times // 4, b=times // 2 - 1) for _ in range(batchsize)], dtype=np.int32)
	labels = np.concatenate([
		np.array([random.randint(a=1, b=vocabsize - 1) for _ in range(lengths[b])], dtype=np.int32)
		for b in range(batchsize)
	])

	return data, datalen, labels, lengths


def hostSoftmax(w):
	e = np.exp(w - np.amax(w, axis=-1, keepdims=True))
	return e / np.sum(e, axis=-1, keepdims=True)


def logPlus(a, b):
	if a <= -math.inf:
		return b
	if b <= -math.inf:
		return a

	return math.log1p(math.exp(-math.fabs(a - b))) + max(a, b)


def ctcLossTest(data, datalen, labels, lengths, blank):
	data = hostSoftmax(data)

	alphas, nll = calcAlphasTest(data, datalen, labels, lengths, blank)
	grad = calcBetasTest(alphas, nll, data, datalen, labels, lengths, blank)

	return np.sum(nll), grad, alphas


def calcAlphasTest(data, datalen, labels, lengths, blank):
	T, batchsize, _ = data.shape
	offsets = np.cumsum(lengths)

	alphas = np.full((T * (2 * offsets[-1] + batchsize), ), fill_value=np.nan, dtype=np.float32)
	nll = np.empty((batchsize, ), dtype=np.float32)

	for b in range(batchsize):
		L, T = int(lengths[b]), int(datalen[b])
		S = 2 * L + 1

		offset = 0 if b == 0 else int(offsets[b - 1])

		extLabels = np.full((S, ), fill_value=blank, dtype=np.int32)
		extLabels[1::2] = labels[offset:offset + L]

		extOffset = 2 * offset + b
		alpha = alphas[extOffset * T:(extOffset + S) * T].reshape(T, S)

		alpha[0, 0] = math.log(data[0, b, blank])
		alpha[0, 1] = math.log(data[0, b, extLabels[1]])

		alpha[0, 2:] = -math.inf

		for t in range(1, T):
			alpha[t, 0] = alpha[t - 1, 0] + math.log(data[t, b, blank])

			for i in range(1, S):
				prevSum = logPlus(alpha[t - 1, i], alpha[t - 1, i - 1])

				if i > 1 and extLabels[i] != blank and extLabels[i] != extLabels[i - 2]:
					prevSum = logPlus(prevSum, alpha[t - 1, i - 2])

				alpha[t, i] = prevSum + math.log(data[t, b, extLabels[i]])

		loglike = logPlus(logPlus(-math.inf, alpha[T - 1, S - 2]), alpha[T - 1, S - 1])
		nll[b] = -loglike

	return alphas, nll


def calcBetasTest(alphas, nll, data, datalen, labels, lengths, blank):
	offsets = np.cumsum(lengths)

	_, batchsize, nlabels = data.shape
	grad = np.full(data.shape, fill_value=-math.inf, dtype=np.float32)

	for b in range(batchsize):
		L, T = int(lengths[b]), int(datalen[b])
		S = 2 * L + 1

		offset = 0 if b == 0 else int(offsets[b - 1])

		extLabels = np.full((S, ), fill_value=blank, dtype=np.int32)
		extLabels[1::2] = labels[offset:offset + L]

		extOffset = 2 * offset + b
		alpha = alphas[extOffset * T:(extOffset + S) * T].reshape(T, S)

		beta = np.empty((T, S), dtype=np.float32)
		beta[T - 1, :S - 2] = -math.inf

		beta[T - 1, S - 2] = math.log(data[T - 1, b, extLabels[S - 2]])
		beta[T - 1, S - 1] = math.log(data[T - 1, b, blank])

		for i in range(S):
			grad[T - 1, b, extLabels[i]] = logPlus(grad[T - 1, b, extLabels[i]], alpha[T - 1, i] + beta[T - 1, i])

		for i in range(nlabels):
			grad[T - 1, b, i] = data[T - 1, b, i] - math.exp(grad[T - 1, b, i] - math.log(data[T - 1, b, i]) + nll[b])

		for t in reversed(range(T - 1)):
			for i in range(S - 1):
				nextSum = logPlus(beta[t + 1, i], beta[t + 1, i + 1])

				if i < S - 2 and extLabels[i] != blank and extLabels[i] != extLabels[i + 2]:
					nextSum = logPlus(nextSum, beta[t + 1, i + 2])

				beta[t, i] = nextSum + math.log(data[t, b, extLabels[i]])

			beta[t, S - 1] = beta[t + 1, S - 1] + math.log(data[t, b, blank])

			for i in range(S):
				grad[t, b, extLabels[i]] = logPlus(grad[t, b, extLabels[i]], alpha[t, i] + beta[t, i])

			for i in range(nlabels):
				grad[t, b, i] = data[t, b, i] - math.exp(grad[t, b, i] - math.log(data[t, b, i]) + nll[b])

		grad[T:, b] = 0

	return -grad


if __name__ == "__main__":
	unittest()
