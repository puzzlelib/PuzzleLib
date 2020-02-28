from string import Template

import numpy as np

from PuzzleLib.Compiler.Codegen.Types import half_t, float_t

from PuzzleLib.Cuda.GPUArray import GPUArray
from PuzzleLib.Cuda.SourceModule import SourceModule
from PuzzleLib.Cuda.Utils import dtypesSupported, device, warpSize, roundUpDiv, memoryPool as memPool


mapTmpl = """

#include <cuda_fp16.h>


__device__ __forceinline__ void map1d(int insize, int outsize, int index, int lpad, int *inindex, int *outindex)
{
	int inoffset = (blockIdx.y + blockIdx.z * gridDim.y) * insize;
	int outoffset = (blockIdx.y + blockIdx.z * gridDim.y) * outsize;

	int instart = max(0, -lpad), outstart = max(0, lpad);

	int x = abs(index - lpad) - abs(index - (insize + lpad - 1)) - index + 2 * lpad + insize - 1 - outstart + instart;
	*inindex = inoffset + x, *outindex = outoffset + index;
}

__device__ __forceinline__ void map2d(int inh, int inw, int outh, int outw, int index, int upad, int lpad,
									  int *inindex, int *outindex)
{
	int inoffset = (blockIdx.y + blockIdx.z * gridDim.y) * inh * inw;
	int outoffset = (blockIdx.y + blockIdx.z * gridDim.y) * outh * outw;

	int outx = index % outw, outy = index / outw;

	int instartx = max(0, -lpad), outstartx = max(0, lpad);
	int instarty = max(0, -upad), outstarty = max(0, upad);

	int inx = abs(outx - lpad) - abs(outx - (inw + lpad - 1)) - outx + 2 * lpad + inw - 1 - outstartx + instartx;
	int iny = abs(outy - upad) - abs(outy - (inh + upad - 1)) - outy + 2 * upad + inh - 1 - outstarty + instarty;

	*inindex = inoffset + iny * inw + inx;
	*outindex = outoffset + outy * outw + outx;
}

__device__ __forceinline__ void gpuAtomicAdd(float *address, float val)
{
	atomicAdd(address, val);
}

__device__ __forceinline__ void gpuAtomicAdd(half *address, half val)
{
#if __CUDA_ARCH__ < 700
	unsigned *addrUI = (unsigned *)((char *)address - ((size_t)address & 2));
	unsigned assumed, old = *addrUI;

	do
	{
		assumed = old;

		half sah = __short_as_half(((size_t)address & 2) ? (old >> 16) : (old & 0xffff));
		sah = (float)sah + (float)val;

		short has = __half_as_short(sah);
		old = ((size_t)address & 2) ? (old & 0xffff) | (has << 16) : (old & 0xffff0000) | has;

		old = atomicCAS(addrUI, assumed, old);
	}
	while (assumed != old);

#else
	atomicAdd(address, val);

#endif
}

"""


padTmpl = Template("""

extern "C"
__global__ void reflectpad1d$ext($T *outdata, const $T *indata, int insize, int lpad, int rpad)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int outsize = insize + lpad + rpad;

	if (index < outsize)
	{
		int inindex = 0, outindex = 0;
		map1d(insize, outsize, index, lpad, &inindex, &outindex);

		outdata[outindex] = indata[inindex];
	}
}

extern "C"
__global__ void reflectpad1dBackward$ext($T *ingrad, const $T *outgrad, int insize, int lpad, int rpad)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int outsize = insize + lpad + rpad;

	if (index < outsize)
	{
		int inindex = 0, outindex = 0;
		map1d(insize, outsize, index, lpad, &inindex, &outindex);

		gpuAtomicAdd(&ingrad[inindex], outgrad[outindex]);
	}
}

extern "C"
__global__ void reflectpad2d$ext($T *outdata, const $T *indata, int inh, int inw,
								 int upad, int bpad, int lpad, int rpad)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int outh = inh + upad + bpad, outw = inw + lpad + rpad;

	if (index < outh * outw)
	{
		int inindex = 0, outindex = 0;
		map2d(inh, inw, outh, outw, index, upad, lpad, &inindex, &outindex);

		outdata[outindex] = indata[inindex];
	}
}

extern "C"
__global__ void reflectpad2dBackward$ext($T *ingrad, const $T *outgrad, int inh, int inw,
										 int upad, int bpad, int lpad, int rpad)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int outh = inh + upad + bpad, outw = inw + lpad + rpad;

	if (index < outh * outw)
	{
		int inindex = 0, outindex = 0;
		map2d(inh, inw, outh, outw, index, upad, lpad, &inindex, &outindex);

		gpuAtomicAdd(&ingrad[inindex], outgrad[outindex]);
	}
}

""")


if device is not None:
	mod = SourceModule("%s%s%s" % (
		mapTmpl, padTmpl.substitute(T=half_t, ext="FP16"), padTmpl.substitute(T=float_t, ext="")
	))


def reflectpad(data, pad, allocator=memPool):
	if data.ndim == 3:
		batchsize, maps, insize = data.shape
		lpad, rpad = pad

		assert insize >= max(lpad, rpad) + 1
		outsize = insize + lpad + rpad

		block = (warpSize, 1, 1)
		grid = (roundUpDiv(outsize, warpSize), maps, batchsize)

		outdata = GPUArray.empty((batchsize, maps, outsize), dtype=data.dtype, allocator=allocator)
		fn = mod.reflectpad1d if data.dtype == np.float32 else mod.reflectpad1dFP16

		fn(outdata, data, np.int32(insize), np.int32(lpad), np.int32(rpad), block=block, grid=grid)

	elif data.ndim == 4:
		batchsize, maps, inh, inw = data.shape
		upad, bpad, lpad, rpad = pad

		assert inh >= max(upad, bpad) + 1 and inw >= max(lpad, rpad) + 1
		outh, outw = inh + upad + bpad, inw + lpad + rpad

		block = (warpSize, 1, 1)
		grid = (roundUpDiv(outh * outw, warpSize), maps, batchsize)

		outdata = GPUArray.empty((batchsize, maps, outh, outw), dtype=data.dtype, allocator=allocator)
		fn = mod.reflectpad2d if data.dtype == np.float32 else mod.reflectpad2dFP16

		fn(
			outdata, data, np.int32(inh), np.int32(inw), np.int32(upad), np.int32(bpad), np.int32(lpad), np.int32(rpad),
			block=block, grid=grid
		)

	else:
		raise NotImplementedError(data.ndim)

	return outdata


def reflectpadBackward(grad, pad, allocator=memPool):
	if grad.ndim == 3:
		batchsize, maps, outsize = grad.shape
		lpad, rpad = pad

		block = (warpSize, 1, 1)
		grid = (roundUpDiv(outsize, warpSize), maps, batchsize)

		insize = outsize - lpad - rpad
		ingrad = GPUArray.zeros((batchsize, maps, insize), dtype=grad.dtype, allocator=allocator)
		fn = mod.reflectpad1dBackward if grad.dtype == np.float32 else mod.reflectpad1dBackwardFP16

		fn(ingrad, grad, np.int32(insize), np.int32(lpad), np.int32(rpad), block=block, grid=grid)

	elif grad.ndim == 4:
		batchsize, maps, outh, outw = grad.shape
		upad, bpad, lpad, rpad = pad

		inh, inw = outh - upad - bpad, outw - lpad - rpad

		block = (warpSize, 1, 1)
		grid = (roundUpDiv(outh * outw, warpSize), maps, batchsize)

		ingrad = GPUArray.zeros((batchsize, maps, inh, inw), dtype=grad.dtype, allocator=allocator)
		fn = mod.reflectpad2dBackward if grad.dtype == np.float32 else mod.reflectpad2dBackwardFP16

		fn(
			ingrad, grad, np.int32(inh), np.int32(inw), np.int32(upad), np.int32(bpad), np.int32(lpad), np.int32(rpad),
			block=block, grid=grid
		)

	else:
		raise NotImplementedError(grad.ndim)

	return ingrad


def unittest():
	for dtype, _ in dtypesSupported():
		reflectpad1dTest(dtype)
		reflectpad2dTest(dtype)


def reflectpad1dTest(dtype):
	batchsize, maps, insize = 4, 8, 48
	lpad, rpad = 2, 3

	hostData = np.random.randn(batchsize, maps, insize).astype(dtype)

	data = GPUArray.toGpu(hostData)
	outdata = reflectpad(data, pad=(lpad, rpad))

	hostOutData = outdata.get()
	outsize = hostOutData.shape[2]

	assert np.allclose(hostOutData[:, :, lpad:insize + lpad], hostData)
	assert np.allclose(hostOutData[:, :, :lpad][:, :, ::-1], hostData[:, :, 1:lpad+1])
	assert np.allclose(hostOutData[:, :, insize + lpad:][:, :, ::-1], hostData[:, :, insize - 1 - rpad:insize - 1])

	hostGrad = np.random.randn(batchsize, maps, outsize).astype(np.float32)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = reflectpadBackward(grad, pad=(lpad, rpad))

	hostInGrad = ingrad.get()

	assert np.allclose(
		hostInGrad[:, :, lpad + 1:insize - rpad - 1], hostGrad[:, :, 2 * lpad + 1:outsize - 2 * rpad - 1]
	)
	assert np.allclose(
		hostInGrad[:, :, 1:lpad + 1], hostGrad[:, :, :lpad][:, :, ::-1] + hostGrad[:, :, lpad + 1:2 * lpad + 1]
	)
	assert np.allclose(
		hostInGrad[:, :, insize - rpad - 1:insize - 1],
		hostGrad[:, :, outsize - rpad:][:, :, ::-1] + hostGrad[:, :, outsize - 2 * rpad - 1:outsize - rpad - 1]
	)


def reflectpad2dTest(dtype):
	batchsize, maps, inh, inw = 4, 8, 12, 15
	upad, bpad, lpad, rpad = 2, 3, 2, 3

	hostData = np.random.randn(batchsize, maps, inh, inw).astype(dtype)

	data = GPUArray.toGpu(hostData)
	outdata = reflectpad(data, pad=(upad, bpad, lpad, rpad))

	hostOutData = outdata.get()
	outh, outw = hostOutData.shape[2:]

	assert np.allclose(hostOutData[:, :, upad:inh + upad, lpad:inw + lpad], hostData)
	assert np.allclose(hostOutData[:, :, :upad, :lpad][:, :, ::-1, ::-1], hostData[:, :, 1:upad + 1, 1:lpad + 1])
	assert np.allclose(
		hostOutData[:, :, inh + upad:, inw + lpad:][:, :, ::-1, ::-1],
		hostData[:, :, inh - 1 - bpad:inh - 1, inw - 1 - rpad:inw - 1]
	)

	hostGrad = np.random.randn(batchsize, maps, outh, outw).astype(np.float32)

	grad = GPUArray.toGpu(hostGrad)
	ingrad = reflectpadBackward(grad, pad=(upad, bpad, lpad, rpad))

	hostInGrad = ingrad.get()

	assert np.allclose(
		hostInGrad[:, :, upad + 1:inh - bpad - 1, lpad + 1:inw - rpad - 1],
		hostGrad[:, :, 2 * upad + 1:outh - 2 * bpad - 1, 2 * lpad + 1:outw - 2 * rpad - 1]
	)
	assert np.allclose(
		hostInGrad[:, :, 1:upad + 1, 1:lpad + 1],
		hostGrad[:, :, :upad, :lpad][:, :, ::-1, ::-1] +
		hostGrad[:, :, upad + 1:2 * upad + 1, lpad + 1:2 * lpad + 1] +
		hostGrad[:, :, :upad, lpad + 1:2 * lpad + 1][:, :, ::-1, :] +
		hostGrad[:, :, upad + 1:2 * upad + 1, :lpad][:, :, :, ::-1]
	)
	assert np.allclose(
		hostInGrad[:, :, inh - bpad - 1:inh - 1, inw - rpad - 1:inw - 1],
		hostGrad[:, :, outh - bpad:, outw - rpad:][:, :, ::-1, ::-1] +
		hostGrad[:, :, outh - 2 * bpad - 1:outh - bpad - 1, outw - 2 * rpad - 1:outw - rpad - 1] +
		hostGrad[:, :, outh - bpad:, outw - 2 * rpad - 1:outw - rpad - 1][:, :, ::-1, :] +
		hostGrad[:, :, outh - 2 * bpad - 1:outh - bpad - 1, outw - rpad:][:, :, :, ::-1]
	)


if __name__ == "__main__":
	unittest()
