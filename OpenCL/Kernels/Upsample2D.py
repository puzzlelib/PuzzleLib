from string import Template

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Utils import memoryPool as memPool, context, queue
from PuzzleLib.OpenCL.Kernels.Utils import warpSize, roundUp, atomicAddTmpl


upsample2dNearestTmpl = Template("""

#define W_BLOCK_SIZE $wBlockSize
#define H_BLOCK_SIZE $hBlockSize


__kernel void upsample2dNearest(__global float *outdata, __global const float *indata, int inh, int inw,
								int outh, int outw, int hscale, int wscale)
{
	__local float shdata[H_BLOCK_SIZE][W_BLOCK_SIZE];

	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_group_id(2);

	if (y >= inh || x >= inw) return;

	shdata[get_local_id(1)][get_local_id(0)] = indata[z * inh * inw + y * inw + x];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < hscale; i++)
		for (int j = 0; j < wscale; j++)
		{
			int outidx = z * outh * outw + (y * hscale + i) * outw + (x * wscale + j);
			outdata[outidx] = shdata[get_local_id(1)][get_local_id(0)];
		}
}

__kernel void upsample2dNearestBackward(__global float *ingrad, __global const float *outgrad, int inw, int outw,
										int hscale, int wscale, int insize)
{
	int idx = get_global_id(0);

	if (idx >= insize) return;

	int x = (idx % inw) * wscale;
	int y = (idx / inw) * hscale;

	float acc = 0.0f;

	for (int i = 0; i < wscale; i++)
		for (int j = 0; j < hscale; j++)
			acc += outgrad[(y + j) * outw + x + i];

	ingrad[idx] = acc;
}

""")


upsample2dLinearTmpl = Template("""

$atomicAdd


__kernel void upsample2dLinear(__global float *outdata, __global const float *indata, int batchsize, int maps,
								int inh, int inw, int outh, int outw, float rh, float rw)
{
	int outx = get_global_id(0);
	int outy = get_global_id(1);

	if (outx >= outw || outy >= outh) return;

	float h1r = rh * outy;
	int h1 = h1r;
	int h1p = (h1 < inh - 1) ? 1 : 0;
	float dh1 = h1r - h1;
	float dh0 = 1.0f - dh1;

	float w1r = rw * outx;
	int w1 = w1r;
	int w1p = (w1 < inw - 1) ? 1 : 0;
	float dw1 = w1r - w1;
	float dw0 = 1.0f - dw1;

	for (int b = 0; b < batchsize ; b++)
	{
		int obstride = b * maps * outh * outw;
		int ibstride = b * maps * inh * inw;

		for (int c = 0; c < maps; c++)
		{
			int ocstride = c * outh * outw;
			int icstride = c * inh * inw;

			float val = dh0 * (dw0 * indata[ibstride + icstride + h1 * inw + w1] +
							   dw1 * indata[ibstride + icstride + h1 * inw + w1 + w1p]) +
						dh1 * (dw0 * indata[ibstride + icstride + (h1 + h1p) * inw + w1] +
							   dw1 * indata[ibstride + icstride + (h1 + h1p) * inw + w1 + w1p]);

			outdata[obstride + ocstride + outy * outw + outx] = val;
		}
	}
}

__kernel void upsample2dLinearBackward(__global float *ingrad, __global const float *outgrad, int batchsize, int maps,
										 int inh, int inw, int outh, int outw, float rh, float rw)
{
	int outx = get_global_id(0);
	int outy = get_global_id(1);

	if (outx >= outw || outy >= outh) return;

	float h1r = rh * outy;
	int h1 = h1r;
	int h1p = (h1 < inh - 1) ? 1 : 0;
	float dh1 = h1r - h1;
	float dh0 = 1.0f - dh1;

	float w1r = rw * outx;
	int w1 = w1r;
	int w1p = (w1 < inw - 1) ? 1 : 0;
	float dw1 = w1r - w1;
	float dw0 = 1.0f - dw1;

	for (int b = 0; b < batchsize; b++)
	{
		int obstride = b * maps * outh * outw;
		int ibstride = b * maps * inh * inw;

		for (int c = 0; c < maps; c++)
		{
			int ocstride = c * outh * outw;
			int icstride = c * inh * inw;

			float val = outgrad[obstride + ocstride + outy * outw + outx];

			atomicAddCAS(&ingrad[ibstride + icstride + h1 * inw + w1], dh0 * dw0 * val);
			atomicAddCAS(&ingrad[ibstride + icstride + h1 * inw + w1 + w1p], dh0 * dw1 * val);
			atomicAddCAS(&ingrad[ibstride + icstride + (h1 + h1p) * inw + w1], dh1 * dw0 * val);
			atomicAddCAS(&ingrad[ibstride + icstride + (h1 + h1p) * inw + w1 + w1p], dh1 * dw1 * val);
		}
	}
}

""")


hblocksize = 4
wblocksize = warpSize


if context:
	nearestMod = Driver.Program(context, upsample2dNearestTmpl.substitute(hBlockSize=hblocksize,
																		  wBlockSize=wblocksize)).build()
	linearMod = Driver.Program(context, upsample2dLinearTmpl.substitute(atomicAdd=atomicAddTmpl)).build()


def upsample2d(data, scale, mode="nearest"):
	batchsize, maps, inh, inw = data.shape

	if isinstance(scale, int):
		hscale, wscale = scale, scale
	else:
		hscale, wscale = scale

	outh, outw = hscale * inh, wscale * inw
	outdata = Driver.empty(queue, (batchsize, maps, outh, outw), dtype=data.dtype, allocator=memPool)

	if mode == "nearest":
		block = (wblocksize, hblocksize, 1)
		grid = (roundUp(inw, block[0]), roundUp(inh, block[1]), batchsize * maps)

		kernel = nearestMod.upsample2dNearest
		kernel(queue, grid, block, outdata.data, data.data, np.int32(inh), np.int32(inw),
			   np.int32(outh), np.int32(outw), np.int32(hscale), np.int32(wscale))

	elif mode == "linear":
		rh = (inh - 1) / (outh - 1)
		rw = (inw - 1) / (outw - 1)

		block = (warpSize // 4, warpSize // 4, 1)
		grid = (roundUp(outw, block[0]), roundUp(outh, block[1]), 1)

		kernel = linearMod.upsample2dLinear
		kernel(queue, grid, block, outdata.data, data.data, np.int32(batchsize), np.int32(maps),
			   np.int32(inh), np.int32(inw), np.int32(outh), np.int32(outw), np.float32(rh), np.float32(rw))

	else:
		raise ValueError("Unsupported upsampling mode")

	return outdata


def upsample2dBackward(grad, scale, mode="nearest"):
	batchsize, maps, outh, outw = grad.shape

	if isinstance(scale, int):
		hscale, wscale = scale, scale
	else:
		hscale, wscale = scale

	inh, inw = outh // hscale, outw // wscale

	if mode == "nearest":
		ingrad = Driver.empty(queue, (batchsize, maps, inh, inw), dtype=grad.dtype, allocator=memPool)

		blk = warpSize * 4
		block = (blk, 1, 1)
		grid = (roundUp(ingrad.size, blk), 1, 1)

		kernel = nearestMod.upsample2dNearestBackward
		kernel(queue, grid, block, ingrad.data, grad.data, np.int32(inw), np.int32(outw),
			   np.int32(hscale), np.int32(wscale), np.int32(ingrad.size))

	elif mode == "linear":
		ingrad = Driver.zeros(queue, (batchsize, maps, inh, inw), dtype=grad.dtype, allocator=memPool)

		block = (warpSize // 4, warpSize // 4, 1)
		grid = (roundUp(outw, block[0]), roundUp(outh, block[1]), 1)

		rh = (inh - 1) / (outh - 1)
		rw = (inw - 1) / (outw - 1)

		kernel = linearMod.upsample2dLinearBackward
		kernel(queue, grid, block, ingrad.data, grad.data, np.int32(batchsize), np.int32(maps),
			   np.int32(inh), np.int32(inw), np.int32(outh), np.int32(outw), np.float32(rh), np.float32(rw))

	else:
		raise ValueError("Unrecognized sampling mode")

	return ingrad


def unittest():
	upsample2dNearestTest()
	upsample2dLinearTest()
	speedTest()


def upsample2dNearestTest():
	batchsize, maps, inh, inw = 1, 2, 16, 15
	scale = 2

	shape = (batchsize, maps, inh, inw)

	data = Driver.to_device(queue, np.random.uniform(low=-1.0, high=1.0, size=shape).astype(np.float32))
	outdata = upsample2d(data, scale, mode="nearest")

	hostData = data.get()
	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(inh):
				for x in range(inw):
					hostOutData[b, c, y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = hostData[b, c, y, x]

	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))
	ingrad = upsample2dBackward(grad, scale)

	hostGrad = grad.get()
	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(inh):
				for x in range(inw):
					for dy in range(scale):
						for dx in range(scale):
							hostInGrad[b, c, y, x] += hostGrad[b, c, y * scale + dy, x * scale + dx]

	assert np.allclose(hostInGrad, ingrad.get())


def upsample2dLinearTest():
	batchsize, maps, inh, inw = 3, 2, 4, 4
	hscale, wscale = 2, 3

	data = Driver.to_device(queue, np.random.randn(batchsize, maps, inh, inw).astype(np.float32))

	outdata = upsample2d(data, (hscale, wscale), mode="linear")

	hostData = data.get()
	hostOutData = np.zeros(outdata.shape, dtype=np.float32)
	rh, rw = (inh - 1) / (inh * hscale - 1), (inw - 1) / (inw * wscale - 1)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(inh * hscale):
				for x in range(inw * wscale):
					iny, inx = int(rh * y), int(rw * x)
					dy, dx = 1.0 - (rh * y - iny), 1.0 - (rw * x - inx)

					xi = 1 if x < inw * wscale - 1 else 0
					yi = 1 if y < inh * hscale - 1 else 0

					hostOutData[b, c, y, x] = \
					dy * (dx * hostData[b, c, iny, inx] + (1 - dx) * hostData[b, c, iny, inx + xi]) + \
					(1 - dy) * (dx * hostData[b, c, iny + yi, inx] + (1 - dx) * hostData[b, c, iny + yi, inx + xi])

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))
	ingrad = upsample2dBackward(grad, (hscale, wscale), mode="linear")

	hostGrad = grad.get()
	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(inh * hscale):
				for x in range(inw * wscale):
					iny, inx = int(rh * y), int(rw * x)
					dy, dx = 1.0 - (rh * y - iny), 1.0 - (rw * x - inx)

					xi = 1 if x < inw * wscale - 1 else 0
					yi = 1 if y < inh * hscale - 1 else 0

					val = hostGrad[b, c, y, x]

					hostInGrad[b, c, iny, inx] += dy * dx * val
					hostInGrad[b, c, iny, inx + xi] += dy * (1 - dx) * val
					hostInGrad[b, c, iny + yi, inx] += (1 - dy) * dx * val
					hostInGrad[b, c, iny + yi, inx + xi] += (1 - dy) * (1 - dx) * val

	assert np.allclose(hostInGrad, ingrad.get())


def speedTest():
	batchsize, maps, inh, inw = 32, 16, 32, 32
	scale = 2

	data = Driver.to_device(queue, np.random.randn(batchsize, maps, inh, inw).astype(np.float32))

	from PuzzleLib.OpenCL.Benchmarks.Utils import timeKernel

	timeKernel(upsample2d, args=(data, scale, "nearest"), logname="nearest mode")
	timeKernel(upsample2d, args=(data, scale, "linear"), logname="linear mode")


if __name__ == "__main__":
	unittest()
