from string import Template

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Utils import memoryPool as memPool, context, queue
from PuzzleLib.OpenCL.Kernels.Utils import warpSize, roundUp, atomicAddTmpl


upsample3dNearestTmpl = Template("""

#define W_BLOCK_SIZE $wBlockSize
#define H_BLOCK_SIZE $hBlockSize


__kernel void upsample3dNearest(__global float *outdata, __global const float *indata, int ind, int inh, int inw,
								int outd, int outh, int outw, int dscale, int hscale, int wscale)
{
	__local float shdata[H_BLOCK_SIZE][W_BLOCK_SIZE];

	int x = get_global_id(0);
	int y = get_global_id(1);
	int z = get_group_id(2);

	if (y >= inh || x >= inw) return;

	shdata[get_local_id(1)][get_local_id(0)] = indata[z * inh * inw + y * inw + x];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < dscale; i++)
		for (int j = 0; j < hscale; j++)
			for (int k = 0; k < wscale; k++)
			{
				int outidx = (z * dscale + i) * outh * outw + (y * hscale + j) * outw + (x * wscale + k);
				outdata[outidx] = shdata[get_local_id(1)][get_local_id(0)];
			}
}

__kernel void upsample3dNearestBackward(__global float *ingrad, __global const float *outgrad, int inh, int inw,
										int outh, int outw, int dscale, int hscale, int wscale, int insize)
{
	int idx = get_global_id(0);

	if (idx >= insize) return;

	int x = (idx % inw) * wscale;
	int y = ((idx % (inh * inw)) / inw) * hscale;
	int z = idx / (inh * inw) * dscale;

	float acc = 0.0f;

	for (int i = 0; i < dscale; i++)
		for (int j = 0; j < hscale; j++)
			for (int k = 0; k < wscale; k++)
				acc += outgrad[(z + i) * outh * outw + (y + j) * outw + x + k];

	ingrad[idx] = acc;
}

""")

upsample3dLinearTmpl = Template("""

$atomicAdd


__kernel void upsample3dLinear(__global float *outdata, __global const float *indata, int batchsize, int maps,
								int ind, int inh, int inw, int outd, int outh, int outw,
								float rd, float rh, float rw)
{
	int outx = get_global_id(0);
	int outy = get_global_id(1);
	int outz = get_group_id(2);

	if (outx >= outw || outy >= outh) return;

	float d1r = rd * outz;
	int d1 = d1r;
	int d1p = (d1 < ind - 1) ? 1 : 0;
	float dd1 = d1r - d1;
	float dd0 = 1.0f - dd1;

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
		int obstride = b * maps * outd * outh * outw;
		int ibstride = b * maps * ind * inh * inw;

		for (int c = 0; c < maps; c++)
		{
			int ocstride = c * outd * outh * outw;
			int icstride = c * ind * inh * inw;

			float val =
			dd0 * (dh0 * (dw0 * indata[ibstride + icstride + d1 * inh *inw + h1 * inw + w1] +
						  dw1 * indata[ibstride + icstride + d1 * inw *inw + h1 * inw + w1 + w1p]) +
				   dh1 * (dw0 * indata[ibstride + icstride + d1 * inh *inw + (h1 + h1p) * inw + w1] +
						  dw1 * indata[ibstride + icstride + d1 * inh *inw + (h1 + h1p) * inw + w1 + w1p])) +
			dd1 * (dh0 * (dw0 * indata[ibstride + icstride + (d1 + d1p) * inh * inw + h1 * inw + w1] +
						  dw1 * indata[ibstride + icstride + (d1 + d1p) * inh * inw + h1 * inw + w1 + w1p]) +
				   dh1 * (dw0 * indata[ibstride + icstride + (d1 + d1p) * inh * inw + (h1 + h1p) * inw + w1] +
						  dw1 * indata[ibstride + icstride + (d1 + d1p) * inh * inw + (h1 + h1p) * inw + w1 + w1p]));

			outdata[obstride + ocstride + outz * outh * outw + outy * outw + outx] = val;
		}
	}
}

__kernel void upsample3dLinearBackward(__global float *ingrad, __global const float *outgrad, int batchsize,
										int maps, int ind, int inh, int inw, int outd, int outh, int outw,
										float rd, float rh, float rw)
{
	int outx = get_global_id(0);
	int outy = get_global_id(1);
	int outz = get_group_id(2);

	if (outx >= outw || outy >= outh) return;

	float d1r = rd * outz;
	int d1 = d1r;
	int d1p = (d1 < ind - 1) ? 1 : 0;
	float dd1 = d1r - d1;
	float dd0 = 1.0f - dd1;

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
		int obstride = b * maps * outd * outh * outw;
		int ibstride = b * maps * ind * inh * inw;

		for (int c = 0; c < maps; c++)
		{
			int ocstride = c * outd * outh * outw;
			int icstride = c * ind * inh * inw;

			float val = outgrad[obstride + ocstride + outz * outh * outw + outy * outw + outx];

			atomicAddCAS(&ingrad[ibstride+icstride + d1 * inh*inw + h1 * inw + w1], dd0 * dh0 * dw0 * val);
			atomicAddCAS(&ingrad[ibstride+icstride + d1 * inh*inw + h1 * inw + w1+w1p], dd0 * dh0 * dw1 * val);
			atomicAddCAS(&ingrad[ibstride+icstride + d1 * inh*inw + (h1+h1p) * inw + w1], dd0 * dh1 * dw0 * val);
			atomicAddCAS(&ingrad[ibstride+icstride + d1 * inh*inw + (h1+h1p) * inw + w1+w1p], dd0 * dh1 * dw1 * val);

			atomicAddCAS(&ingrad[ibstride+icstride + (d1+d1p) * inh*inw + h1 * inw + w1], dd1 * dh0 * dw0 * val);
			atomicAddCAS(&ingrad[ibstride+icstride + (d1+d1p) * inh*inw + h1 * inw + w1+w1p], dd1 * dh0 * dw1 * val);
			atomicAddCAS(&ingrad[ibstride+icstride + (d1+d1p) * inh*inw + (h1+h1p) * inw + w1], dd1 * dh1 * dw0 * val);
			atomicAddCAS(&ingrad[ibstride+icstride + (d1+d1p) * inh*inw + (h1+h1p) * inw + w1+w1p],
						 dd1 * dh1 * dw1 * val);
		}
	}
}

""")


hblocksize = 4
wblocksize = warpSize


if context:
	nearestMod = Driver.Program(context, upsample3dNearestTmpl.substitute(hBlockSize=hblocksize,
																		  wBlockSize=wblocksize)).build()
	linearMod = Driver.Program(context, upsample3dLinearTmpl.substitute(atomicAdd=atomicAddTmpl)).build()


def upsample3d(data, scale, mode="nearest"):
	batchsize, maps, ind, inh, inw = data.shape

	if isinstance(scale, int):
		dscale, hscale, wscale = scale, scale, scale
	else:
		dscale, hscale, wscale = scale

	outd, outh, outw = dscale * ind, hscale * inh, wscale * inw
	outdata = Driver.empty(queue, (batchsize, maps, outd, outh, outw), dtype=data.dtype, allocator=memPool)

	if mode == "nearest":
		block = (wblocksize, hblocksize, 1)
		grid = (roundUp(inw, block[0]), roundUp(inh, block[1]), batchsize * maps * ind)

		kernel = nearestMod.upsample3dNearest
		kernel(queue, grid, block, outdata.data, data.data, np.int32(ind), np.int32(inh), np.int32(inw),
			   np.int32(outd), np.int32(outh), np.int32(outw), np.int32(dscale), np.int32(hscale), np.int32(wscale))

	elif mode == "linear":
		rd = (ind - 1) / (outd - 1)
		rh = (inh - 1) / (outh - 1)
		rw = (inw - 1) / (outw - 1)

		block = (warpSize // 4, warpSize // 4, 1)
		grid = (roundUp(outw, block[0]), roundUp(outh, block[1]), outd)

		kernel = linearMod.upsample3dLinear
		kernel(queue, grid, block, outdata.data, data.data, np.int32(batchsize), np.int32(maps),
			   np.int32(ind), np.int32(inh), np.int32(inw), np.int32(outd), np.int32(outh), np.int32(outw),
			   np.float32(rd), np.float32(rh), np.float32(rw))

	else:
		raise ValueError("Unsupported upsampling mode")

	return outdata


def upsample3dBackward(grad, scale, mode="nearest"):
	batchsize, maps, outd, outh, outw = grad.shape

	if isinstance(scale, int):
		dscale, hscale, wscale = scale, scale, scale
	else:
		dscale, hscale, wscale = scale

	ind, inh, inw = outd // dscale, outh // hscale, outw // wscale

	if mode == "nearest":
		ingrad = Driver.empty(queue, (batchsize, maps, ind, inh, inw), dtype=grad.dtype, allocator=memPool)

		blk = warpSize * 4
		block = (blk, 1, 1)

		grid = (roundUp(ingrad.size, blk), 1, 1)

		kernel = nearestMod.upsample3dNearestBackward
		kernel(queue, grid, block, ingrad.data, grad.data, np.int32(inh), np.int32(inw), np.int32(outh), np.int32(outw),
			   np.int32(dscale), np.int32(hscale), np.int32(wscale), np.int32(ingrad.size))

	elif mode == "linear":
		ingrad = Driver.zeros(queue, (batchsize, maps, ind, inh, inw), dtype=grad.dtype, allocator=memPool)

		block = (warpSize // 4, warpSize // 4, 1)
		grid = (roundUp(outw, block[0]), roundUp(outh, block[1]), outd)

		rd = (ind - 1) / (outd - 1)
		rh = (inh - 1) / (outh - 1)
		rw = (inw - 1) / (outw - 1)

		kernel = linearMod.upsample3dLinearBackward

		kernel(queue, grid, block, ingrad.data, grad.data, np.int32(batchsize), np.int32(maps),
			   np.int32(ind), np.int32(inh), np.int32(inw), np.int32(outd), np.int32(outh), np.int32(outw),
			   np.float32(rd), np.float32(rh), np.float32(rw))

	else:
		raise ValueError("Unrecognized sampling mode")

	return ingrad


def unittest():
	upsample3dNearestTest()
	upsample3dLinearTest()
	speedTest()


def upsample3dNearestTest():
	batchsize, maps, ind, inh, inw = 4, 2, 3, 5, 3
	scale = 2

	data = Driver.to_device(queue, np.random.randn(batchsize, maps, ind, inh, inw).astype(np.float32))

	outdata = upsample3d(data, scale, mode="nearest")

	hostData = data.get()
	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for z in range(ind):
				for y in range(inh):
					for x in range(inw):
						hostOutData[b, c, z * scale:(z+1) * scale, y * scale:(y+1) * scale, x * scale:(x+1) * scale] = \
							hostData[b, c, z, y, x]

	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))
	ingrad = upsample3dBackward(grad, scale)

	hostGrad = grad.get()
	hostInGrad = np.zeros(data.shape, dtype=np.float32)
	for b in range(batchsize):
		for c in range(maps):
			for z in range(ind):
				for y in range(inh):
					for x in range(inw):
						for dz in range(scale):
							for dy in range(scale):
								for dx in range(scale):
									hostInGrad[b, c, z, y, x] += \
										hostGrad[b, c, z * scale + dz, y * scale + dy, x * scale + dx]

	assert np.allclose(hostInGrad, ingrad.get())


def upsample3dLinearTest():
	batchsize, maps, ind, inh, inw = 1, 2, 2, 2, 2
	dscale, hscale, wscale = 2, 2, 1

	data = Driver.to_device(queue, np.random.randn(batchsize, maps, ind, inh, inw).astype(np.float32))

	outdata = upsample3d(data, (dscale, hscale, wscale), mode="linear")

	hostData = data.get()
	hostOutData = np.zeros(outdata.shape, dtype=np.float32)
	rd, rh, rw = (ind - 1) / (ind * dscale - 1), (inh - 1) / (inh * hscale - 1), (inw - 1) / (inw * wscale - 1)

	for b in range(batchsize):
		for c in range(maps):
			for z in range(ind * dscale):
				for y in range(inh * hscale):
					for x in range(inw * wscale):
						inz, iny, inx = int(rd * z), int(rh * y), int(rw * x)
						dz, dy, dx = 1.0 - (rd * z - inz), 1.0 - (rh * y - iny), 1.0 - (rw * x - inx)

						xi = 1 if x < inw * wscale - 1 else 0
						yi = 1 if y < inh * hscale - 1 else 0
						zi = 1 if z < ind * dscale - 1 else 0

						hostOutData[b, c, z, y, x] = \
						dz*(dy*(dx*hostData[b,c,inz,iny,inx] + (1-dx)*hostData[b,c,inz,iny,inx+xi]) +
							(1-dy)*(dx*hostData[b,c,inz,iny+yi,inx] + (1-dx)*hostData[b,c,inz,iny+yi,inx+xi])) + \
						(1-dz)*(dy*(dx*hostData[b,c,inz+zi,iny,inx] + (1-dx)*hostData[b,c,inz+zi,iny,inx+xi]) +
								(1-dy)*(dx*hostData[b,c,inz+zi,iny+yi,inx] + (1-dx)*hostData[b,c,inz+zi,iny+yi,inx+xi]))

	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(*outdata.shape).astype(np.float32))
	ingrad = upsample3dBackward(grad, (dscale, hscale, wscale), mode="linear")

	hostGrad = grad.get()
	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for z in range(ind * dscale):
				for y in range(inh * hscale):
					for x in range(inw * wscale):
						inz, iny, inx = int(rd * z), int(rh * y), int(rw * x)
						dz, dy, dx = 1.0 - (rd * z - inz), 1.0 - (rh * y - iny), 1.0 - (rw * x - inx)

						xi = 1 if x < inw * wscale - 1 else 0
						yi = 1 if y < inh * hscale - 1 else 0
						zi = 1 if z < ind * dscale - 1 else 0

						val = hostGrad[b, c, z, y, x]

						hostInGrad[b, c, inz, iny, inx] += dz * dy * dx * val
						hostInGrad[b, c, inz, iny, inx + xi] += dz * dy * (1 - dx) * val
						hostInGrad[b, c, inz, iny + yi, inx] += dz * (1 - dy) * dx * val
						hostInGrad[b, c, inz, iny + yi, inx + xi] += dz * (1 - dy) * (1 - dx) * val

						hostInGrad[b, c, inz + zi, iny, inx] += (1 - dz) * dy * dx * val
						hostInGrad[b, c, inz + zi, iny, inx + xi] += (1 - dz) * dy * (1 - dx) * val
						hostInGrad[b, c, inz + zi, iny + yi, inx] += (1 - dz) * (1 - dy) * dx * val
						hostInGrad[b, c, inz + zi, iny + yi, inx + xi] += (1 - dz) * (1 - dy) * (1 - dx) * val

	assert np.allclose(hostInGrad, ingrad.get())


def speedTest():
	batchsize, maps, ind, inh, inw = 32, 16, 4, 32, 32
	scale = 2

	data = Driver.to_device(queue, np.random.randn(batchsize, maps, ind, inh, inw).astype(np.float32))

	from PuzzleLib.OpenCL.Benchmarks.Utils import timeKernel

	timeKernel(upsample3d, args=(data, scale, "nearest"), logname="nearest mode")
	timeKernel(upsample3d, args=(data, scale, "linear"), logname="linear mode")


if __name__ == "__main__":
	unittest()
