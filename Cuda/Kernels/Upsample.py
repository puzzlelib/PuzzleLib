import itertools
from string import Template

import numpy as np
from PuzzleLib.Cuda.Utils import roundUpDiv


upsampleNearestTmpl = Template("""

extern "C"
__global__ void upsample2dNearest(float *outdata, const float *indata, int inh, int inw, int outh, int outw,
								  int hscale, int wscale)
{
	__shared__ float shdata[$hBlockSize][$wBlockSize];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z;

	if (y >= inh || x >= inw) return;

	shdata[threadIdx.y][threadIdx.x] = indata[z * inh * inw + y * inw + x];
	__syncthreads();

	for (int i = 0; i < hscale; i++)
		for (int j = 0; j < wscale; j++)
		{
			int outidx = z * outh * outw + (y * hscale + i) * outw + (x * wscale + j);
			outdata[outidx] = shdata[threadIdx.y][threadIdx.x];
		}
}


extern "C"
__global__ void upsample2dNearestBackward(float *ingrad, const float *outgrad, int inw, int outw,
										  int hscale, int wscale, int insize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= insize) return;

	int x = (idx % inw) * wscale;
	int y = (idx / inw) * hscale;

	float acc = 0.0f;

	for (int i = 0; i < wscale; i++)
		for (int j = 0; j < hscale; j++)
			acc += outgrad[(y + j) * outw + x + i];

	ingrad[idx] = acc;
}


extern "C"
__global__ void upsample3dNearest(float *outdata, const float *indata, int ind, int inh, int inw,
								  int outd, int outh, int outw, int dscale, int hscale, int wscale)
{
	__shared__ float shdata[$hBlockSize][$wBlockSize];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z;

	if (y >= inh || x >= inw) return;

	shdata[threadIdx.y][threadIdx.x] = indata[z * inh * inw + y * inw + x];
	__syncthreads();

	for (int i = 0; i < dscale; i++)
		for (int j = 0; j < hscale; j++)
			for (int k = 0; k < wscale; k++)
			{
				int outidx = (z * dscale + i) * outh * outw + (y * hscale + j) * outw + (x * wscale + k);
				outdata[outidx] = shdata[threadIdx.y][threadIdx.x];
			}
}


extern "C"
__global__ void upsample3dNearestBackward(float *ingrad, const float *outgrad, int inh, int inw, int outh, int outw,
										  int dscale, int hscale, int wscale, int insize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
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


upsampleLinearTmpl = Template("""

extern "C"
__global__ void upsample2dLinear(float *outdata, const float *indata, int batchsize, int maps, int inh, int inw,
								 int outh, int outw, float rh, float rw)
{
	int outx = blockIdx.x * blockDim.x + threadIdx.x;
	int outy = blockIdx.y * blockDim.y + threadIdx.y;

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


extern "C"
__global__ void upsample2dLinearBackward(float *ingrad, const float *outgrad, int batchsize, int maps,
										 int inh, int inw, int outh, int outw, float rh, float rw)
{
	int outx = blockIdx.x * blockDim.x + threadIdx.x;
	int outy = blockIdx.y * blockDim.y + threadIdx.y;

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

			atomicAdd(&ingrad[ibstride + icstride + h1 * inw + w1], dh0 * dw0 * val);
			atomicAdd(&ingrad[ibstride + icstride + h1 * inw + w1 + w1p], dh0 * dw1 * val);
			atomicAdd(&ingrad[ibstride + icstride + (h1 + h1p) * inw + w1], dh1 * dw0 * val);
			atomicAdd(&ingrad[ibstride + icstride + (h1 + h1p) * inw + w1 + w1p], dh1 * dw1 * val);
		}
	}
}


extern "C"
__global__ void upsample3dLinear(float *outdata, const float *indata, int batchsize, int maps,
								 int ind, int inh, int inw, int outd, int outh, int outw, float rd, float rh, float rw)
{
	int outx = blockIdx.x * blockDim.x + threadIdx.x;
	int outy = blockIdx.y * blockDim.y + threadIdx.y;
	int outz = blockIdx.z;

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


extern "C"
__global__ void upsample3dLinearBackward(float *ingrad, const float *outgrad, int batchsize, int maps,
										 int ind, int inh, int inw, int outd, int outh, int outw,
										 float rd, float rh, float rw)
{
	int outx = blockIdx.x * blockDim.x + threadIdx.x;
	int outy = blockIdx.y * blockDim.y + threadIdx.y;
	int outz = blockIdx.z;

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

			atomicAdd(&ingrad[ibstride+icstride + d1 * inh*inw + h1 * inw + w1], dd0 * dh0 * dw0 * val);
			atomicAdd(&ingrad[ibstride+icstride + d1 * inh*inw + h1 * inw + w1+w1p], dd0 * dh0 * dw1 * val);
			atomicAdd(&ingrad[ibstride+icstride + d1 * inh*inw + (h1+h1p) * inw + w1], dd0 * dh1 * dw0 * val);
			atomicAdd(&ingrad[ibstride+icstride + d1 * inh*inw + (h1+h1p) * inw + w1+w1p], dd0 * dh1 * dw1 * val);

			atomicAdd(&ingrad[ibstride+icstride + (d1+d1p) * inh*inw + h1 * inw + w1], dd1 * dh0 * dw0 * val);
			atomicAdd(&ingrad[ibstride+icstride + (d1+d1p) * inh*inw + h1 * inw + w1+w1p], dd1 * dh0 * dw1 * val);
			atomicAdd(&ingrad[ibstride+icstride + (d1+d1p) * inh*inw + (h1+h1p) * inw + w1], dd1 * dh1 * dw0 * val);
			atomicAdd(&ingrad[ibstride+icstride + (d1+d1p) * inh*inw + (h1+h1p) * inw + w1+w1p], dd1 * dh1 * dw1 * val);
		}
	}
}

""")


class UpsampleModule:
	def __init__(self, backend):
		self.backend, self.GPUArray = backend, backend.GPUArray
		self.warpSize, self.nthreads = backend.warpSize, backend.nthreads

		self.hblocksize, self.wblocksize = 4, self.warpSize

		self.nearestMod = backend.SourceModule(upsampleNearestTmpl.substitute(
			hBlockSize=self.hblocksize, wBlockSize=self.wblocksize
		))
		self.linearMod = backend.SourceModule(upsampleLinearTmpl.substitute())


	def upsample2d(self, data, scale, mode="nearest", allocator=None):
		batchsize, maps, inh, inw = data.shape
		hscale, wscale = (scale, scale) if isinstance(scale, int) else scale

		outh, outw = hscale * inh, wscale * inw
		outdata = self.GPUArray.empty((batchsize, maps, outh, outw), dtype=data.dtype, allocator=allocator)

		if mode == "nearest":
			block = (self.wblocksize, self.hblocksize, 1)
			grid = (roundUpDiv(inw, block[0]), roundUpDiv(inh, block[1]), batchsize * maps)

			self.nearestMod.upsample2dNearest(
				outdata, data, np.int32(inh), np.int32(inw), np.int32(outh), np.int32(outw),
				np.int32(hscale), np.int32(wscale), block=block, grid=grid
			)

		elif mode == "linear":
			block = (self.warpSize, self.nthreads // self.warpSize, 1)
			grid = (roundUpDiv(outw, block[0]), roundUpDiv(outh, block[1]), 1)

			rh, rw = (inh - 1) / (outh - 1), (inw - 1) / (outw - 1)

			self.linearMod.upsample2dLinear(
				outdata, data, np.int32(batchsize), np.int32(maps), np.int32(inh), np.int32(inw),
				np.int32(outh), np.int32(outw), np.float32(rh), np.float32(rw), block=block, grid=grid
			)

		else:
			raise NotImplementedError(mode)

		return outdata


	def upsample2dBackward(self, grad, scale, mode="nearest", allocator=None):
		batchsize, maps, outh, outw = grad.shape
		hscale, wscale = (scale, scale) if isinstance(scale, int) else scale

		inh, inw = outh // hscale, outw // wscale

		if mode == "nearest":
			ingrad = self.GPUArray.empty((batchsize, maps, inh, inw), dtype=grad.dtype, allocator=allocator)

			blk = self.warpSize * 8
			block = (blk, 1, 1)
			grid = (roundUpDiv(ingrad.size, blk), 1, 1)

			self.nearestMod.upsample2dNearestBackward(
				ingrad, grad, np.int32(inw), np.int32(outw), np.int32(hscale), np.int32(wscale), np.int32(ingrad.size),
				block=block, grid=grid
			)

		elif mode == "linear":
			ingrad = self.GPUArray.zeros((batchsize, maps, inh, inw), dtype=grad.dtype, allocator=allocator)

			block = (self.warpSize, self.nthreads // self.warpSize, 1)
			grid = (roundUpDiv(outw, block[0]), roundUpDiv(outh, block[1]), 1)

			rh, rw = (inh - 1) / (outh - 1), (inw - 1) / (outw - 1)

			self.linearMod.upsample2dLinearBackward(
				ingrad, grad, np.int32(batchsize), np.int32(maps), np.int32(inh), np.int32(inw),
				np.int32(outh), np.int32(outw), np.float32(rh), np.float32(rw), block=block, grid=grid
			)

		else:
			raise NotImplementedError(mode)

		return ingrad


	def upsample3d(self, data, scale, mode="nearest", allocator=None):
		batchsize, maps, ind, inh, inw = data.shape
		dscale, hscale, wscale = (scale, scale, scale) if isinstance(scale, int) else scale

		outd, outh, outw = dscale * ind, hscale * inh, wscale * inw
		outdata = self.GPUArray.empty((batchsize, maps, outd, outh, outw), dtype=data.dtype, allocator=allocator)

		if mode == "nearest":
			block = (self.wblocksize, self.hblocksize, 1)
			grid = (roundUpDiv(inw, block[0]), roundUpDiv(inh, block[1]), batchsize * maps * ind)

			self.nearestMod.upsample3dNearest(
				outdata, data, np.int32(ind), np.int32(inh), np.int32(inw),
				np.int32(outd), np.int32(outh), np.int32(outw), np.int32(dscale), np.int32(hscale), np.int32(wscale),
				block=block, grid=grid
			)

		elif mode == "linear":
			block = (self.warpSize, self.nthreads // self.warpSize, 1)
			grid = (roundUpDiv(outw, block[0]), roundUpDiv(outh, block[1]), outd)

			rd, rh, rw = (ind - 1) / (outd - 1), (inh - 1) / (outh - 1), (inw - 1) / (outw - 1)

			self.linearMod.upsample3dLinear(
				outdata, data, np.int32(batchsize), np.int32(maps), np.int32(ind), np.int32(inh), np.int32(inw),
				np.int32(outd), np.int32(outh), np.int32(outw), np.float32(rd), np.float32(rh), np.float32(rw),
				block=block, grid=grid
			)

		else:
			raise NotImplementedError(mode)

		return outdata


	def upsample3dBackward(self, grad, scale, mode="nearest", allocator=None):
		batchsize, maps, outd, outh, outw = grad.shape
		dscale, hscale, wscale = (scale, scale, scale) if isinstance(scale, int) else scale

		ind, inh, inw = outd // dscale, outh // hscale, outw // wscale

		if mode == "nearest":
			ingrad = self.GPUArray.empty((batchsize, maps, ind, inh, inw), dtype=grad.dtype, allocator=allocator)

			blk = self.warpSize * 8
			block = (blk, 1, 1)

			grid = (roundUpDiv(ingrad.size, blk), 1, 1)

			self.nearestMod.upsample3dNearestBackward(
				ingrad, grad, np.int32(inh), np.int32(inw), np.int32(outh), np.int32(outw),
				np.int32(dscale), np.int32(hscale), np.int32(wscale), np.int32(ingrad.size), block=block, grid=grid
			)

		elif mode == "linear":
			ingrad = self.GPUArray.zeros((batchsize, maps, ind, inh, inw), dtype=grad.dtype, allocator=allocator)

			block = (self.warpSize, self.nthreads // self.warpSize, 1)
			grid = (roundUpDiv(outw, block[0]), roundUpDiv(outh, block[1]), outd)

			rd, rh, rw = (ind - 1) / (outd - 1), (inh - 1) / (outh - 1), (inw - 1) / (outw - 1)

			self.linearMod.upsample3dLinearBackward(
				ingrad, grad, np.int32(batchsize), np.int32(maps), np.int32(ind), np.int32(inh), np.int32(inw),
				np.int32(outd), np.int32(outh), np.int32(outw), np.float32(rd), np.float32(rh), np.float32(rw),
				block=block, grid=grid
			)

		else:
			raise NotImplementedError(mode)

		return ingrad


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		module = UpsampleModule(Backend.getBackend(deviceIdx))

		upsample2dNearestTest(module)
		upsample2dLinearTest(module)
		upsample2dSpeedTest(module)

		upsample3dNearestTest(module)
		upsample3dLinearTest(module)
		upsample3dSpeedTest(module)


def upsample2dNearestTest(module):
	batchsize, maps, inh, inw = 1, 2, 16, 15
	scale = 2

	hostData = np.random.uniform(low=-1.0, high=1.0, size=(batchsize, maps, inh, inw)).astype(np.float32)

	data = module.GPUArray.toGpu(hostData)
	outdata = module.upsample2d(data, scale, mode="nearest")

	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for b, c, y, x in itertools.product(range(batchsize), range(maps), range(inh), range(inw)):
		hostOutData[b, c, y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = hostData[b, c, y, x]

	assert np.allclose(hostOutData, outdata.get())

	hostGrad = np.random.randn(*outdata.shape).astype(np.float32)

	grad = module.GPUArray.toGpu(hostGrad)
	ingrad = module.upsample2dBackward(grad, scale)

	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b, c, y, x, dy, dx in itertools.product(
		range(batchsize), range(maps), range(inh), range(inw), range(scale), range(scale)
	):
		hostInGrad[b, c, y, x] += hostGrad[b, c, y * scale + dy, x * scale + dx]

	assert np.allclose(hostInGrad, ingrad.get(), atol=1e-5)


def upsample2dLinearTest(module):
	batchsize, maps, inh, inw = 3, 2, 4, 4
	hscale, wscale = 2, 3

	hostData = np.random.randn(batchsize, maps, inh, inw).astype(np.float32)

	data = module.GPUArray.toGpu(hostData)
	outdata = module.upsample2d(data, (hscale, wscale), mode="linear")

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)
	rh, rw = (inh - 1) / (inh * hscale - 1), (inw - 1) / (inw * wscale - 1)

	for b, c, y, x, in itertools.product(range(batchsize), range(maps), range(inh * hscale), range(inw * wscale)):
		iny, inx = int(rh * y), int(rw * x)
		dy, dx = 1.0 - (rh * y - iny), 1.0 - (rw * x - inx)

		yi, xi = 1 if y < inh * hscale - 1 else 0, 1 if x < inw * wscale - 1 else 0

		hostOutData[b, c, y, x] = dy * (dx * hostData[b, c, iny, inx] + (1 - dx) * hostData[b, c, iny, inx + xi]) + \
								  (1 - dy) * (dx * hostData[b, c, iny + yi, inx] +
								  (1 - dx) * hostData[b, c, iny + yi, inx + xi])

	hostGrad = np.random.randn(*outdata.shape).astype(np.float32)

	grad = module.GPUArray.toGpu(hostGrad)
	ingrad = module.upsample2dBackward(grad, (hscale, wscale), mode="linear")

	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b, c, y, x in itertools.product(range(batchsize), range(maps), range(inh * hscale), range(inw * wscale)):
		iny, inx = int(rh * y), int(rw * x)
		dy, dx = 1.0 - (rh * y - iny), 1.0 - (rw * x - inx)

		yi, xi = 1 if y < inh * hscale - 1 else 0, 1 if x < inw * wscale - 1 else 0
		val = hostGrad[b, c, y, x]

		hostInGrad[b, c, iny, inx] += dy * dx * val
		hostInGrad[b, c, iny, inx + xi] += dy * (1 - dx) * val
		hostInGrad[b, c, iny + yi, inx] += (1 - dy) * dx * val
		hostInGrad[b, c, iny + yi, inx + xi] += (1 - dy) * (1 - dx) * val

	assert np.allclose(hostInGrad, ingrad.get(), atol=1e-5)


def upsample3dNearestTest(module):
	batchsize, maps, ind, inh, inw = 4, 2, 3, 5, 3
	scale = 2

	hostData = np.random.randn(batchsize, maps, ind, inh, inw).astype(np.float32)

	data = module.GPUArray.toGpu(hostData)
	outdata = module.upsample3d(data, scale, mode="nearest")

	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for b, c, z, y, x in itertools.product(range(batchsize), range(maps), range(ind), range(inh), range(inw)):
		hostOutData[b, c, z * scale:(z + 1) * scale, y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = \
			hostData[b, c, z, y, x]

	assert np.allclose(hostOutData, outdata.get())

	hostGrad = np.random.randn(*outdata.shape).astype(np.float32)

	grad = module.GPUArray.toGpu(hostGrad)
	ingrad = module.upsample3dBackward(grad, scale)

	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b, c, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(maps), range(ind), range(inh), range(inw), range(scale), range(scale), range(scale)
	):
		hostInGrad[b, c, z, y, x] += hostGrad[b, c, z * scale + dz, y * scale + dy, x * scale + dx]

	assert np.allclose(hostInGrad, ingrad.get())


def upsample3dLinearTest(module):
	batchsize, maps, ind, inh, inw = 1, 2, 2, 2, 2
	dscale, hscale, wscale = 2, 2, 1

	hostData = np.random.randn(batchsize, maps, ind, inh, inw).astype(np.float32)

	data = module.GPUArray.toGpu(hostData)
	outdata = module.upsample3d(data, (dscale, hscale, wscale), mode="linear")

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)
	rd, rh, rw = (ind - 1) / (ind * dscale - 1), (inh - 1) / (inh * hscale - 1), (inw - 1) / (inw * wscale - 1)

	for b, c, z, y, x in itertools.product(
		range(batchsize), range(maps), range(ind * dscale), range(inh * hscale), range(inw * wscale)
	):
		inz, iny, inx = int(rd * z), int(rh * y), int(rw * x)
		dz, dy, dx = 1.0 - (rd * z - inz), 1.0 - (rh * y - iny), 1.0 - (rw * x - inx)

		zi = 1 if z < ind * dscale - 1 else 0
		yi = 1 if y < inh * hscale - 1 else 0
		xi = 1 if x < inw * wscale - 1 else 0

		hostOutData[b, c, z, y, x] = dz * (dy * (
			dx * hostData[b, c, inz, iny, inx] + (1 - dx) * hostData[b, c, inz, iny, inx + xi]
		) + (1 - dy) * (
			dx * hostData[b, c, inz, iny + yi, inx] + (1 - dx) * hostData[b, c, inz, iny + yi, inx + xi]
		)) + (1 - dz) * (dy * (
			dx * hostData[b, c, inz+zi, iny, inx] + (1 - dx) * hostData[b, c, inz + zi, iny, inx + xi]
		) + (1 - dy) * (
			dx * hostData[b, c, inz + zi, iny + yi, inx] + (1 - dx) * hostData[b, c, inz + zi, iny + yi, inx + xi]
		))

	assert np.allclose(hostOutData, outdata.get())

	hostGrad = np.random.randn(*outdata.shape).astype(np.float32)

	grad = module.GPUArray.toGpu(hostGrad)
	ingrad = module.upsample3dBackward(grad, (dscale, hscale, wscale), mode="linear")

	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b, c, z, y, x in itertools.product(
			range(batchsize), range(maps), range(ind * dscale), range(inh * hscale), range(inw * wscale)
	):
		inz, iny, inx = int(rd * z), int(rh * y), int(rw * x)
		dz, dy, dx = 1.0 - (rd * z - inz), 1.0 - (rh * y - iny), 1.0 - (rw * x - inx)

		zi = 1 if z < ind * dscale - 1 else 0
		yi = 1 if y < inh * hscale - 1 else 0
		xi = 1 if x < inw * wscale - 1 else 0

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


def upsample2dSpeedTest(module):
	batchsize, maps, inh, inw = 32, 16, 32, 32
	scale = 2

	data = module.GPUArray.toGpu(np.random.randn(batchsize, maps, inh, inw).astype(np.float32))

	bnd = module.backend
	bnd.timeKernel(module.upsample2d, args=(data, scale, "nearest", bnd.memoryPool), logname="nearest 2d mode")
	bnd.timeKernel(module.upsample2d, args=(data, scale, "linear", bnd.memoryPool), logname="linear 2d mode")


def upsample3dSpeedTest(module):
	batchsize, maps, ind, inh, inw = 32, 16, 4, 32, 32
	scale = 2

	data = module.GPUArray.toGpu(np.random.randn(batchsize, maps, ind, inh, inw).astype(np.float32))

	bnd = module.backend
	bnd.timeKernel(module.upsample3d, args=(data, scale, "nearest", bnd.memoryPool), logname="nearest 3d mode")
	bnd.timeKernel(module.upsample3d, args=(data, scale, "linear", bnd.memoryPool), logname="linear 3d mode")


if __name__ == "__main__":
	unittest()
