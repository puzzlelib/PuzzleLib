from string import Template

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Utils import memoryPool as memPool, context, queue
from PuzzleLib.OpenCL.Kernels.Utils import roundUp, nthreads


poolTmpl = Template("""

__kernel void maxpool2d(__global float *outdata, __global const float *indata, __global int *mask, int inh, int inw,
						int outh, int outw, int maps, int hstride, int wstride, int hpad, int wpad, int fh, int fw,
						int size)
{
	for (int index = get_global_id(0); index < size; index += get_global_size(0))
	{
		const int pw = index % outw;
		const int ph = (index / outw) % outh;
		const int c = (index / outw / outh) % maps;
		const int n = index / outw / outh / maps;

		int hstart = ph * hstride - hpad;
		int wstart = pw * wstride - wpad;

		const int hend = min(hstart + fh, inh);
		const int wend = min(wstart + fw, inw);

		hstart = max(hstart, 0);
		wstart = max(wstart, 0);

		float maxval = $initVal;
		int maxidx = -1;

		__global const float *slice = indata + (n * maps + c) * inh * inw;

		for (int h = hstart; h < hend; ++h)
			for (int w = wstart; w < wend; ++w)
			{
				if (slice[h * inw + w] > maxval)
				{
					maxidx = h * inw + w;
					maxval = slice[maxidx];
				}
			}

		outdata[index] = maxval;
		mask[index] = maxidx;
	}
}

__kernel void maxunpool2d(__global float *outdata, __global const float *indata, __global const int *mask,
						  int inh, int inw, int outh, int outw, int maps, int size)
{
	for (int index = get_global_id(0); index < size; index += get_global_size(0))
	{
		int c = (index / inw / inh) % maps;
		int n = index / inw / inh / maps;

		__global float *slice = outdata + (n * maps + c) * outh * outw;
		int maxind = mask[index];

		slice[maxind] = indata[index];
	}
}

__kernel void maxpool2dBackward(__global float *ingrad, __global const float *outgrad, __global const int *mask,
								int inh, int inw, int outh, int outw, int maps, int hstride, int wstride,
								int hpad, int wpad, int fh, int fw, int size)
{
	for (int index = get_global_id(0); index < size; index += get_global_size(0))
	{
		const int w = index % inw;
		const int h = (index / inw) % inh;
		const int c = (index / inw / inh) % maps;
		const int n = index / inw / inh / maps;

		const int phstart = (h + hpad < fh) ? 0 : (h + hpad - fh) / hstride + 1;
		const int phend = min((h + hpad) / hstride + 1, outh);

		const int pwstart = (w + wpad < fw) ? 0 : (w + wpad - fw) / wstride + 1;
		const int pwend = min((w + wpad) / wstride + 1, outw);

		float grad = 0.0f;
		const int offset = (n * maps + c) * outh * outw;

		__global const float *slice = outgrad + offset;
		__global const int *maskSlice = mask + offset;

		for (int ph = phstart; ph < phend; ++ph)
			for (int pw = pwstart; pw < pwend; ++pw)
				if (maskSlice[ph * outw + pw] == h * inw + w)
					grad += slice[ph * outw + pw];

		ingrad[index] = grad;
	}
}

__kernel void maxunpool2dBackward(__global float *ingrad, __global const float *outgrad, __global const int *mask,
								  int inh, int inw, int outh, int outw, int maps, int size)
{
	for (int index = get_global_id(0); index < size; index += get_global_size(0))
	{
		int c = (index / inw / inh) % maps;
		int n = index / inw / inh / maps;

		__global const float *slice = outgrad + (n * maps + c) * outh * outw;
		int maxind = mask[index];

		ingrad[index] = slice[maxind];
	}
}

""")


if context:
	mod = Driver.Program(context, poolTmpl.substitute(initVal=str(np.finfo(np.float32).min))).build()


def maxpool2d(data, size, stride, pad):
	assert data.dtype == np.float32
	batchsize, maps, inh, inw = data.shape

	fh, fw = size
	hstride, wstride = stride
	hpad, wpad = pad

	outh = (inh - fh + 2 * hpad) // hstride + 1
	outw = (inw - fw + 2 * wpad) // wstride + 1

	outdata = Driver.empty(queue, (batchsize, maps, outh, outw), dtype=np.float32, allocator=memPool)
	mask = Driver.empty(queue, (batchsize, maps, outh, outw), dtype=np.int32, allocator=memPool)

	kernel = mod.maxpool2d

	size = int(np.prod(outdata.shape))

	block = (nthreads, 1, 1)
	grid = (roundUp(size, nthreads), 1, 1)

	kernel(queue, grid, block, outdata.data, data.data, mask.data, np.int32(inh), np.int32(inw), np.int32(outh),
		   np.int32(outw), np.int32(maps), np.int32(hstride), np.int32(wstride), np.int32(hpad), np.int32(wpad),
		   np.int32(fh), np.int32(fw), np.int32(size))

	return outdata, mask


def maxpool2dBackward(grad, origshape, mask, size, stride, pad):
	assert grad.dtype == np.float32 and mask.dtype == np.int32
	batchsize, maps, outh, outw = grad.shape

	fh, fw = size
	hstride, wstride = stride
	hpad, wpad = pad

	inh, inw = origshape[2], origshape[3]

	ingrad = Driver.empty(queue, (batchsize, maps, inh, inw), dtype=np.float32, allocator=memPool)

	kernel = mod.maxpool2dBackward

	size = int(np.prod(ingrad.shape))

	block = (nthreads, 1, 1)
	grid = (roundUp(size, nthreads), 1, 1)

	kernel(queue, grid, block, ingrad.data, grad.data, mask.data, np.int32(inh), np.int32(inw),
		   np.int32(outh), np.int32(outw), np.int32(maps), np.int32(hstride), np.int32(wstride),
		   np.int32(hpad), np.int32(wpad), np.int32(fh), np.int32(fw), np.int32(size))

	return ingrad


def maxunpool2d(data, origshape, mask):
	assert data.dtype == np.float32
	batchsize, maps, inh, inw = data.shape

	outh, outw = origshape[2], origshape[3]

	outdata = Driver.zeros(queue, (batchsize, maps, outh, outw), dtype=np.float32, allocator=memPool)

	kernel = mod.maxunpool2d

	size = int(np.prod(data.shape))

	block = (nthreads, 1, 1)
	grid = (roundUp(size, nthreads), 1, 1)

	kernel(queue, grid, block, outdata.data, data.data, mask.data, np.int32(inh), np.int32(inw),
		   np.int32(outh), np.int32(outw), np.int32(maps), np.int32(size))

	return outdata


def maxunpool2dBackward(grad, poolshape, mask):
	assert grad.dtype == np.float32 and mask.dtype == np.int32
	batchsize, maps, outh, outw = grad.shape

	inh, inw = poolshape[2], poolshape[3]

	ingrad = Driver.empty(queue, (batchsize, maps, inh, inw), dtype=np.float32, allocator=memPool)

	kernel = mod.maxunpool2dBackward

	size = int(np.prod(ingrad.shape))

	block = (nthreads, 1, 1)
	grid = (roundUp(size, nthreads), 1, 1)

	kernel(queue, grid, block, ingrad.data, grad.data, mask.data, np.int32(inh), np.int32(inw),
		   np.int32(outh), np.int32(outw), np.int32(maps), np.int32(size))

	return ingrad


def unittest():
	poolTest()
	unpoolTest()


def poolTest():
	batchsize, maps, h, w = 10, 4, 6, 6
	size, stride, pad = 2, 2, 1

	indata = Driver.to_device(queue, np.random.randn(batchsize, maps, h, w).astype(np.float32))

	pooldata, mask = maxpool2d(indata, [size, size], [stride, stride], [pad, pad])

	hostInData = np.zeros((batchsize, maps, h + 2 * pad, w + 2 * pad), dtype=np.float32)
	hostInData[:, :, pad:-pad, pad:-pad] = indata.get()

	hostPoolData = np.empty(pooldata.shape, dtype=np.float32)
	hostMask = np.empty(mask.shape, dtype=np.int32)

	for b in range(batchsize):
		for c in range(maps):
			for py in range(hostPoolData.shape[2]):
				for px in range(hostPoolData.shape[3]):
					maxval = -np.finfo(np.float32).max
					maxidx = -1

					iny = max(py * stride - pad, 0)
					inx = max(px * stride - pad, 0)

					for y in range(iny, min(h, py * stride - pad + size)):
						for x in range(inx, min(w, px * stride - pad + size)):
							val = hostInData[b, c, y + pad, x + pad]
							if val > maxval:
								maxval = val
								maxidx = y * h + x

					hostPoolData[b, c, py, px] = maxval
					hostMask[b, c, py, px] = maxidx

	assert np.allclose(hostPoolData, pooldata.get())
	assert (hostMask == mask.get()).all()

	grad = Driver.to_device(queue, np.random.randn(*pooldata.shape).astype(np.float32))
	ingrad = maxpool2dBackward(grad, indata.shape, mask, [size, size], [stride, stride], [pad, pad])

	hostGrad = grad.get()
	hostInGrad = np.empty(ingrad.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(h):
				for x in range(w):
					outy = 0 if y + pad < size else (y + pad - size) // stride + 1
					outx = 0 if x + pad < size else (x + pad - size) // stride + 1

					gr = 0.0

					for py in range(outy, min((y + pad) // stride + 1, hostPoolData.shape[2])):
						for px in range(outx, min((x + pad) // stride + 1, hostPoolData.shape[3])):
							if hostMask[b, c, py, px] == y * w + x:
								gr += hostGrad[b, c, py, px]

					hostInGrad[b, c, y, x] = gr

	assert np.allclose(hostInGrad, ingrad.get())


def unpoolTest():
	batchsize, maps, h, w = 10, 4, 6, 6
	size, stride, pad = 2, 2, 1

	indata = Driver.to_device(queue, np.random.randn(batchsize, maps, h, w).astype(np.float32))

	pooldata, mask = maxpool2d(indata, [size, size], [stride, stride], [pad, pad])
	unpooldata = maxunpool2d(pooldata, indata.shape, mask)

	hostPoolData = pooldata.get()
	hostMask = mask.get()

	hostUnpoolData = np.zeros(unpooldata.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(pooldata.shape[2]):
				for x in range(pooldata.shape[3]):
					maxidx = hostMask[b, c, y, x]
					hostUnpoolData[b, c].ravel()[maxidx] = hostPoolData[b, c, y, x]

	assert np.allclose(hostUnpoolData, unpooldata.get())

	grad = Driver.to_device(queue, np.random.randn(*unpooldata.shape).astype(np.float32))
	ingrad = maxunpool2dBackward(grad, pooldata.shape, mask)

	hostGrad = grad.get()
	hostInGrad = np.empty(ingrad.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(pooldata.shape[2]):
				for x in range(pooldata.shape[3]):
					maxidx = hostMask[b, c, y, x]
					hostInGrad[b, c, y, x] = hostGrad[b, c].ravel()[maxidx]

	assert np.allclose(hostInGrad, ingrad.get())


if __name__ == "__main__":
	unittest()
