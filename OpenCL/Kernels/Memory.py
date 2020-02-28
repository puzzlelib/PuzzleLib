import itertools
from string import Template

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Kernels.Utils import warpSize, roundUp
from PuzzleLib.OpenCL.Utils import context, queue, memoryPool as memPool, copy


transformTmpl = Template("""

#define WARP_SIZE $warpSize


__kernel void transform(__global const $dtype * __restrict src, __global $dtype * __restrict dst,
						int srcOffset, int srcStride0, int srcStride1, int srcStride2, int srcStride3,
						int srcLen0, int srcLen1, int srcLen2, int srcLen3, int srcLen4, int srcSize,
						int dstOffset, int dstStride0, int dstStride1, int dstStride2, int dstStride3, int dstStride4,
						int dstSize, int dims)
{
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int gidz = get_global_id(2);

	switch (dims)
	{
		case 1:
			for (int idx = gidx; idx < srcLen0; idx += (WARP_SIZE * 4) * 256)
			{
				if (idx < dstSize && idx < srcSize)
					dst[idx + dstOffset] = src[idx + srcOffset];
			}
			break;

		case 2:
			for (int yidx = gidy; yidx < srcLen0; yidx += 256)
				for (int xidx = gidx; xidx < srcLen1; xidx += WARP_SIZE * 4)
				{
					int sindex = srcStride0 * yidx + xidx;
					int dindex = dstStride0 * yidx + dstStride1 * xidx;

					if (dindex < dstSize && sindex < srcSize)
						dst[dindex + dstOffset] = src[sindex + srcOffset];
				}
			break;

		case 3:
			for (int zidx = gidz; zidx < srcLen0; zidx += 16)
				for (int yidx = gidy; yidx < srcLen1; yidx += 64)
					for (int xidx = gidx; xidx < srcLen2; xidx += WARP_SIZE)
					{
						int sindex = srcStride0 * zidx + srcStride1 * yidx + xidx;
						int dindex = dstStride0 * zidx + dstStride1 * yidx + dstStride2 * xidx;

						if (dindex < dstSize && sindex < srcSize)
							dst[dindex + dstOffset] = src[sindex + srcOffset];
					}
			break;

		case 4:
			for (int zidx = gidz; zidx < srcLen1; zidx += 16)
				for (int yidx = gidy; yidx < srcLen2; yidx += 64)
					for (int xidx = gidx; xidx < srcLen3; xidx += WARP_SIZE)
					{
						int stmp = srcStride1 * zidx + srcStride2 * yidx + xidx;
						int dtmp = dstStride1 * zidx + dstStride2 * yidx + dstStride3 * xidx;

						#pragma unroll
						for (int idx = 0; idx < srcLen0; idx++)
						{
							int sindex = srcStride0 * idx + stmp; 
							int dindex = dstStride0 * idx + dtmp;

							if (dindex < dstSize && sindex < srcSize)
								dst[dindex + dstOffset] = src[sindex + srcOffset];
						}
					}
			break;

		case 5:
			for (int zidx = gidz; zidx < srcLen2; zidx += 16)
				for (int yidx = gidy; yidx < srcLen3; yidx += 64)
					for (int xidx = gidx; xidx < srcLen4; xidx += WARP_SIZE)
					{
						int stmp = srcStride2 * zidx + srcStride3 * yidx + xidx;
						int dtmp = dstStride2 * zidx + dstStride3 * yidx + dstStride4 * xidx;

						#pragma unroll
						for (int idx = 0; idx < srcLen0; idx++)
						{
							int stmp2 = srcStride0 * idx + stmp;
							int dtmp2 = dstStride0 * idx + dtmp;

							#pragma unroll
							for (int jdx = 0; jdx < srcLen1; jdx++)
							{
								int sindex = srcStride1 * jdx + stmp2;
								int dindex = dstStride1 * jdx + dtmp2;

								if (dindex < dstSize && sindex < srcSize)
									dst[dindex + dstOffset] = src[sindex + srcOffset];
							}
						}
					}
			break;

		default:
			break;
	}
}

""")


if context:
	mod = Driver.Program(context, transformTmpl.substitute(dtype="float", warpSize=warpSize)).build()


def transformTensor(tensor, strides, inoffset=0, outoffset=0, shape=None, out=None):
	assert tensor.dtype == np.float32 and tensor.ndim <= 5
	assert tensor.ndim == len(strides)

	if shape is None:
		shape = tensor.shape

	size = np.prod(shape)
	ndim = len(shape)

	if out is None:
		out = Driver.empty(queue, shape, dtype=tensor.dtype, allocator=memPool)

	instrides = tuple(s // tensor.dtype.itemsize for s in tensor.strides)
	outstrides = tuple(s // tensor.dtype.itemsize for s in strides)

	sh0, sh1, sh2, sh3, sh4 = 0, 0, 0, 0, 0
	ss0, ss1, ss2, ss3 = 0, 0, 0, 0

	ds0, ds1, ds2, ds3, ds4 = 0, 0, 0, 0, 0

	if ndim == 1:
		block = (warpSize * 4, 1, 1)
		grid = (min(roundUp(size, block[0]), block[0] * 256), 1, 1)

		sh0 = shape[0]
		ds0 = outstrides[0]

	elif ndim == 2:
		block = (warpSize * 4, 1, 1)
		grid = (block[0], min(shape[0], 256), 1)

		sh0, sh1 = shape[:2]
		ss0 = instrides[0]

		ds0, ds1 = outstrides[:2]

	elif ndim == 3:
		block = (warpSize, 1, 1)
		grid = (block[0], min(shape[1], 64), min(shape[0], 16))

		sh0, sh1, sh2 = tensor.shape[:3]
		ss0, ss1 = instrides[:2]

		ds0, ds1, ds2 = outstrides[:3]

	elif ndim == 4:
		block = (warpSize, 1, 1)
		grid = (block[0], min(shape[2], 64), min(shape[1], 16))

		sh0, sh1, sh2, sh3 = shape[:4]
		ss0, ss1, ss2 = instrides[:3]

		ds0, ds1, ds2, ds3 = outstrides[:4]

	elif ndim == 5:
		block = (warpSize, 1, 1)
		grid = (block[0], min(shape[3], 64), min(shape[2], 16))

		sh0, sh1, sh2, sh3, sh4 = shape
		ss0, ss1, ss2, ss3 = instrides[:4]

		ds0, ds1, ds2, ds3, ds4 = outstrides

	else:
		raise NotImplementedError()

	inoffset //= tensor.dtype.itemsize
	outoffset //= tensor.dtype.itemsize

	mod.transform(queue, grid, block, tensor.base_data, out.base_data,
				  np.int32(tensor.offset + inoffset), np.int32(ss0), np.int32(ss1), np.int32(ss2), np.int32(ss3),
				  np.int32(sh0), np.int32(sh1), np.int32(sh2), np.int32(sh3), np.int32(sh4), np.int32(tensor.size),
				  np.int32(out.offset + outoffset), np.int32(ds0), np.int32(ds1), np.int32(ds2), np.int32(ds3),
				  np.int32(ds4), np.int32(out.size), np.int32(ndim))
	return out


def moveaxis(tensor, src, dst, out=None):
	assert tensor.dtype == np.float32
	assert src != dst

	if src < dst:
		shape = tensor.shape[:src] + tensor.shape[src+1:dst+1] + (tensor.shape[src], ) + tensor.shape[dst+1:]
	else:
		shape = tensor.shape[:dst] + (tensor.shape[src], ) + tensor.shape[dst:src] + tensor.shape[src+1:]

	if out is None:
		out = Driver.empty(queue, shape, dtype=np.float32, allocator=memPool)
	else:
		assert out.shape == shape

	if src < dst:
		outstrides = out.strides[:src] + (out.strides[dst], ) + out.strides[src:dst] + out.strides[dst + 1:]
	else:
		outstrides = out.strides[:dst] + out.strides[dst+1:src+1] + (out.strides[dst], ) + out.strides[src+1:]

	transformTensor(tensor, outstrides, out=out)
	return out


def swapaxes(tensor, axis1, axis2, out=None):
	assert tensor.dtype == np.float32

	if axis1 == axis2:
		if out is None:
			return copy(None, tensor)
		else:
			assert out.shape == tensor.shape
			copy(out, tensor)
			return out

	if axis1 > axis2:
		axis1, axis2 = axis2, axis1

	shape = tensor.shape[:axis1] + (tensor.shape[axis2], ) + tensor.shape[axis1+1:axis2] + \
			(tensor.shape[axis1], ) + tensor.shape[axis2+1:]

	if out is None:
		out = Driver.empty(queue, shape, dtype=np.float32, allocator=memPool)
	else:
		assert out.shape == shape

	outstrides = out.strides[:axis1] + (out.strides[axis2], ) + out.strides[axis1 + 1:axis2] + \
				 (out.strides[axis1], ) + out.strides[axis2 + 1:]

	transformTensor(tensor, outstrides, out=out)
	return out


def transpose(tensor, axes=None, out=None):
	assert tensor.dtype == np.float32
	assert axes is None or len(set(axes)) == tensor.ndim

	if axes is None:
		axes = tuple(reversed(range(tensor.ndim)))

	shape = tuple(tensor.shape[axis] for axis in axes)

	if out is None:
		out = Driver.empty(queue, shape, dtype=np.float32, allocator=memPool)
	else:
		assert out.shape == shape

	outstrides = [0] * len(axes)
	for i, axis in enumerate(axes):
		outstrides[axis] = out.strides[i]

	transformTensor(tensor, outstrides, out=out)
	return out


def depthConcat(tensors, out=None):
	assert all(tn.ndim == 4 and tn.dtype == tensors[0].dtype for tn in tensors)
	assert all(tn.shape[0] == tensors[0].shape[0] for tn in tensors)

	h, w, depth = 0, 0, 0
	for tn in tensors:
		depth += tn.shape[1]
		h, w = max(h, tn.shape[2]), max(w, tn.shape[3])

	if out is None:
		out = Driver.zeros(queue, shape=(tensors[0].shape[0], depth, h, w), dtype=np.float32, allocator=memPool)
	else:
		assert out.shape == (tensors[0].shape[0], depth, h, w)

	stride = 0
	for i, tn in enumerate(tensors):
		center = (h - tn.shape[2]) // 2 * out.strides[2] + (w - tn.shape[3]) // 2 * out.strides[3]

		transformTensor(tn, out.strides, outoffset=stride + center, out=out)
		stride += out.strides[1] * tn.shape[1]

	return out


def depthSplit(grad, tensors):
	assert all(tn.ndim == 4 and tn.dtype == tensors[0].dtype for tn in tensors)
	assert all(tn.shape[0] == tensors[0].shape[0] for tn in tensors)

	ingrads = [Driver.empty(queue, shape=tn.shape, dtype=np.float32, allocator=memPool) for tn in tensors]

	stride = 0
	for i, gr in enumerate(ingrads):
		center = (grad.shape[2] - gr.shape[2])//2 * grad.strides[2] + (grad.shape[3] - gr.shape[3])//2 * grad.strides[3]

		transformTensor(grad, gr.strides, inoffset=stride + center, shape=gr.shape, out=gr)
		stride += grad.strides[1] * gr.shape[1]

	return ingrads


def unittest():
	transformTensorTest()
	moveAxisTest()
	swapAxesTest()
	transposeTest()
	depthConcatTest()


def transformTensorTest():
	tensor = Driver.to_device(queue, np.random.randn(10, 10, 10, 10).astype(np.float32))
	outTensor = transformTensor(tensor, tensor.strides)

	assert np.allclose(tensor.get(), outTensor.get())


def moveAxisTest():
	shape = (10, 3, 5, 4, 2)

	for src in range(len(shape)):
		for dst in range(len(shape)):
			if src == dst:
				continue

			tensor = Driver.to_device(queue, np.random.randn(*shape).astype(np.float32))
			outTensor = moveaxis(tensor, src=src, dst=dst)

			hostOut = np.moveaxis(tensor.get(), source=src, destination=dst)
			assert np.allclose(outTensor.get(), hostOut)


def swapAxesTest():
	shape = (10, 3, 5, 4, 2)

	for axis1 in range(len(shape)):
		for axis2 in range(axis1+1, len(shape)):
			tensor = Driver.to_device(queue, np.random.randn(*shape).astype(np.float32))
			outTensor = swapaxes(tensor, axis1=axis1, axis2=axis2)

			hostOut = np.swapaxes(tensor.get(), axis1=axis1, axis2=axis2)
			assert np.allclose(outTensor.get(), hostOut)


def transposeTest():
	shape = (10, 3, 5, 4, 2)

	for axes in itertools.permutations((0, 1, 2, 3, 4)):
		tensor = Driver.to_device(queue, np.random.randn(*shape).astype(np.float32))
		outTensor = transpose(tensor, axes=axes)

		hostOut = np.transpose(tensor.get(), axes=axes)
		assert np.allclose(outTensor.get(), hostOut)


def depthConcatTest():
	data1 = Driver.to_device(queue, np.random.randn(3, 4, 3, 3).astype(np.float32))
	data2 = Driver.to_device(queue, np.random.randn(3, 2, 6, 6).astype(np.float32))
	data3 = Driver.to_device(queue, np.random.randn(3, 5, 4, 4).astype(np.float32))

	alldata = [data1, data2, data3]
	outdata = depthConcat(alldata)

	depth, h, w = 0, 0, 0
	for data in alldata:
		depth += data.shape[1]
		h, w = max(h, data.shape[2]), max(w, data.shape[3])

	hostOutData = np.zeros(shape=(data1.shape[0], depth, h, w), dtype=np.float32)

	hostOutData[:, :4, 1:4, 1:4] = data1.get()
	hostOutData[:, 4:6, :, :] = data2.get()
	hostOutData[:, 6:, 1:5, 1:5] = data3.get()

	assert np.allclose(hostOutData, outdata.get())

	grad = Driver.to_device(queue, np.random.randn(*hostOutData.shape).astype(np.float32))
	ingrads = depthSplit(grad, alldata)

	hostInGrads = [np.empty(data.shape, dtype=np.float32) for data in alldata]

	hostInGrads[0] = grad.get()[:, :4, 1:4, 1:4]
	hostInGrads[1] = grad.get()[:, 4:6, :, :]
	hostInGrads[2] = grad.get()[:, 6:, 1:5, 1:5]

	assert all(np.allclose(hostInGrad, ingrads[i].get()) for i, hostInGrad in enumerate(hostInGrads))


if __name__ == "__main__":
	unittest()
