import itertools
from string import Template

import numpy as np

from PuzzleLib.Compiler.Codegen.Types import half_t, float_t
from PuzzleLib.Cuda.Utils import roundUpDiv


transformTmpl = Template("""

extern "C"
__global__ void transform2d$ext($T * __restrict__ dst, const $T * __restrict__ src, int dstOffset, int srcOffset,
								int dstStride0, int dstStride1, int srcStride0, int len0, int len1)
{
	for (int i = threadIdx.x + $NT * blockIdx.x; i < len0 * len1; i += $NT * blockDim.x)
	{
		int i0 = i / len1, i1 = i % len1;
		dst[dstOffset + i0 * dstStride0 + i1 * dstStride1] = src[srcOffset + i0 * srcStride0 + i1];
	}
}


extern "C"
__global__ void transform3d$ext($T * __restrict__ dst, const $T * __restrict__ src, int dstOffset, int srcOffset,
								int dstStride0, int dstStride1, int dstStride2, int srcStride0, int srcStride1,
								int len0, int len1, int len2)
{
	for (int i = threadIdx.x + $NT * blockIdx.x; i < len0 * len1 * len2; i += $NT * blockDim.x)
	{
		int i0 = i / (len1 * len2), i1 = (i / len2) % len1, i2 = i % len2;

		int outoffset = dstOffset + i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2;
		int inoffset = srcOffset + i0 * srcStride0 + i1 * srcStride1 + i2;

		dst[outoffset] = src[inoffset];
	}
}


extern "C"
__global__ void transform4d$ext($T * __restrict__ dst, const $T * __restrict__ src, int dstOffset, int srcOffset,
								int dstStride0, int dstStride1, int dstStride2, int dstStride3,
								int srcStride0, int srcStride1, int srcStride2, int len0, int len1, int len2, int len3)
{
	for (int i = threadIdx.x + $NT * blockIdx.x; i < len0 * len1 * len2 * len3; i += $NT * blockDim.x)
	{
		int i0 = i / (len1 * len2 * len3), i1 = (i / (len2 * len3)) % len1;
		int i2 = (i / len3) % len2, i3 = i % len3;

		int outoffset = dstOffset + i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2 + i3 * dstStride3;
		int inoffset = srcOffset + i0 * srcStride0 + i1 * srcStride1 + i2 * srcStride2 + i3;

		dst[outoffset] = src[inoffset];
	}
}


extern "C"
__global__ void transform5d$ext($T * __restrict__ dst, const $T * __restrict__ src, int dstOffset, int srcOffset,
								int dstStride0, int dstStride1, int dstStride2, int dstStride3, int dstStride4,
								int srcStride0, int srcStride1, int srcStride2, int srcStride3,
								int len0, int len1, int len2, int len3, int len4)
{
	for (int i = threadIdx.x + $NT * blockIdx.x; i < len0 * len1 * len2 * len3 * len4; i += $NT * blockDim.x)
	{
		int i0 = i / (len1 * len2 * len3 * len4), i1 = (i / (len2 * len3 * len4)) % len1;
		int i2 = (i / (len3 * len4)) % len2, i3 = (i / len4) % len3, i4 = i % len4;
		
		int offs = dstOffset + i0 * dstStride0 + i1 * dstStride1 + i2 * dstStride2 + i3 * dstStride3 + i4 * dstStride4;
		int inoffset = srcOffset + i0 * srcStride0 + i1 * srcStride1 + i2 * srcStride2 + i3 * srcStride3 + i4;

		dst[offs] = src[inoffset];
	}
}

""")


class MemoryModule:
	def __init__(self, backend):
		self.backend = backend
		self.GPUArray, self.NT = backend.GPUArray, backend.nthreads

		self.mod = backend.SourceModule("#include <cuda_fp16.h>\n\n%s%s" % (
			transformTmpl.substitute(NT=self.NT, T=half_t, ext="FP16"),
			transformTmpl.substitute(NT=self.NT, T=float_t, ext="")
		))


	def transform(self, tensor, shape, strides, out, inoffset=0, outoffset=0):
		assert tensor.dtype == np.float32 or tensor.dtype == np.float16
		assert tensor.ndim <= 5 and tensor.ndim == len(strides) and tensor.ndim == len(shape)

		ndim = tensor.ndim

		if ndim == 1:
			out.set(tensor)
			return out

		if ndim == 2:
			transform = self.mod.transform2d if tensor.dtype == np.float32 else self.mod.transform2dFP16
		elif ndim == 3:
			transform = self.mod.transform3d if tensor.dtype == np.float32 else self.mod.transform3dFP16
		elif ndim == 4:
			transform = self.mod.transform4d if tensor.dtype == np.float32 else self.mod.transform4dFP16
		elif ndim == 5:
			transform = self.mod.transform5d if tensor.dtype == np.float32 else self.mod.transform5dFP16
		else:
			assert False

		transform(
			out, tensor, np.int32(outoffset), np.int32(inoffset),
			*(np.int32(s // tensor.dtype.itemsize) for s in strides),
			*(np.int32(s // tensor.dtype.itemsize) for s in tensor.strides[:-1]),
			*(np.int32(dim) for dim in shape),
			block=(self.NT, 1, 1), grid=(roundUpDiv(tensor.size, self.NT), 1, 1)
		)

		return out


	def transpose(self, tensor, axes=None, out=None, allocator=None):
		assert axes is None or len(axes) == tensor.ndim

		axes = tuple(reversed(range(tensor.ndim))) if axes is None else axes
		shape = tuple(tensor.dimAt(axis) for axis in axes)

		if out is None:
			out = self.GPUArray.empty(shape, dtype=tensor.dtype, allocator=allocator)
		else:
			assert out.shape == shape

		outstrides = [0] * len(axes)
		for i, axis in enumerate(axes):
			outstrides[axis] = out.strideAt(i)

		return self.transform(tensor, tensor.shape, outstrides, out)


	def moveaxis(self, data, src, dst, out=None, allocator=None):
		if src < dst:
			axes = tuple(range(src)) + tuple(range(src + 1, dst + 1)) + (src, ) + tuple(range(dst + 1, data.ndim))
		else:
			axes = tuple(range(dst)) + (src, ) + tuple(range(dst, src)) + tuple(range(src + 1, data.ndim))

		return self.transpose(data, axes, out=out, allocator=allocator)


	def swapaxes(self, data, axis1, axis2, out=None, allocator=None):
		if axis1 == axis2:
			axes = tuple(range(data.ndim))

		else:
			axis1, axis2 = (axis1, axis2) if axis1 < axis2 else (axis2, axis1)
			axes = tuple(range(axis1)) + (axis2, ) + tuple(range(axis1 + 1, axis2)) + \
				   (axis1, ) + tuple(range(axis2 + 1, data.ndim))

		return self.transpose(data, axes, out=out, allocator=allocator)


	def depthConcat(self, tensors, out=None, allocator=None):
		assert all(tn.ndim == 4 and tn.dtype == tensors[0].dtype for tn in tensors)
		assert all(tn.dimAt(0) == tensors[0].dimAt(0) for tn in tensors)

		h, w, depth = 0, 0, 0
		for tn in tensors:
			depth += tn.dimAt(1)
			h, w = max(h, tn.dimAt(2)), max(w, tn.dimAt(3))

		if out is None:
			out = self.GPUArray.zeros(
				shape=(tensors[0].dimAt(0), depth, h, w), dtype=tensors[0].dtype, allocator=allocator
			)
		else:
			assert out.shape == (tensors[0].dimAt(0), depth, h, w)

		stride = 0
		for i, tn in enumerate(tensors):
			center = (h - tn.dimAt(2)) // 2 * out.strideAt(2) + (w - tn.dimAt(3)) // 2 * out.strideAt(3)

			self.transform(tn, tn.shape, out.strides, out=out, outoffset=stride + center // tn.dtype.itemsize)
			stride += out.strideAt(1) * tn.dimAt(1) // tn.dtype.itemsize

		return out


	def depthSplit(self, grad, tensors, allocator=None):
		assert all(tn.ndim == 4 and tn.dtype == tensors[0].dtype for tn in tensors)
		assert all(tn.dimAt(0) == tensors[0].dimAt(0) for tn in tensors)

		ingrads = [self.GPUArray.empty(shape=tn.shape, dtype=tn.dtype, allocator=allocator) for tn in tensors]

		stride = 0
		for i, gr in enumerate(ingrads):
			center = (grad.dimAt(2) - gr.dimAt(2)) // 2 * grad.strideAt(2) + (grad.dimAt(3) - gr.dimAt(3)) // 2 * \
					 grad.strideAt(3)

			self.transform(grad, gr.shape, gr.strides, gr, inoffset=stride + center // gr.dtype.itemsize)
			stride += grad.strideAt(1) * gr.dimAt(1) // gr.dtype.itemsize

		return ingrads


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		module = MemoryModule(Backend.getBackend(deviceIdx, initmode=2))

		for dtype, _ in module.backend.dtypesSupported():
			transposeTest(module.backend, module, dtype)
			moveAxisTest(module.backend, module, dtype)
			swapAxesTest(module.backend, module, dtype)
			depthConcatTest(module.backend, module, dtype)


def transposeTest(bnd, module, dtype):
	shapes = [(10, ), (10, 3), (10, 3, 5, 4, 2)]

	for shape in shapes:
		for axes in itertools.permutations(range(len(shape))):
			hostData = np.random.randn(*shape).astype(dtype)

			data = bnd.GPUArray.toGpu(hostData)
			outdata = module.transpose(data, axes=axes)

			hostOutData = np.transpose(hostData, axes=axes)
			assert np.allclose(hostOutData, outdata.get())


def moveAxisTest(bnd, module, dtype):
	shapes = [(10, ), (10, 3), (10, 3, 5, 4, 2)]

	for shape in shapes:
		for src, dst in itertools.product(range(len(shape)), range(len(shape))):
			hostData = np.random.randn(*shape).astype(dtype)

			data = bnd.GPUArray.toGpu(hostData)
			outdata = module.moveaxis(data, src=src, dst=dst)

			hostOutData = np.moveaxis(hostData, source=src, destination=dst)
			assert np.allclose(hostOutData, outdata.get())


def swapAxesTest(bnd, module, dtype):
	shapes = [(10, ), (10, 3), (10, 3, 5, 4, 2)]

	for shape in shapes:
		for axis1, axis2 in itertools.product(range(len(shape)), range(len(shape))):
			hostData = np.random.randn(*shape).astype(dtype)

			data = bnd.GPUArray.toGpu(hostData)
			outdata = module.swapaxes(data, axis1=axis1, axis2=axis2)

			hostOutData = np.swapaxes(hostData, axis1=axis1, axis2=axis2)
			assert np.allclose(hostOutData, outdata.get())


def depthConcatTest(bnd, module, dtype):
	hostData1 = np.random.randn(3, 4, 3, 3).astype(dtype)
	hostData2 = np.random.randn(3, 2, 6, 6).astype(dtype)
	hostData3 = np.random.randn(3, 5, 4, 4).astype(dtype)
	allHostData = [hostData1, hostData2, hostData3]

	allData = [bnd.GPUArray.toGpu(data) for data in allHostData]
	outdata = module.depthConcat(allData)

	depth, h, w = 0, 0, 0
	for data in allHostData:
		depth += data.shape[1]
		h, w = max(h, data.shape[2]), max(w, data.shape[3])

	hostOutData = np.zeros(shape=(allHostData[0].shape[0], depth, h, w), dtype=dtype)

	hostOutData[:, :4, 1:4, 1:4] = hostData1
	hostOutData[:, 4:6, :, :] = hostData2
	hostOutData[:, 6:, 1:5, 1:5] = hostData3

	assert np.allclose(hostOutData, outdata.get())

	hostGrad = np.random.randn(*hostOutData.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrads = module.depthSplit(grad, allData)

	hostInGrads = [
		hostGrad[:, :4, 1:4, 1:4],
		hostGrad[:, 4:6, :, :],
		hostGrad[:, 6:, 1:5, 1:5]
	]

	assert all(np.allclose(hostInGrad, ingrads[i].get()) for i, hostInGrad in enumerate(hostInGrads))


if __name__ == "__main__":
	unittest()
