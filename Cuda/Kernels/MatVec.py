from string import Template
import numpy as np

from PuzzleLib.Compiler.Codegen.Types import half_t, float_t
from PuzzleLib.Cuda.Utils import prod, roundUpDiv


minMaxTmpl = Template("""

extern "C"
__global__ void minMaxOnRow$ext(int *idx, const $T *mat, int w)
{
	float curMax = $initVal;
	int curIdx = -1;

	for (int i = threadIdx.x; i < w; i += $warpSize)
	{
		float val = mat[blockIdx.x * w + i];

		if (val $cmpOp curMax)
			curMax = val, curIdx = i;
	}

	for (int mask = $warpSize / 2; mask > 0; mask /= 2)
	{
		float mx = __shfl_xor_sync((unsigned)-1, curMax, mask, $warpSize);
		int idx = __shfl_xor_sync((unsigned)-1, curIdx, mask, $warpSize);

		if (mx $cmpOp curMax)
			curMax = mx, curIdx = idx;
	}

	idx[blockIdx.x] = curIdx;
}


extern "C"
__global__ void minMaxOnCol$ext(int *idx, const $T *mat, int w, int h)
{
	float curMax = $initVal;
	int curIdx = -1;

	int gid = threadIdx.x + blockIdx.x * $NT;
	if (gid >= w) return;

	for (int i = 0; i < h; i += 1)
	{
		float val = mat[blockIdx.z * h * w + i * w + gid];

		if (val $cmpOp curMax)
			curMax = val, curIdx = i;
	}

	idx[blockIdx.z * w + gid] = curIdx;
}

""")


sumTmpl = Template("""

extern "C"
__global__ void sumOnRow$ext($T *out, const $T *mat, int w, float alpha, float beta)
{
	float acc = 0.0f;

	for (int i = threadIdx.x; i < w; i += $warpSize)
		acc += (float)mat[blockIdx.x * w + i];

	for (int mask = $warpSize / 2; mask > 0; mask /= 2)
		acc += __shfl_xor_sync((unsigned)-1, acc, mask, $warpSize);

	out[blockIdx.x] = ($T)(beta * (float)out[blockIdx.x] + alpha * acc);
}


extern "C"
__global__ void sumOnCol$ext($T *out, const $T *mat, int w, int h, float alpha, float beta)
{
	float acc = 0.0f;

	int gid = threadIdx.x + blockIdx.x * $NT;
	if (gid >= w) return;

	for (int i = 0; i < h; i += 1)
		acc += (float)mat[blockIdx.z * h * w + i * w + gid];

	out[blockIdx.z * w + gid] = ($T)(beta * (float)out[blockIdx.z * w + gid] + alpha * acc);
}

""")


vecMulTmpl = Template("""

extern "C"
__global__ void vecMulOnRow$ext($T *out, const $T *mat, const $T *vec, int w, int h, float alpha, float beta)
{
	float acc = 0.0f;

	for (int i = threadIdx.x; i < w; i += $warpSize)
		acc += (float)mat[blockIdx.z * h * w + blockIdx.x * w + i] * (float)vec[blockIdx.z * w + i];

	for (int mask = $warpSize / 2; mask > 0; mask /= 2)
		acc += __shfl_xor_sync((unsigned)-1, acc, mask, $warpSize);

	out[blockIdx.z * h + blockIdx.x] = ($T)(beta * (float)out[blockIdx.z * h + blockIdx.x] + alpha * acc);
}


extern "C"
__global__ void vecMulOnCol$ext($T *out, const $T *mat, const $T *vec, int w, int h, float alpha, float beta)
{
	float acc = 0.0f;

	int gid = threadIdx.x + blockIdx.x * $NT;
	if (gid >= w) return;

	for (int i = 0; i < h; i += 1)
		acc += (float)mat[blockIdx.z * h * w + i * w + gid] * (float)vec[blockIdx.z * h + i];

	out[blockIdx.z * w + gid] = ($T)(beta * (float)out[blockIdx.z * w + gid] + alpha * acc);
}

""")


vecMatTmpl = Template("""

extern "C"
__global__ void opRowVecToMat$ext($T *out, const $T *vec, const $T *mat, int n, int m)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int tidz = blockIdx.z;

	int offset = tidz * m * n + tidy * m + tidx;

	if (tidy < n && tidx < m)
		out[offset] = (float)mat[offset] $op (float)vec[tidz * m + tidx];
}


extern "C"
__global__ void opColVecToMat$ext($T *out, const $T *vec, const $T *mat, int n, int m)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int tidz = blockIdx.z;

	int offset = tidz * m * n + tidy * m + tidx;

	if (tidy < n && tidx < m)
		out[offset] = (float)mat[offset] $op (float)vec[tidz * n + tidy];
}


extern "C"
__global__ void opRowOneVecToMat$ext($T *out, const $T *vec, const $T *mat, int n, int m, int p)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int tidz = blockIdx.z;

	int offset = tidz * m * n + tidy * m + tidx;

	if (tidy < n && tidx < m)
		out[offset] = (float)mat[offset] $op (float)vec[tidz * m + tidx % p];
}

""")


class MatModule:
	def __init__(self, backend):
		self.backend = backend

		self.GPUArray, self.warpSize = backend.GPUArray, backend.warpSize
		self.NT = 256

		self.maxmod = backend.SourceModule(
			"#include <cuda_fp16.h>\n\n%s%s" % (
				minMaxTmpl.substitute(
					warpSize=self.warpSize, NT=self.NT, initVal=np.finfo(np.float32).min, cmpOp=">",
					T=half_t, ext="FP16"
				),
				minMaxTmpl.substitute(
					warpSize=self.warpSize, NT=self.NT, initVal=np.finfo(np.float32).min, cmpOp=">",
					T=float_t, ext=""
				)
			)
		)

		self.minmod = backend.SourceModule(
			"#include <cuda_fp16.h>\n\n%s%s" % (
				minMaxTmpl.substitute(
					warpSize=self.warpSize, NT=self.NT, initVal=np.finfo(np.float32).max, cmpOp="<",
					T=half_t, ext="FP16"
				),
				minMaxTmpl.substitute(
					warpSize=self.warpSize, NT=self.NT, initVal=np.finfo(np.float32).max, cmpOp="<",
					T=float_t, ext=""
				)
			)
		)

		self.summod = backend.SourceModule(
			"#include <cuda_fp16.h>\n\n%s%s" % (
				sumTmpl.substitute(warpSize=self.warpSize, NT=self.NT, T=half_t, ext="FP16"),
				sumTmpl.substitute(warpSize=self.warpSize, NT=self.NT, T=float_t, ext="")
			)
		)

		self.mulmod = backend.SourceModule(
			"#include <cuda_fp16.h>\n\n%s%s" % (
				vecMulTmpl.substitute(warpSize=self.warpSize, NT=self.NT, T=half_t, ext="FP16"),
				vecMulTmpl.substitute(warpSize=self.warpSize, NT=self.NT, T=float_t, ext="")
			)
		)

		self.addmod = backend.SourceModule(
			"#include <cuda_fp16.h>\n\n%s%s" % (
				vecMatTmpl.substitute(op="+", T=half_t, ext="FP16"),
				vecMatTmpl.substitute(op="+", T=float_t, ext="")
			)
		)

		self.addblock = (self.warpSize, backend.nthreads // self.warpSize, 1)


	def argminmax(self, tensor, axis, mode, allocator=None):
		assert tensor.dtype == np.float32 or tensor.dtype == np.float16
		assert 0 <= axis < tensor.ndim

		mod = {
			"max": self.maxmod,
			"min": self.minmod
		}[mode]

		if axis == tensor.ndim - 1:
			block = (self.warpSize, 1, 1)
			grid = (prod(tensor.shape[:-1]), 1, 1)

			idx = self.GPUArray.empty(tensor.shape[:-1], dtype=np.int32, allocator=allocator)
			fn = mod.minMaxOnRow if tensor.dtype == np.float32 else mod.minMaxOnRowFP16

			fn(idx, tensor, np.int32(tensor.dimAt(-1)), block=block, grid=grid)

		else:
			z, width = prod(tensor.shape[:axis]), prod(tensor.shape[axis + 1:])

			block = (self.NT, 1, 1)
			grid = (roundUpDiv(width, block[0]), 1, z)

			idx = self.GPUArray.empty(
				tensor.shape[:axis] + tensor.shape[axis + 1:], dtype=np.int32, allocator=allocator
			)
			fn = mod.minMaxOnCol if tensor.dtype == np.float32 else mod.minMaxOnColFP16

			fn(idx, tensor, np.int32(width), np.int32(tensor.dimAt(axis)), block=block, grid=grid)

		return idx


	def argmax(self, tensor, axis=0, allocator=None):
		return self.argminmax(tensor, axis, "max", allocator)


	def argmin(self, tensor, axis=0, allocator=None):
		return self.argminmax(tensor, axis, "min", allocator)


	def matsum(self, tensor, axis=0, out=None, alpha=1.0, beta=0.0, allocator=None):
		assert tensor.dtype == np.float32 or tensor.dtype == np.float16
		assert 0 <= axis < tensor.ndim

		if axis == tensor.ndim - 1:
			block = (self.warpSize, 1, 1)
			grid = (prod(tensor.shape[:-1]), 1, 1)

			if out is None:
				out = self.GPUArray.zeros(tensor.shape[:-1], dtype=tensor.dtype, allocator=allocator)
			else:
				assert out.shape == tensor.shape[:-1]

			fn = self.summod.sumOnRow if tensor.dtype == np.float32 else self.summod.sumOnRowFP16
			fn(out, tensor, np.int32(tensor.dimAt(-1)), np.float32(alpha), np.float32(beta), block=block, grid=grid)

		else:
			z, width = prod(tensor.shape[:axis]), prod(tensor.shape[axis + 1:])

			block = (self.NT, 1, 1)
			grid = (roundUpDiv(width, block[0]), 1, z)

			if out is None:
				out = self.GPUArray.zeros(
					tensor.shape[:axis] + tensor.shape[axis + 1:], dtype=tensor.dtype, allocator=allocator
				)
			else:
				assert out.shape == tensor.shape[:axis] + tensor.shape[axis + 1:]

			fn = self.summod.sumOnCol if tensor.dtype == np.float32 else self.summod.sumOnColFP16
			fn(
				out, tensor, np.int32(width), np.int32(tensor.dimAt(axis)), np.float32(alpha), np.float32(beta),
				block=block, grid=grid
			)

		return out


	def matvec(self, mat, vec, axis=0, out=None, alpha=1.0, beta=0.0, allocator=None):
		assert vec.dtype == mat.dtype and (mat.dtype == np.float32 or mat.dtype == np.float16)
		assert vec.ndim == mat.ndim - 1 and 0 <= axis < 2

		h, w = mat.shape[-2:]

		if axis == 1:
			assert mat.dimAt(-1) == vec.dimAt(-1)

			block = (self.warpSize, 1, 1)
			grid = (h, 1, prod(mat.shape[:-2]))

			if out is None:
				out = self.GPUArray.zeros(mat.shape[:-1], dtype=mat.dtype, allocator=allocator)
			else:
				assert out.shape == mat.shape[:-1]

			fn = self.mulmod.vecMulOnRow if mat.dtype == np.float32 else self.mulmod.vecMulOnRowFP16
			fn(out, mat, vec, np.int32(w), np.int32(h), np.float32(alpha), np.float32(beta), block=block, grid=grid)

		else:
			block = (self.NT, 1, 1)
			grid = (roundUpDiv(w, block[0]), 1, prod(mat.shape[:-2]))

			if out is None:
				out = self.GPUArray.zeros(mat.shape[:-2] + (w, ), dtype=mat.dtype, allocator=allocator)
			else:
				assert out.shape == mat.shape[:-2] + (w, )

			fn = self.mulmod.vecMulOnCol if mat.dtype == np.float32 else self.mulmod.vecMulOnColFP16
			fn(out, mat, vec, np.int32(w), np.int32(h), np.float32(alpha), np.float32(beta), block=block, grid=grid)

		return out


	def addVecToMat(self, vec, mat, axis=0, out=None, allocator=None):
		assert vec.dtype == mat.dtype and (mat.dtype == np.float32 or mat.dtype == np.float16)
		assert vec.ndim == mat.ndim - 1 and 0 <= axis < 2

		assert mat.shape[:-2] == vec.shape[:-1]
		out = self.GPUArray.empty(mat.shape, dtype=mat.dtype, allocator=allocator) if out is None else out

		z = prod(mat.shape[:-2])
		n, m = mat.shape[-2:]

		xblock, yblock, _ = self.addblock
		grid = (roundUpDiv(m, xblock), roundUpDiv(n, yblock), z)

		if axis == 1:
			if mat.dimAt(-1) == vec.dimAt(-1):
				fn = self.addmod.opRowVecToMat if mat.dtype == np.float32 else self.addmod.opRowVecToMatFP16
				fn(out, vec, mat, np.int32(n), np.int32(m), block=self.addblock, grid=grid)

			else:
				assert mat.dimAt(-1) % vec.dimAt(-1) == 0

				fn = self.addmod.opRowOneVecToMat if mat.dtype == np.float32 else self.addmod.opRowOneVecToMatFP16
				fn(out, vec, mat, np.int32(n), np.int32(m), np.int32(vec.dimAt(-1)), block=self.addblock, grid=grid)

		else:
			fn = self.addmod.opColVecToMat if mat.dtype == np.float32 else self.addmod.opColVecToMatFP16
			fn(out, vec, mat, np.int32(n), np.int32(m), block=self.addblock, grid=grid)

		return out


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		module = MatModule(Backend.getBackend(deviceIdx))

		for dtype, atol in module.backend.dtypesSupported():
			calcTest(module, dtype, atol)
			batchCalcTest(module, dtype, atol)

			speedTest(module, dtype)
			batchSpeedTest(module, dtype)


def calcTest(module, dtype, atol):
	hostA = np.random.randn(128, 500).astype(dtype)
	hostU = np.random.randn(500).astype(dtype)
	hostV = np.random.randn(128).astype(dtype)
	hostW = np.random.randn(125).astype(dtype)

	A = module.GPUArray.toGpu(hostA)
	u, v, w = module.GPUArray.toGpu(hostU), module.GPUArray.toGpu(hostV), module.GPUArray.toGpu(hostW)

	assert np.allclose(module.addVecToMat(u, A, axis=1).get(), hostA + hostU[np.newaxis, :], atol=atol)
	assert np.allclose(module.addVecToMat(v, A, axis=0).get(), hostA + hostV[:, np.newaxis], atol=atol)
	assert np.allclose(module.addVecToMat(w, A, axis=1).get(), hostA + np.tile(hostW, 4)[np.newaxis, :], atol=atol)

	assert np.allclose(
		module.matsum(A, axis=1).get(), np.sum(hostA.astype(np.float32), axis=1).astype(dtype), atol=atol
	)
	assert np.allclose(
		module.matsum(A, axis=0).get(), np.sum(hostA.astype(np.float32), axis=0).astype(dtype), atol=atol
	)

	out = module.matvec(A, u, axis=1)
	assert np.allclose(out.get(), np.dot(hostA.astype(np.float32), hostU.astype(np.float32)).astype(dtype), atol=atol)

	out = module.matvec(A, v, axis=0)
	assert np.allclose(out.get(), np.dot(hostA.T.astype(np.float32), hostV.astype(np.float32)).astype(dtype), atol=atol)

	hostA = 16.0 * np.random.randn(129, 501).astype(dtype)
	A = module.GPUArray.toGpu(hostA)

	assert np.allclose(module.argmax(A, axis=1).get(), np.argmax(hostA, axis=1))
	assert np.allclose(module.argmax(A, axis=0).get(), np.argmax(hostA, axis=0))


def batchCalcTest(module, dtype, atol):
	hostA = np.random.randn(8, 32, 64).astype(dtype)
	hostV = np.random.randn(8, 64).astype(dtype)
	hostW = np.random.randn(8, 32).astype(dtype)

	A = module.GPUArray.toGpu(hostA)
	v, w = module.GPUArray.toGpu(hostV), module.GPUArray.toGpu(hostW)

	assert np.allclose(module.addVecToMat(w, A, axis=0).get(), hostA + hostW[:, :, np.newaxis])
	assert np.allclose(module.addVecToMat(v, A, axis=1).get(), hostA + hostV[:, np.newaxis, :])

	assert np.allclose(
		module.matsum(A, axis=1).get(), np.sum(hostA.astype(np.float32), axis=1).astype(dtype), atol=atol
	)
	assert np.allclose(
		module.matsum(A, axis=2).get(), np.sum(hostA.astype(np.float32), axis=2).astype(dtype), atol=atol
	)

	out = module.matvec(A, v, axis=1)
	hostOut = np.empty(out.shape, dtype=np.float32)

	for i in range(hostA.shape[0]):
		np.dot(hostA[i].astype(np.float32), hostV[i].astype(np.float32), out=hostOut[i])

	assert np.allclose(out.get(), hostOut.astype(dtype), atol=atol)

	out = module.matvec(A, w, axis=0)
	hostOut = np.empty(out.shape, dtype=np.float32)

	for i in range(hostA.shape[0]):
		np.dot(hostA[i].T.astype(np.float32), hostW[i].astype(np.float32), out=hostOut[i])

	assert np.allclose(out.get(), hostOut.astype(dtype), atol=atol)

	hostA = np.random.normal(scale=16.0, size=(9, 33, 65)).astype(dtype)
	A = module.GPUArray.toGpu(hostA)

	assert np.allclose(module.argmax(A, axis=1).get(), np.argmax(hostA, axis=1))
	assert np.allclose(module.argmax(A, axis=2).get(), np.argmax(hostA, axis=2))


def speedTest(module, dtype):
	A = module.GPUArray.toGpu(np.random.randn(1024, 1024).astype(dtype))
	v = module.GPUArray.toGpu(np.random.randn(1024).astype(dtype))

	bnd = module.backend
	bnd.timeKernel(module.addVecToMat, (v, A, 1, A), logname="%s addVecToMat on rows" % dtype)
	bnd.timeKernel(module.addVecToMat, (v, A, 0, A), logname="%s addVecToMat on cols" % dtype)

	bnd.timeKernel(module.argmax, (A, 1, bnd.memoryPool), logname="%s argmax on rows" % dtype)
	bnd.timeKernel(module.argmax, (A, 0, bnd.memoryPool), logname="%s argmax on cols" % dtype)

	bnd.timeKernel(module.matsum, (A, 1), kwargs={"allocator": bnd.memoryPool}, logname="%s matsum on rows" % dtype)
	bnd.timeKernel(module.matsum, (A, 0), kwargs={"allocator": bnd.memoryPool}, logname="%s matsum on cols" % dtype)


def batchSpeedTest(module, dtype):
	A = module.GPUArray.toGpu(np.random.randn(32, 128, 128).astype(dtype))
	v = module.GPUArray.toGpu(np.random.randn(32, 128).astype(dtype))

	bnd = module.backend
	bnd.timeKernel(module.addVecToMat, (v, A, 1, A), logname="%s batched addVecToMat on rows" % dtype)
	bnd.timeKernel(module.addVecToMat, (v, A, 0, A), logname="%s batched addVecToMat on cols" % dtype)

	bnd.timeKernel(module.argmax, (A, 2, bnd.memoryPool), logname="%s batched argmax on rows" % dtype)
	bnd.timeKernel(module.argmax, (A, 1, bnd.memoryPool), logname="%s batched argmax on cols" % dtype)

	bnd.timeKernel(
		module.matsum, (A, 2), kwargs={"allocator": bnd.memoryPool}, logname="%s batched matsum on rows" % dtype
	)
	bnd.timeKernel(
		module.matsum, (A, 1), kwargs={"allocator": bnd.memoryPool}, logname="%s batched matsum on cols" % dtype
	)


if __name__ == "__main__":
	unittest()
