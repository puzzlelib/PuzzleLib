from string import Template

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Utils import memoryPool as memPool, context, queue
from PuzzleLib.OpenCL.Kernels.MatVec import maxmod, minmod
from PuzzleLib.OpenCL.Kernels.Utils import warpSize, roundUp


minMaxBatchTmpl = Template("""

#define NT $NT


__kernel void minMaxBatchOnCol(__global const float *mat, __global float *target, __global int *idxTarget, int w, int h)
{
	__local float maxVals[NT];
	__local int maxIdxs[NT];

	int tidz = get_global_id(2);

	float curMax = $initVal;
	int curIdx = 0;
	float val = 0;

	for (int i = get_local_id(0); i < h; i += NT)
	{
		val = mat[tidz * w * h + get_group_id(0) + i * w];

		if (val $cmpOp curMax)
		{
			curMax = val;
			curIdx = i;
		}
	}

	maxVals[get_local_id(0)] = curMax;
	maxIdxs[get_local_id(0)] = curIdx;
	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_local_id(0) == 0)
	{
		curMax = $initVal;
		curIdx = 0;

		for (int i = 0; i < NT; i++)
		{
			if (maxVals[i] $cmpOp curMax)
			{
				curMax = maxVals[i];
				curIdx = maxIdxs[i];
			}
		}

		target[tidz * w + get_group_id(0)] = curMax;
		idxTarget[tidz * w + get_group_id(0)] = curIdx;
	}
}

""")


vecMatBatchTmpl = Template("""

#define NT $NT


__kernel void opColVecToMatBatch(__global const float *mat, __global const float *vec, __global float *out,
								 int n, int m)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int tidx = get_global_id(0);
	int tidy = get_global_id(1);
	int tidz = get_global_id(2);

	__local float sharedVec[NT];

	if (tx == 0 && tidy < n)
		sharedVec[ty] = vec[tidz * n + tidy];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tidy < n && tidx < m)
		out[tidz * m * n + tidy * m + tidx] = mat[tidz * m * n + tidy * m + tidx] $binaryOp sharedVec[ty];
}

__kernel void opColOneVecToMatBatch(__global const float *mat, __global const float *vec, __global float *out,
									int n, int m)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int tidx = get_global_id(0);
	int tidy = get_global_id(1);
	int tidz = get_global_id(2);

	__local float sharedVec[NT];

	if (tx == 0 && tidy < n)
		sharedVec[ty] = vec[tidy];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tidy < n && tidx < m)
		out[tidz * m * n + tidy * m + tidx] = mat[tidz * m * n + tidy * m + tidx] $binaryOp sharedVec[ty];
}

__kernel void opRowVecToMatBatch(__global const float *mat, __global const float *vec, __global float *out,
								 int n, int m)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int tidx = get_global_id(0);
	int tidy = get_global_id(1);
	int tidz = get_global_id(2);

	__local float sharedVec[NT];

	if (ty == 0 && tidx < m)
		sharedVec[tx] = vec[tidz * m + tidx];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tidy < n && tidx < m)
		out[tidz * m * n + tidy * m + tidx] = mat[tidz * m * n + tidy * m + tidx] $binaryOp sharedVec[tx];
}

__kernel void opRowOneVecToMatBatch(__global const float *mat, __global const float *vec, __global float *out,
									int n, int m)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int tidx = get_global_id(0);
	int tidy = get_global_id(1);
	int tidz = get_global_id(2);

	__local float sharedVec[NT];

	if (ty == 0 && tidx < m)
		sharedVec[tx] = vec[tidx];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tidy < n && tidx < m)
		out[tidz * m * n + tidy * m + tidx] = mat[tidz * m * n + tidy * m + tidx] $binaryOp sharedVec[tx];
}

""")


if context:
	initVal = str(np.finfo(np.float32).min)
	maxBatchMod = Driver.Program(context, minMaxBatchTmpl.substitute(NT=warpSize, initVal=initVal, cmpOp=">")).build()

	initVal = str(np.finfo(np.float32).max)
	minBatchMod = Driver.Program(context, minMaxBatchTmpl.substitute(NT=warpSize, initVal=initVal, cmpOp="<")).build()

	vecMatBatchMod = Driver.Program(context, vecMatBatchTmpl.substitute(NT=warpSize, binaryOp="+")).build()


def argminmaxBatch(mats, axis, mode):
	assert mats.ndim == 3 and mats.dtype == np.float32

	block = (warpSize, 1, 1)

	if axis == 1:
		if mode == "max":
			mod = maxBatchMod
		else:
			mod = minBatchMod

		colKernel = mod.minMaxBatchOnCol
		target = Driver.empty(queue, (mats.shape[0], mats.shape[2]), dtype=np.float32, allocator=memPool)
		idx = Driver.empty(queue, (mats.shape[0], mats.shape[2]), dtype=np.int32, allocator=memPool)

		grid = (mats.shape[2] * warpSize, 1, mats.shape[0])

		colKernel(queue, grid, block, mats.data, target.data, idx.data,
				  np.int32(mats.shape[2]), np.int32(mats.shape[1]))

	elif axis == 2:
		if mode == "max":
			mod = maxmod
		else:
			mod = minmod

		rowKernel = mod.minMaxOnRow

		target = Driver.empty(queue, mats.shape[:2], dtype=np.float32, allocator=memPool)
		idx = Driver.empty(queue, mats.shape[:2], dtype=np.int32, allocator=memPool)

		grid = (mats.shape[0] * mats.shape[1] * block[0], 1, 1)

		rowKernel(queue, grid, block, mats.data, target.data, idx.data,
				  np.int32(mats.shape[2]), np.int32(mats.shape[1]))

	else:
		raise ValueError("Unsupported axis %s was given" % axis)

	return idx


def argmaxBatch(mats, axis=0):
	return argminmaxBatch(mats, axis=axis, mode="max")


def argminBatch(mats, axis=0):
	return argminmaxBatch(mats, axis=axis, mode="min")


def addVecToMatBatch(vecs, mats, axis=0, inplace=False, out=None):
	assert vecs.dtype == mats.dtype and mats.dtype == np.float32
	assert (vecs.ndim == 2 or vecs.ndim == 1) and mats.ndim == 3
	if vecs.ndim == 1:
		vecs = vecs.reshape(1, *vecs.shape)

	assert vecs.shape[0] == 1 or vecs.shape[0] == mats.shape[0]

	if axis == 0:
		assert vecs.shape[1] == mats.shape[1]
	elif axis == 1:
		assert vecs.shape[1] == mats.shape[2]

	block = (warpSize // 4, warpSize // 4, 1)

	g, n, m = mats.shape
	grid = (roundUp(m, block[0]), roundUp(n, block[1]), g)

	if inplace:
		out = mats

	elif out is None:
		out = Driver.empty(queue, mats.shape, dtype=np.float32, allocator=memPool)

	if axis == 0:
		if vecs.shape[0] > 1:
			colKernel = vecMatBatchMod.opColVecToMatBatch
		else:
			colKernel = vecMatBatchMod.opColOneVecToMatBatch

		colKernel(queue, grid, block, mats.data, vecs.data, out.data, np.int32(n), np.int32(m))

	elif axis == 1:
		if vecs.shape[0] > 1:
			rowKernel = vecMatBatchMod.opRowVecToMatBatch
		else:
			rowKernel = vecMatBatchMod.opRowOneVecToMatBatch

		rowKernel(queue, grid, block, mats.data, vecs.data, out.data, np.int32(n), np.int32(m))

	else:
		raise ValueError("Unsupported axis %s was given" % axis)

	return out


def unittest():
	calcTest()
	speedTest()


def calcTest():
	A = Driver.to_device(queue, np.random.randn(16, 2048, 64).astype(np.float32))

	v = Driver.to_device(queue, np.random.randn(16, 64).astype(np.float32))
	w = Driver.to_device(queue, np.random.randn(16, 2048).astype(np.float32))

	assert np.allclose(addVecToMatBatch(w, A, axis=0, inplace=False).get(), A.get() + w.get()[:, :, np.newaxis])
	assert np.allclose(addVecToMatBatch(v, A, axis=1, inplace=False).get(), A.get() + v.get()[:, np.newaxis, :])

	assert np.allclose(argmaxBatch(A, axis=1).get(), np.argmax(A.get(), axis=1))
	assert np.allclose(argmaxBatch(A, axis=2).get(), np.argmax(A.get(), axis=2))


def speedTest():
	from PuzzleLib.OpenCL.Benchmarks.Utils import timeKernel

	A = Driver.to_device(queue, np.random.randn(32, 128, 128).astype(np.float32))
	v = Driver.to_device(queue, np.random.randn(32, 128).astype(np.float32))

	timeKernel(addVecToMatBatch, (v, A, 0, True), logname="addVecToMatBatch on cols")
	timeKernel(addVecToMatBatch, (v, A, 1, True), logname="addVecToMatBatch on rows")

	timeKernel(argmaxBatch, (A, 1), logname="argmaxBatch on cols")
	timeKernel(argmaxBatch, (A, 2), logname="argmaxBatch on rows")


if __name__ == "__main__":
	unittest()
