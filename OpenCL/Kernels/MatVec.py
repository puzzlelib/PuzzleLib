from string import Template

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Utils import memoryPool as memPool, context, queue
from PuzzleLib.OpenCL.Kernels.Utils import roundUp


minMaxTmpl = Template("""

#define NT1 $NT1
#define NT2 $NT2


__kernel void minMaxOnRow(__global const float *mat, __global float *target, __global int *idxTarget, int w, int h)
{
	__local float maxVals[NT1];
	__local int maxIdxs[NT1];

	float curMax = $initVal;
	int curIdx = 0;
	float val = 0;

	for (int i = get_local_id(0); i < w; i += NT1)
	{
		val = mat[get_group_id(0) * w + i];

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

		for (int i = 0; i < NT1; i++)
		{
			if (maxVals[i] $cmpOp curMax)
			{
				curMax = maxVals[i];
				curIdx = maxIdxs[i];
			}
		}

		target[get_group_id(0)] = curMax;
		idxTarget[get_group_id(0)] = curIdx;
	}
}

__kernel void minMaxOnCol(__global const float *mat, __global float *target, __global int *idxTarget, int w, int h)
{
	float curMax = $initVal;
	int curIdx = -1;

	int gid = get_local_id(0) + get_group_id(0) * NT2;
	if (gid > w) return;

	for (int i = 0; i < h; i++)
	{
		float val = mat[gid + i * w];

		if (val $cmpOp curMax)
		{
			curMax = val;
			curIdx = i;
		}
	}

	target[gid] = curMax;
	idxTarget[gid] = curIdx;
}

""")


vecMatTmpl = Template("""

#define NT $NT


__kernel void opColVecToMat(__global const float *mat, __global const float *vec, __global float *out, int n, int m,
							float alpha, float beta, float gamma)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int tidx = get_global_id(0);
	int tidy = get_global_id(1);

	__local float sharedVec[NT];

	if (tx == 0 && tidy < n)
		sharedVec[ty] = vec[tidy];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tidy < n && tidx < m)
		if ($nonZeroGamma)
			out[tidy * m + tidx] = gamma * out[tidy * m + tidx] +
			alpha * mat[tidy * m + tidx] $binaryOp beta * sharedVec[ty];
		else
			out[tidy * m + tidx] = alpha * mat[tidy * m + tidx] $binaryOp beta * sharedVec[ty];
}

__kernel void opRowVecToMat(__global const float *mat, __global const float *vec, __global float *out, int n, int m,
							float alpha, float beta, float gamma)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int tidx = get_global_id(0);
	int tidy = get_global_id(1);

	__local float sharedVec[NT];

	if (ty == 0 && tidx < m)
		sharedVec[tx] = vec[tidx];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tidy < n && tidx < m)
		if ($nonZeroGamma)
			out[tidy * m + tidx] = gamma * out[tidy * m + tidx] +
			alpha * mat[tidy * m + tidx] $binaryOp beta * sharedVec[tx];
		else
			out[tidy * m + tidx] = alpha * mat[tidy * m + tidx] $binaryOp beta * sharedVec[tx];
}

__kernel void opRowOneVecToMat(__global const float *mat, __global const float *vec, __global float *out,
								int n, int m, int p, float alpha, float beta, float gamma)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int tidx = get_global_id(0);
	int tidy = get_global_id(1);

	__local float sharedVec[NT];

	if (ty == 0 && tidx < m)
		sharedVec[tx] = vec[tidx % p];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tidy < n && tidx < m)
		if ($nonZeroGamma)
			out[tidy * m + tidx] = gamma * out[tidy * m + tidx] +
			alpha * mat[tidy * m + tidx] $binaryOp beta * sharedVec[tx];
		else
			out[tidy * m + tidx] = alpha * mat[tidy * m + tidx] $binaryOp beta * sharedVec[tx];
}

""")


maxmod, minmod = None, None
NT1 = 64
NT2 = NT1 * 4

if context:
	initVal = str(np.finfo(np.float32).min)
	maxmod = Driver.Program(context, minMaxTmpl.substitute(NT1=NT1, NT2=NT2, initVal=initVal, cmpOp=">")).build()

	initVal = str(np.finfo(np.float32).max)
	minmod = Driver.Program(context, minMaxTmpl.substitute(NT1=NT1, NT2=NT2, initVal=initVal, cmpOp="<")).build()

	gammaVecMatMod = Driver.Program(context, vecMatTmpl.substitute(NT=16, binaryOp="+", nonZeroGamma="true")).build()
	nonGammaVecMatMod = Driver.Program(context, vecMatTmpl.substitute(NT=16, binaryOp="+",
																	  nonZeroGamma="false")).build()


def argminmax(mat, axis, mode):
	assert mat.ndim == 2 and mat.dtype == np.float32

	if mode == "max":
		mod = maxmod
	else:
		mod = minmod

	if axis == 0:
		colKernel = mod.minMaxOnCol

		block = (NT2, 1, 1)
		grid = (roundUp(mat.shape[1], block[0]), 1, 1)

		target = Driver.empty(queue, (mat.shape[1], ), dtype=np.float32, allocator=memPool)
		idx = Driver.empty(queue, (mat.shape[1], ), dtype=np.int32, allocator=memPool)

		colKernel(queue, grid, block, mat.data, target.data, idx.data, np.int32(mat.shape[1]), np.int32(mat.shape[0]))

	elif axis == 1:
		rowKernel = mod.minMaxOnRow

		block = (NT1, 1, 1)
		grid = (mat.shape[0] * block[0], 1, 1)

		target = Driver.empty(queue, (mat.shape[0], ), dtype=np.float32, allocator=memPool)
		idx = Driver.empty(queue, (mat.shape[0], ), dtype=np.int32, allocator=memPool)

		rowKernel(queue, grid, block, mat.data, target.data, idx.data, np.int32(mat.shape[1]), np.int32(mat.shape[0]))

	else:
		raise NotImplementedError()

	return idx


def argmax(mat, axis=0):
	return argminmax(mat, axis, "max")


def argmin(mat, axis=0):
	return argminmax(mat, axis, "min")


def addVecToMat(vec, mat, axis=0, inplace=True, out=None, alpha=1.0, beta=1.0, gamma=0.0):
	assert vec.dtype == mat.dtype and mat.dtype == np.float32
	assert vec.ndim == 1 and mat.ndim == 2
	if axis == 0:
		assert vec.shape[0] == mat.shape[0]
	elif axis == 1:
		assert vec.shape[0] == mat.shape[1] or mat.shape[1] % vec.shape[0] == 0

	block = (16, 16, 1)

	if gamma != 0.0:
		mod = gammaVecMatMod
	else:
		mod = nonGammaVecMatMod

	n, m = mat.shape
	gridx = roundUp(m, block[0])
	gridy = roundUp(n, block[1])
	grid = (gridx, gridy, 1)

	if inplace:
		out = mat

	elif out is None:
		out = Driver.empty(queue, mat.shape, dtype=np.float32, allocator=memPool)

	if axis == 0:
		colKernel = mod.opColVecToMat
		colKernel(queue, grid, block, mat.data, vec.data, out.data, np.int32(n), np.int32(m),
				  np.float32(beta), np.float32(alpha), np.float32(gamma))

	elif axis == 1:
		if vec.shape[0] == mat.shape[1]:
			rowKernel = mod.opRowVecToMat
			rowKernel(queue, grid, block, mat.data, vec.data, out.data, np.int32(n), np.int32(m),
					  np.float32(beta), np.float32(alpha), np.float32(gamma))
		else:
			rowKernel = mod.opRowOneVecToMat
			rowKernel(queue, grid, block, mat.data, vec.data, out.data, np.int32(n), np.int32(m),
					  np.int32(vec.shape[0]), np.float32(beta), np.float32(alpha), np.float32(gamma))

	else:
		raise ValueError("Unknown axis %s was given" % axis)

	return out


def unittest():
	calcTest()
	speedTest()


def calcTest():
	A = Driver.to_device(queue, np.random.randn(128, 500).astype(np.float32))

	v = Driver.to_device(queue, np.random.randn(500).astype(np.float32))
	w = Driver.to_device(queue, np.random.randn(128).astype(np.float32))
	m = Driver.to_device(queue, np.random.randn(125).astype(np.float32))

	assert np.allclose(A.get() + v.get()[np.newaxis, :], addVecToMat(v, A, axis=1).get())
	assert np.allclose(A.get() + w.get()[:, np.newaxis], addVecToMat(w, A, axis=0).get())
	assert np.allclose(A.get() + np.tile(m.get(), 4)[np.newaxis, :], addVecToMat(m, A, axis=1).get())

	assert np.allclose(argmax(A, axis=1).get(), np.argmax(A.get(), axis=1))
	assert np.allclose(argmax(A, axis=0).get(), np.argmax(A.get(), axis=0))


def speedTest():
	from PuzzleLib.OpenCL.Benchmarks.Utils import timeKernel

	A = Driver.to_device(queue, np.random.randn(1024, 1024).astype(np.float32))
	v = Driver.to_device(queue, np.random.randn(1024).astype(np.float32))

	timeKernel(addVecToMat, (v, A, 1, True), logname="addVecToMat on rows")
	timeKernel(addVecToMat, (v, A, 0, True), logname="addVecToMat on cols")

	timeKernel(argmax, (A, 1), logname="argmax on rows")
	timeKernel(argmax, (A, 0), logname="argmax on cols")


if __name__ == "__main__":
	unittest()
