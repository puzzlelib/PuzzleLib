import multiprocessing
import numpy as np

from PuzzleLib import Config

from PuzzleLib.Cuda.Driver import CuBlas
from PuzzleLib.Cuda.Utils import dtypesSupported
from PuzzleLib.Cuda.GPUArray import GPUArray


context = None


def autoinit():
	global context
	context = CuBlas.BlasContext()

	if Config.systemLog:
		print("[%s]: Created CuBlas context (Using version: %s)" % (Config.libname, context.getVersion()))

	context.enableTensorOps(True)


if context is None and (multiprocessing.current_process().name == "MainProcess" or Config.allowMultiContext):
	autoinit()


def unittest():
	vectorTest()

	for dtype, atol in dtypesSupported():
		matrixTest(dtype, atol)
		gbpGbpTest(dtype, atol)
		gbpBgpTest(dtype, atol)
		bgpGbpTest(dtype, atol)
		bgpBgpTest(dtype, atol)


def vectorTest():
	hostX, hostY = np.random.randn(5).astype(np.float32), np.random.randn(5).astype(np.float32)
	x, y = GPUArray.toGpu(hostX), GPUArray.toGpu(hostY)

	assert np.isclose(context.dot(x, y), np.dot(hostX, hostY))
	assert np.isclose(context.l1norm(x), np.linalg.norm(hostX, ord=1))
	assert np.isclose(context.l2norm(x), np.linalg.norm(hostX, ord=2))


def matrixTest(dtype, atol):
	hostA, hostB = np.random.randn(5, 3).astype(dtype), np.random.randn(3, 4).astype(dtype)
	A, B = GPUArray.toGpu(hostA), GPUArray.toGpu(hostB)

	C = context.gemm(A, B)
	hostC = C.get()

	assert np.allclose(np.dot(hostA, hostB), hostC)

	D = context.gemm(B, C, transpB=True)
	hostD = D.get()

	assert np.allclose(np.dot(hostB, hostC.T), hostD)

	E = context.gemm(D, B, transpA=True)
	assert np.allclose(np.dot(hostD.T, hostB), E.get(), atol=atol)


def gbpGbpTest(dtype, atol):
	formatA, formatB, formatOut = CuBlas.GROUPFORMAT_GBP, CuBlas.GROUPFORMAT_GBP, CuBlas.GROUPFORMAT_GBP
	groups = 3

	hostA = np.random.randn(groups, 4, 3).astype(dtype)
	hostB = np.random.randn(groups, hostA.shape[2], 5).astype(dtype)
	hostC = np.random.randn(groups, hostA.shape[1], 6).astype(dtype)
	hostD = np.random.randn(groups, 8, hostC.shape[2]).astype(dtype)

	A, B, C, D = GPUArray.toGpu(hostA), GPUArray.toGpu(hostB), GPUArray.toGpu(hostC), GPUArray.toGpu(hostD)
	out = context.gemmBatched(A, B, formatA=formatA, formatB=formatB, formatOut=formatOut)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostA[i], hostB[i], out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = context.gemmBatched(C, A, formatA=formatA, formatB=formatB, formatOut=formatOut, transpA=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostC[i].T, hostA[i], out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = context.gemmBatched(C, D, formatA=formatA, formatB=formatB, formatOut=formatOut, transpB=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostC[i], hostD[i].T, out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)


def gbpBgpTest(dtype, atol):
	formatA, formatB, formatOut = CuBlas.GROUPFORMAT_GBP, CuBlas.GROUPFORMAT_BGP, CuBlas.GROUPFORMAT_BGP
	groups = 3

	hostA = np.random.randn(groups, 4, 7).astype(dtype)
	hostB = np.random.randn(hostA.shape[2], groups, 5).astype(dtype)
	hostC = np.random.randn(hostA.shape[1], groups, 8).astype(dtype)
	hostD = np.random.randn(6, groups, hostA.shape[2]).astype(dtype)

	A, B, C, D = GPUArray.toGpu(hostA), GPUArray.toGpu(hostB), GPUArray.toGpu(hostC), GPUArray.toGpu(hostD)
	out = context.gemmBatched(A, B, formatA=formatA, formatB=formatB, formatOut=formatOut)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[i], hostB[:, i, :])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = context.gemmBatched(A, C, formatA=formatA, formatB=formatB, formatOut=formatOut, transpA=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[i].T, hostC[:, i, :])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = context.gemmBatched(A, D, formatA=formatA, formatB=formatB, formatOut=formatOut, transpB=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[i], hostD[:, i, :].T)

	assert np.allclose(hostOut, out.get(), atol=atol)


def bgpGbpTest(dtype, atol):
	formatA, formatB, formatOut = CuBlas.GROUPFORMAT_BGP, CuBlas.GROUPFORMAT_GBP, CuBlas.GROUPFORMAT_BGP
	groups = 3

	hostA = np.random.randn(4, groups, 7).astype(dtype)
	hostB = np.random.randn(groups, hostA.shape[2], 5).astype(dtype)
	hostC = np.random.randn(groups, hostA.shape[0], 8).astype(dtype)
	hostD = np.random.randn(groups, 6, hostA.shape[2]).astype(dtype)

	A, B, C, D = GPUArray.toGpu(hostA), GPUArray.toGpu(hostB), GPUArray.toGpu(hostC), GPUArray.toGpu(hostD)
	out = context.gemmBatched(A, B, formatA=formatA, formatB=formatB, formatOut=formatOut)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[:, i, :], hostB[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = context.gemmBatched(A, C, formatA=formatA, formatB=formatB, formatOut=formatOut, transpA=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[:, i, :].T, hostC[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = context.gemmBatched(A, D, formatA=formatA, formatB=formatB, formatOut=formatOut, transpB=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[:, i, :], hostD[i].T)

	assert np.allclose(hostOut, out.get(), atol=atol)


def bgpBgpTest(dtype, atol):
	formatA, formatB, formatOut = CuBlas.GROUPFORMAT_BGP, CuBlas.GROUPFORMAT_BGP, CuBlas.GROUPFORMAT_GBP
	groups = 3

	hostA = np.random.randn(4, groups, 7).astype(dtype)
	hostB = np.random.randn(hostA.shape[2], groups, 5).astype(dtype)
	hostC = np.random.randn(hostA.shape[0], groups, hostB.shape[2]).astype(dtype)

	A, B, C = GPUArray.toGpu(hostA), GPUArray.toGpu(hostB), GPUArray.toGpu(hostC)
	out = context.gemmBatched(A, B, formatA=formatA, formatB=formatB, formatOut=formatOut)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostA[:, i, :], hostB[:, i, :], out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = context.gemmBatched(A, C, formatA=formatA, formatB=formatB, formatOut=formatOut, transpA=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostA[:, i, :].T, hostC[:, i, :], out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = context.gemmBatched(B, C, formatA=formatA, formatB=formatB, formatOut=formatOut, transpB=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostB[:, i, :], hostC[:, i, :].T, out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)


if __name__ == "__main__":
	unittest()
