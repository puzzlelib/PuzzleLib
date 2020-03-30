import numpy as np


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx, initmode=1)

		vectorTest(bnd)

		for dtype, atol in bnd.dtypesSupported():
			matrixTest(bnd, dtype, atol)
			gbpGbpTest(bnd, dtype, atol)
			gbpBgpTest(bnd, dtype, atol)
			bgpGbpTest(bnd, dtype, atol)
			bgpBgpTest(bnd, dtype, atol)


def vectorTest(bnd):
	hostX, hostY = np.random.randn(5).astype(np.float32), np.random.randn(5).astype(np.float32)
	x, y = bnd.GPUArray.toGpu(hostX), bnd.GPUArray.toGpu(hostY)

	assert np.isclose(bnd.blas.dot(x, y), np.dot(hostX, hostY))
	assert np.isclose(bnd.blas.l1norm(x), np.linalg.norm(hostX, ord=1))
	assert np.isclose(bnd.blas.l2norm(x), np.linalg.norm(hostX, ord=2))


def matrixTest(bnd, dtype, atol):
	hostA, hostB = np.random.randn(5, 3).astype(dtype), np.random.randn(3, 4).astype(dtype)
	A, B = bnd.GPUArray.toGpu(hostA), bnd.GPUArray.toGpu(hostB)

	C = bnd.blas.gemm(A, B)
	hostC = C.get()

	assert np.allclose(np.dot(hostA, hostB), hostC)

	D = bnd.blas.gemm(B, C, transpB=True)
	hostD = D.get()

	assert np.allclose(np.dot(hostB, hostC.T), hostD)

	E = bnd.blas.gemm(D, B, transpA=True)
	assert np.allclose(np.dot(hostD.T, hostB), E.get(), atol=atol)


def gbpGbpTest(bnd, dtype, atol):
	formatA, formatB, formatOut = bnd.GroupFormat.gbp.value, bnd.GroupFormat.gbp.value, bnd.GroupFormat.gbp.value
	groups = 3

	hostA = np.random.randn(groups, 4, 3).astype(dtype)
	hostB = np.random.randn(groups, hostA.shape[2], 5).astype(dtype)
	hostC = np.random.randn(groups, hostA.shape[1], 6).astype(dtype)
	hostD = np.random.randn(groups, 8, hostC.shape[2]).astype(dtype)

	A, B = bnd.GPUArray.toGpu(hostA), bnd.GPUArray.toGpu(hostB)
	C, D = bnd.GPUArray.toGpu(hostC), bnd.GPUArray.toGpu(hostD)
	out = bnd.blas.gemmBatched(A, B, formatA=formatA, formatB=formatB, formatOut=formatOut)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostA[i], hostB[i], out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = bnd.blas.gemmBatched(C, A, formatA=formatA, formatB=formatB, formatOut=formatOut, transpA=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostC[i].T, hostA[i], out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = bnd.blas.gemmBatched(C, D, formatA=formatA, formatB=formatB, formatOut=formatOut, transpB=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostC[i], hostD[i].T, out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)


def gbpBgpTest(bnd, dtype, atol):
	formatA, formatB, formatOut = bnd.GroupFormat.gbp.value, bnd.GroupFormat.bgp.value, bnd.GroupFormat.bgp.value
	groups = 3

	hostA = np.random.randn(groups, 4, 7).astype(dtype)
	hostB = np.random.randn(hostA.shape[2], groups, 5).astype(dtype)
	hostC = np.random.randn(hostA.shape[1], groups, 8).astype(dtype)
	hostD = np.random.randn(6, groups, hostA.shape[2]).astype(dtype)

	A, B = bnd.GPUArray.toGpu(hostA), bnd.GPUArray.toGpu(hostB)
	C, D = bnd.GPUArray.toGpu(hostC), bnd.GPUArray.toGpu(hostD)
	out = bnd.blas.gemmBatched(A, B, formatA=formatA, formatB=formatB, formatOut=formatOut)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[i], hostB[:, i, :])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = bnd.blas.gemmBatched(A, C, formatA=formatA, formatB=formatB, formatOut=formatOut, transpA=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[i].T, hostC[:, i, :])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = bnd.blas.gemmBatched(A, D, formatA=formatA, formatB=formatB, formatOut=formatOut, transpB=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[i], hostD[:, i, :].T)

	assert np.allclose(hostOut, out.get(), atol=atol)


def bgpGbpTest(bnd, dtype, atol):
	formatA, formatB, formatOut = bnd.GroupFormat.bgp.value, bnd.GroupFormat.gbp.value, bnd.GroupFormat.bgp.value
	groups = 3

	hostA = np.random.randn(4, groups, 7).astype(dtype)
	hostB = np.random.randn(groups, hostA.shape[2], 5).astype(dtype)
	hostC = np.random.randn(groups, hostA.shape[0], 8).astype(dtype)
	hostD = np.random.randn(groups, 6, hostA.shape[2]).astype(dtype)

	A, B = bnd.GPUArray.toGpu(hostA), bnd.GPUArray.toGpu(hostB)
	C, D = bnd.GPUArray.toGpu(hostC), bnd.GPUArray.toGpu(hostD)
	out = bnd.blas.gemmBatched(A, B, formatA=formatA, formatB=formatB, formatOut=formatOut)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[:, i, :], hostB[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = bnd.blas.gemmBatched(A, C, formatA=formatA, formatB=formatB, formatOut=formatOut, transpA=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[:, i, :].T, hostC[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = bnd.blas.gemmBatched(A, D, formatA=formatA, formatB=formatB, formatOut=formatOut, transpB=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(hostA[:, i, :], hostD[i].T)

	assert np.allclose(hostOut, out.get(), atol=atol)


def bgpBgpTest(bnd, dtype, atol):
	formatA, formatB, formatOut = bnd.GroupFormat.bgp.value, bnd.GroupFormat.bgp.value, bnd.GroupFormat.gbp.value
	groups = 3

	hostA = np.random.randn(4, groups, 7).astype(dtype)
	hostB = np.random.randn(hostA.shape[2], groups, 5).astype(dtype)
	hostC = np.random.randn(hostA.shape[0], groups, hostB.shape[2]).astype(dtype)

	A, B, C = bnd.GPUArray.toGpu(hostA), bnd.GPUArray.toGpu(hostB), bnd.GPUArray.toGpu(hostC)
	out = bnd.blas.gemmBatched(A, B, formatA=formatA, formatB=formatB, formatOut=formatOut)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostA[:, i, :], hostB[:, i, :], out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = bnd.blas.gemmBatched(A, C, formatA=formatA, formatB=formatB, formatOut=formatOut, transpA=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostA[:, i, :].T, hostC[:, i, :], out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)

	out = bnd.blas.gemmBatched(B, C, formatA=formatA, formatB=formatB, formatOut=formatOut, transpB=True)

	hostOut = np.empty(out.shape, dtype=dtype)
	for i in range(groups):
		np.dot(hostB[:, i, :], hostC[:, i, :].T, out=hostOut[i])

	assert np.allclose(hostOut, out.get(), atol=atol)


if __name__ == "__main__":
	unittest()
