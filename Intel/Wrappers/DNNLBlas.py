import numpy as np

from PuzzleLib.CPU.CPUArray import CPUArray
from PuzzleLib.Intel.ThirdParty import libdnnl


def mulMatrixOnMatrix(A, B, out=None, transpA=False, transpB=False, alpha=1.0, beta=0.0):
	assert not (transpA and transpB)
	assert A.ndim == 2 and B.ndim == 2

	assert A.dtype == B.dtype and A.dtype == np.float32
	assert A.flags.c_contiguous and B.flags.c_contiguous

	if transpA:
		assert A.shape[0] == B.shape[0]
		shape = (A.shape[1], B.shape[1])
	elif transpB:
		assert A.shape[1] == B.shape[1]
		shape = (A.shape[0], B.shape[0])
	else:
		assert A.shape[1] == B.shape[0]
		shape = (A.shape[0], B.shape[1])

	if out is None:
		out = CPUArray.empty(shape, dtype=np.float32)

	if transpA:
		k, m = A.shape
		n = B.shape[1]
		libdnnl.dnnl_sgemm('t', 'n', m, n, k, alpha, A.ptr, m, B.ptr, n, beta, out.ptr, n)
	elif transpB:
		m, k = A.shape
		n = B.shape[0]
		libdnnl.dnnl_sgemm('n', 't', m, n, k, alpha, A.ptr, k, B.ptr, k, beta, out.ptr, n)
	else:
		m, k = A.shape
		n = B.shape[1]
		libdnnl.dnnl_sgemm('n', 'n', m, n, k, alpha, A.ptr, k, B.ptr, n, beta, out.ptr, n)

	return out


def unittest():
	A = CPUArray.toDevice(np.random.randn(5, 3).astype(np.float32))
	B = CPUArray.toDevice(np.random.randn(3, 4).astype(np.float32))

	C = mulMatrixOnMatrix(A, B)
	assert np.allclose(np.dot(A.get(), B.get()), C.get())

	F = mulMatrixOnMatrix(B, C, transpB=True)
	assert np.allclose(np.dot(B.get(), C.get().T), F.get())

	G = mulMatrixOnMatrix(F, B, transpA=True)
	assert np.allclose(np.dot(F.get().T, B.get()), G.get())


if __name__ == "__main__":
	unittest()
