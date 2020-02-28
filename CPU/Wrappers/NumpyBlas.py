import numpy as np

from PuzzleLib.CPU.CPUArray import CPUArray
from PuzzleLib.CPU.Kernels import ElementWise


def sumOnMatrix(A, out=None, cols=True, alpha=1.0, beta=0.0):
	assert A.ndim == 2
	assert A.flags.c_contiguous
	assert A.dtype == np.float32

	if out is None:
		out = CPUArray.empty((A.shape[1], ) if cols else (A.shape[0], ), dtype=np.float32)

	if alpha == 1.0 and beta == 0.0:
		np.sum(A.data, axis=0 if cols else 1, out=out.data)

	else:
		s = np.sum(A.data, axis=0 if cols else 1)
		np.add(beta * out.data, alpha * s, out=out.data)

	return out


def toVectorAddVector(y, x, alpha=1.0):
	assert x.ndim == 1
	assert x.shape == y.shape
	assert y.flags.forc and x.flags.forc

	assert x.dtype == y.dtype
	assert x.dtype == np.float32

	ElementWise.toVectorAddVectorKer(y.dtype)(y, x, alpha)


def addVectorToVector(x, y, out=None, alpha=1.0, beta=1.0):
	assert x.ndim == 1
	assert x.flags.forc and y.flags.forc
	assert x.shape == y.shape
	assert x.dtype == y.dtype and x.dtype == np.float32

	if out is None:
		out = CPUArray.empty(x.shape, dtype=np.float32)

	ElementWise.addVectorToVectorKer(out, x, y, alpha, beta)
	return out


def vectorL1Norm(x):
	assert x.ndim == 1
	assert x.flags.forc
	assert x.dtype == np.float32

	return np.linalg.norm(x.data, ord=1)


def dot(x, y):
	assert x.ndim == 1
	assert x.shape == y.shape
	assert x.flags.forc and y.flags.forc
	assert x.dtype == y.dtype and y.dtype == np.float32

	return np.vdot(x.data, y.data)


def mulMatrixOnMatrix(A, B, out=None, transpA=False, transpB=False, alpha=1.0, beta=0.0):
	assert not (transpA and transpB)
	assert A.ndim == 2 and B.ndim == 2

	assert alpha == 1.0 and beta == 0.0

	if transpA:
		assert A.shape[0] == B.shape[0]
		shape = (A.shape[1], B.shape[1])

	elif transpB:
		assert A.shape[1] == B.shape[1]
		shape = (A.shape[0], B.shape[0])

	else:
		assert A.shape[1] == B.shape[0]
		shape = (A.shape[0], B.shape[1])

	A = A.data.T if transpA else A.data
	B = B.data.T if transpB else B.data

	if out is None:
		out = CPUArray.empty(shape, dtype=np.float32)

	np.dot(A, B, out=out.data)
	return out
