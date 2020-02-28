import multiprocessing

import numpy as np

from PuzzleLib import Config
from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL import Utils
from PuzzleLib.OpenCL.Utils import queue, globalFills, memoryPool as memPool
from PuzzleLib.OpenCL.ThirdParty import libclblast as libclblast
from PuzzleLib.OpenCL.Kernels.ElementWise import addKer


def autoinit():
	if Config.systemLog:
		print("[%s]: Created CLBlast context" % (Config.libname, ))

	def finishUp():
		libclblast.clblastClearCache()

	import atexit
	atexit.register(finishUp)


if multiprocessing.current_process().name == "MainProcess" or Config.allowMultiContext:
	autoinit()


def sumOnMatrix(A, out=None, cols=True, alpha=1.0, beta=0.0):
	assert A.ndim == 2
	assert A.dtype == np.float32

	if out is None:
		out = Driver.zeros(queue, (A.shape[1], ) if cols else (A.shape[0], ), dtype=np.float32, allocator=memPool)

	if cols:
		ones = globalFills((A.shape[0], ), dtype=A.dtype)
		mulMatrixOnVec(A, ones, out=out, transpMat=True, alpha=alpha, beta=beta)

	else:
		ones = globalFills((A.shape[1], ), dtype=A.dtype)
		mulMatrixOnVec(A, ones, out=out, alpha=alpha, beta=beta)

	return out


def toVectorAddVector(y, x, alpha=1.0):
	assert x.ndim == 1
	assert x.shape == y.shape

	assert x.dtype == y.dtype
	assert x.dtype == np.float32

	n = x.shape[0]
	libclblast.clblastSaxpy(queue.int_ptr, n, alpha, x.int_ptr, x.item_offset, 1, y.int_ptr, y.item_offset, 1)

	return y


def addVectorToVector(x, y, out=None, alpha=1.0, beta=1.0):
	assert x.ndim == 1
	assert x.shape == y.shape
	assert x.dtype == y.dtype and x.dtype == np.float32

	if out is None:
		out = Driver.empty(queue, x.shape, dtype=np.float32, allocator=memPool)

	addKer(np.float32)(out, x, alpha, y, beta)
	return out


def vectorL1Norm(x):
	assert x.ndim == 1
	assert x.dtype == np.float32

	n = x.shape[0]
	asum = Driver.empty(queue, (), dtype=x.dtype, allocator=memPool)

	libclblast.clblastSasum(queue.int_ptr, n, asum.int_ptr, 0, x.int_ptr, x.item_offset, 1)

	return asum.get()


def vectorL2Norm(x):
	assert x.ndim == 1
	assert x.dtype == np.float32

	n = x.shape[0]
	snrm2 = Driver.empty(queue, (), dtype=x.dtype, allocator=memPool)

	libclblast.clblastSnrm2(queue.int_ptr, n, snrm2.int_ptr, 0, x.int_ptr, x.item_offset, 1)

	return snrm2.get()


def mulVectorOnScalar(x, scalar):
	assert x.ndim == 1
	assert x.dtype == np.float32

	n = x.shape[0]
	libclblast.clblastSscal(queue.int_ptr, n, scalar, x.int_ptr, x.item_offset, 1)

	return x


def dot(x, y):
	assert x.ndim == 1
	assert x.shape == y.shape
	assert x.dtype == y.dtype and y.dtype == np.float32

	n = x.shape[0]
	dotp = Driver.empty(queue, (), dtype=x.dtype, allocator=memPool)

	libclblast.clblastSdot(queue.int_ptr, n, dotp.int_ptr, 0, x.int_ptr, x.item_offset, 1, y.int_ptr, y.item_offset, 1)
	return dotp.get()


def mulMatrixOnVec(A, x, out=None, transpMat=False, alpha=1.0, beta=0.0):
	assert A.ndim == 2 and x.ndim == 1
	assert A.dtype == x.dtype and A.dtype == np.float32

	if transpMat:
		assert A.shape[0] == x.shape[0]
	else:
		assert A.shape[1] == x.shape[0]

	if out is None:
		out = Driver.zeros(queue, (A.shape[1] if transpMat else A.shape[0], ), dtype=np.float32, allocator=memPool)

	if transpMat:
		n, m = A.shape
		libclblast.clblastSgemv(queue.int_ptr, 'c', 'n', m, n, alpha, A.int_ptr, A.item_offset,
								m, x.int_ptr, x.item_offset, 1, beta, out.int_ptr, out.item_offset, 1)
	else:
		m, n = A.shape
		libclblast.clblastSgemv(queue.int_ptr, 'c', 't', n, m, alpha, A.int_ptr, A.item_offset,
								n, x.int_ptr, x.item_offset, 1, beta, out.int_ptr, out.item_offset, 1)

	return out


def addMatrixToMatrix(A, B, out=None, alpha=1.0, beta=1.0):
	assert A.ndim == 2 and B.ndim == 2

	assert A.shape == B.shape
	assert A.dtype == B.dtype and A.dtype == np.float32

	if out is None:
		out = Driver.empty(queue, A.shape, dtype=np.float32, allocator=memPool)

	addKer(np.float32)(out, A, alpha, B, beta)
	return out


def transpose(A, out=None):
	assert A.ndim == 2
	assert A.dtype == np.float32

	if out is None:
		out = Driver.empty(queue, (A.shape[1], A.shape[0]), dtype=np.float32, allocator=memPool)

	m, n = A.shape
	libclblast.clblastSomatcopy(queue.int_ptr, 'r', 't', m, n, 1, A.int_ptr, A.item_offset, n, out.int_ptr, 0, m)

	return out


def mulMatrixOnMatrix(A, B, out=None, transpA=False, transpB=False, alpha=1.0, beta=0.0):
	assert not (transpA and transpB)

	assert A.ndim == 2 and B.ndim == 2
	assert A.dtype == B.dtype and A.dtype == np.float32

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
		out = Driver.empty(queue, shape, dtype=np.float32, allocator=memPool)

	if transpA:
		k, m = B.shape
		n = A.shape[1]
		libclblast.clblastSgemm(queue.int_ptr, 'c', 'n', 't', m, n, k, alpha, B.int_ptr,
								B.item_offset, m, A.int_ptr, A.item_offset, n, beta, out.int_ptr, out.item_offset, m)
	elif transpB:
		m, k = B.shape
		n = A.shape[0]
		libclblast.clblastSgemm(queue.int_ptr, 'c', 't', 'n', m, n, k, alpha, B.int_ptr,
								B.item_offset, k, A.int_ptr, A.item_offset, k, beta, out.int_ptr, out.item_offset, m)
	else:
		k, m = B.shape
		n = A.shape[0]
		libclblast.clblastSgemm(queue.int_ptr, 'c', 'n', 'n', m, n, k, alpha, B.int_ptr,
								B.item_offset, m, A.int_ptr, A.item_offset, k, beta, out.int_ptr, out.item_offset, m)

	return out


def outer(x, y, out=None, alpha=1.0, beta=0.0):
	assert x.ndim == 1 and y.ndim == 1
	assert x.dtype == y.dtype and x.dtype == np.float32

	if out is None:
		out = Driver.empty(queue, (x.shape[0], y.shape[0]), dtype=np.float32, allocator=memPool)

	m = x.shape[0]
	n = y.shape[0]
	libclblast.clblastSgemm(queue.int_ptr, 'c', 'n', 't', n, m, 1, alpha, y.int_ptr,
							y.item_offset, n, x.int_ptr, x.item_offset, m, beta, out.int_ptr, out.item_offset, n)

	return out


def unittest():
	vecTest()
	matrixTest()
	onMatrixTest()


def vecTest():
	A = Driver.to_device(queue, np.random.randn(5, 3).astype(np.float32))
	x = Driver.to_device(queue, np.random.randn(A.shape[1]).astype(np.float32))
	y = Driver.to_device(queue, np.random.randn(A.shape[1]).astype(np.float32))

	out = mulMatrixOnVec(A, x)
	assert np.allclose(np.dot(A.get(), x.get()), out.get())

	z = Driver.empty(queue, y.shape, dtype=np.float32)
	Utils.copy(z, y)
	toVectorAddVector(y, x)
	assert np.allclose(z.get() + x.get(), y.get())

	x = Driver.to_device(queue, np.random.randn(A.shape[0]).astype(np.float32))

	out = mulMatrixOnVec(A, x, transpMat=True)
	assert np.allclose(np.dot(A.get().T, x.get()), out.get())

	B = outer(x, y)
	assert np.allclose(np.outer(x.get(), y.get()), B.get())

	x = Driver.to_device(queue, np.random.randn(y.shape[0]).astype(np.float32))
	z = addVectorToVector(x, y)
	assert np.allclose(x.get() + y.get(), z.get())

	assert np.isclose(vectorL1Norm(x), np.linalg.norm(x.get(), ord=1))
	assert np.isclose(vectorL2Norm(x), np.linalg.norm(x.get(), ord=2))

	Utils.copy(z, x)
	mulVectorOnScalar(x, 3.14)
	assert np.allclose(x.get(), z.get() * 3.14)

	assert np.isclose(dot(x, z), np.dot(x.get(), z.get()))


def matrixTest():
	A = Driver.to_device(queue, np.random.randn(5, 3).astype(np.float32))
	B = Driver.to_device(queue, np.random.randn(3, 4).astype(np.float32))
	C = Driver.empty(queue, (A.shape[0], B.shape[1]), dtype=np.float32)
	D = Driver.empty(queue, C.shape, dtype=np.float32)

	C = mulMatrixOnMatrix(A, B)
	assert np.allclose(np.dot(A.get(), B.get()), C.get())

	Utils.copy(D, C)
	E = addMatrixToMatrix(C, D)
	assert np.allclose(C.get() + D.get(), E.get())

	F = mulMatrixOnMatrix(B, C, transpB=True)
	assert np.allclose(np.dot(B.get(), C.get().T), F.get())

	G = mulMatrixOnMatrix(F, B, transpA=True)
	assert np.allclose(np.dot(F.get().T, B.get()), G.get())

	H = transpose(A)
	assert np.allclose(A.get().T, H.get())


def onMatrixTest():
	A = Driver.to_device(queue, np.random.randn(10, 8).astype(np.float32))

	out = sumOnMatrix(A)
	assert np.allclose(np.sum(A.get(), axis=0), out.get())

	out = sumOnMatrix(A, cols=False)
	assert np.allclose(np.sum(A.get(), axis=1), out.get())


if __name__ == "__main__":
	unittest()
