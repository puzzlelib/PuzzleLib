import sys, os, ctypes


_version_list = ["2.12.0"]
if sys.platform == "linux":
	_libclblas_libname_list = ["libclBLAS.so.%s" % v for v in _version_list]
elif sys.platform == "win32":
	_libclblas_libname_list = [os.path.join(os.path.dirname(__file__), "../Libs/clBLAS.dll")]
else:
	raise RuntimeError("Unsupported platform for CLBlas")


_libclblas = None
for _libclblas_libname in _libclblas_libname_list:
	try:
		if sys.platform == "win32":
			_libclblas = ctypes.windll.LoadLibrary(_libclblas_libname)
		else:
			_libclblas = ctypes.cdll.LoadLibrary(_libclblas_libname)
	except OSError:
		pass
	else:
		break
if _libclblas is None:
	raise OSError("CLBlas library not found (searched for following version(s): %s)" % _version_list)


class clblasError(Exception):
	pass

class clblasInvalidValue(clblasError):
	pass

class clblasInvalidCommandQueue(clblasError):
	pass

class clblasInvalidContext(clblasError):
	pass

class clblasInvalidMemObject(clblasError):
	pass

class clblasInvalidDevice(clblasError):
	pass

class clblasInvalidEventWaitList(clblasError):
	pass

class clblasOutOfResources(clblasError):
	pass

class clblasOutOfHostMemory(clblasError):
	pass

class clblasInvalidOperation(clblasError):
	pass

class clblasCompilerNotAvailable(clblasError):
	pass

class clblasBuildProgramFailure(clblasError):
	pass

class clblasNotImplemented(clblasError):
	pass

class clblasNotInitialized(clblasError):
	pass

class clblasInvalidMatA(clblasError):
	pass

class clblasInvalidMatB(clblasError):
	pass

class clblasInvalidMatC(clblasError):
	pass

class clblasInvalidVecX(clblasError):
	pass

class clblasInvalidVecY(clblasError):
	pass

class clblasInvalidDim(clblasError):
	pass

class clblasInvalidLeadDimA(clblasError):
	pass

class clblasInvalidLeadDimB(clblasError):
	pass

class clblasInvalidLeadDimC(clblasError):
	pass

class clblasInvalidIncX(clblasError):
	pass

class clblasInvalidIncY(clblasError):
	pass

class clblasInsufficientMemMatA(clblasError):
	pass

class clblasInsufficientMemMatB(clblasError):
	pass

class clblasInsufficientMemMatC(clblasError):
	pass

class clblasInsufficientMemVecX(clblasError):
	pass

class clblasInsufficientMemVecY(clblasError):
	pass


clblasExceptions = {
	-30: clblasInvalidValue,
	-36: clblasInvalidCommandQueue,
	-34: clblasInvalidContext,
	-38: clblasInvalidMemObject,
	-33: clblasInvalidDevice,
	-57: clblasInvalidEventWaitList,
	-5: clblasOutOfResources,
	-6: clblasOutOfHostMemory,
	-59: clblasInvalidOperation,
	-3: clblasCompilerNotAvailable,
	-11: clblasBuildProgramFailure,
	-1024: clblasNotImplemented,
	-1023: clblasNotInitialized,
	-1022: clblasInvalidMatA,
	-1021: clblasInvalidMatB,
	-1020: clblasInvalidMatC,
	-1019: clblasInvalidVecX,
	-1018: clblasInvalidVecY,
	-1017: clblasInvalidDim,
	-1016: clblasInvalidLeadDimA,
	-1015: clblasInvalidLeadDimB,
	-1014: clblasInvalidLeadDimC,
	-1013: clblasInvalidIncX,
	-1012: clblasInvalidIncY,
	-1011: clblasInsufficientMemMatA,
	-1010: clblasInsufficientMemMatB,
	-1009: clblasInsufficientMemMatC,
	-1008: clblasInsufficientMemVecX,
	-1007: clblasInsufficientMemVecY
}


clblasOrder = {
	"r": 0,
	"c": 1
}

clblasTranspose = {
	"n": 0,
	"t": 1,
	"c": 2
}


def clblasCheckStatus(status):
	if status != 0:
		try:
			raise clblasExceptions[status]
		except KeyError:
			raise clblasError


_libclblas.clblasGetVersion.restype = int
_libclblas.clblasGetVersion.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def clblasGetVersion():
	major, minor, patch = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()

	status = +_libclblas.clblasGetVersion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
	clblasCheckStatus(status)

	return major.value, minor.value, patch.value


_libclblas.clblasSetup.restype = int
_libclblas.clblasSetup.argtypes = []
def clblasSetup():
	status = _libclblas.clblasSetup()
	clblasCheckStatus(status)


_libclblas.clblasTeardown.restype = int
_libclblas.clblasTeardown.argtypes = []
def clblasTeardown():
	status = _libclblas.clblasTeardown()
	clblasCheckStatus(status)


_libclblas.clblasSaxpy.restype = int
_libclblas.clblasSaxpy.argtypes = [ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
								   ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint,
								   ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
def clblasSaxpy(queue, N, alpha, X, offx, incx, Y, offy, incy):
	queue = ctypes.c_void_p(queue)

	status = _libclblas.clblasSaxpy(N, alpha, X, offx, incx, Y, offy, incy, 1, ctypes.byref(queue), 0, None, None)
	clblasCheckStatus(status)


_libclblas.clblasSasum.restype = int
_libclblas.clblasSasum.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
								   ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint,
								   ctypes.c_void_p, ctypes.c_void_p]
def clblasSasum(queue, N, asum, offAsum, X, offx, incx, scratchBuff):
	queue = ctypes.c_void_p(queue)

	status = _libclblas.clblasSasum(N, asum, offAsum, X, offx, incx, scratchBuff, 1, ctypes.byref(queue), 0, None, None)
	clblasCheckStatus(status)


_libclblas.clblasSnrm2.restype = int
_libclblas.clblasSnrm2.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
								   ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint,
								   ctypes.c_void_p, ctypes.c_void_p]
def clblasSnrm2(queue, N, nrm2, offNrm2, X, offx, incx, scratchBuff):
	queue = ctypes.c_void_p(queue)

	status = _libclblas.clblasSnrm2(N, nrm2, offNrm2, X, offx, incx, scratchBuff, 1, ctypes.byref(queue), 0, None, None)
	clblasCheckStatus(status)


_libclblas.clblasSscal.restype = int
_libclblas.clblasSscal.argtypes = [ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
								   ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
def clblasSscal(queue, N, alpha, X, offx, incx):
	queue = ctypes.c_void_p(queue)

	status = _libclblas.clblasSscal(N, alpha, X, offx, incx, 1, ctypes.byref(queue), 0, None, None)
	clblasCheckStatus(status)


_libclblas.clblasSdot.restype = int
_libclblas.clblasSdot.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
								  ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
								  ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
def clblasSdot(queue, N, dotp, offDotp, X, offx, incx, Y, offy, incy, scratchBuff):
	queue = ctypes.c_void_p(queue)

	status = _libclblas.clblasSdot(N, dotp, offDotp, X, offx, incx, Y, offy, incy, scratchBuff, 1, ctypes.byref(queue),
								   0, None, None)
	clblasCheckStatus(status)


_libclblas.clblasSgemv.restype = int
_libclblas.clblasSgemv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
								   ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
								   ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
								   ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
def clblasSgemv(queue, order, transA, M, N, alpha, A, offA, lda, x, offx, incx, beta, y, offy, incy):
	queue = ctypes.c_void_p(queue)

	status = _libclblas.clblasSgemv(clblasOrder[order], clblasTranspose[transA], M, N, alpha, A, offA, lda,
									x, offx, incx, beta, y, offy, incy, 1, ctypes.byref(queue), 0, None, None)
	clblasCheckStatus(status)


_libclblas.clblasSgemm.restype = int
_libclblas.clblasSgemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t,
								   ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
								   ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p,
								   ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint,
								   ctypes.c_void_p, ctypes.c_void_p]
def clblasSgemm(queue, order, transA, transB, M, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc):
	queue = ctypes.c_void_p(queue)

	status = _libclblas.clblasSgemm(clblasOrder[order], clblasTranspose[transA], clblasTranspose[transB], M, N, K,
									alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc, 1, ctypes.byref(queue),
									0, None, None)
	clblasCheckStatus(status)
