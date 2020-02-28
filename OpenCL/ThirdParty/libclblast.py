import sys, os, ctypes


_version_list = ["1.4.1"]
if sys.platform == "linux":
	_libclblast_libname_list = ["libclblast.so.%s" % v for v in _version_list]
elif sys.platform == "win32":
	_libclblast_libname_list = [os.path.join(os.path.dirname(__file__), "../Libs/clblast.dll")]
else:
	raise RuntimeError("Unsupported platform for CLBlast")


_libclblast = None
for _libclblast_libname in _libclblast_libname_list:
	try:
		if sys.platform == "win32":
			_libclblast = ctypes.windll.LoadLibrary(_libclblast_libname)
		else:
			_libclblast = ctypes.cdll.LoadLibrary(_libclblast_libname)
	except OSError:
		pass
	else:
		break
if _libclblast is None:
	raise OSError("CLBlast library not found (searched for following version(s): %s)" % _version_list)


class clblastError(Exception):
	pass

class clblastOpenCLCompilerNotAvailable(clblastError):
	pass

class clblastTempBufferAllocFailure(clblastError):
	pass

class clblastOpenCLOutOfResources(clblastError):
	pass

class clblastOpenCLOutOfHostMemory(clblastError):
	pass

class clblastOpenCLBuildProgramFailure(clblastError):
	pass

class clblastInvalidValue(clblastError):
	pass

class clblastInvalidCommandQueue(clblastError):
	pass

class clblastInvalidMemObject(clblastError):
	pass

class clblastInvalidBinary(clblastError):
	pass

class clblastInvalidBuildOptions(clblastError):
	pass

class clblastInvalidProgram(clblastError):
	pass

class clblastInvalidProgramExecutable(clblastError):
	pass

class clblastInvalidKernelName(clblastError):
	pass

class clblastInvalidKernelDefinition(clblastError):
	pass

class clblastInvalidKernel(clblastError):
	pass

class clblastInvalidArgIndex(clblastError):
	pass

class clblastInvalidArgValue(clblastError):
	pass

class clblastInvalidArgSize(clblastError):
	pass

class clblastInvalidKernelArgs(clblastError):
	pass

class clblastInvalidLocalNumDimensions(clblastError):
	pass

class clblastInvalidLocalThreadsTotal(clblastError):
	pass

class clblastInvalidLocalThreadsDim(clblastError):
	pass

class clblastInvalidGlobalOffset(clblastError):
	pass

class clblastInvalidEventWaitList(clblastError):
	pass

class clblastInvalidEvent(clblastError):
	pass

class clblastInvalidOperation(clblastError):
	pass

class clblastInvalidBufferSize(clblastError):
	pass

class clblastInvalidGlobalWorkSize(clblastError):
	pass


class clblastNotImplemented(clblastError):
	pass

class clblastInvalidMatrixA(clblastError):
	pass

class clblastInvalidMatrixB(clblastError):
	pass

class clblastInvalidMatrixC(clblastError):
	pass

class clblastInvalidVectorX(clblastError):
	pass

class clblastInvalidVectorY(clblastError):
	pass

class clblastInvalidDimension(clblastError):
	pass

class clblastInvalidLeadDimA(clblastError):
	pass

class clblastInvalidLeadDimB(clblastError):
	pass

class clblastInvalidLeadDimC(clblastError):
	pass

class clblastInvalidIncrementX(clblastError):
	pass

class clblastInvalidIncrementY(clblastError):
	pass

class clblastInsufficientMemoryA(clblastError):
	pass

class clblastInsufficientMemoryB(clblastError):
	pass

class clblastInsufficientMemoryC(clblastError):
	pass

class clblastInsufficientMemoryX(clblastError):
	pass

class clblastInsufficientMemoryY(clblastError):
	pass


class clblastInvalidBatchCount(clblastError):
	pass

class clblastInvalidOverrideKernel(clblastError):
	pass

class clblastMissingOverrideParameter(clblastError):
	pass

class clblastInvalidLocalMemUsage(clblastError):
	pass

class clblastNoHalfPrecision(clblastError):
	pass

class clblastNoDoublePrecision(clblastError):
	pass

class clblastInvalidVectorScalar(clblastError):
	pass

class clblastInsufficientMemoryScalar(clblastError):
	pass

class clblastDatabaseError(clblastError):
	pass

class clblastUnknownError(clblastError):
	pass

class clblastUnexpectedError(clblastError):
	pass


clblastExceptions = {
	-3: clblastOpenCLCompilerNotAvailable,
	-4: clblastTempBufferAllocFailure,
	-5: clblastOpenCLOutOfResources,
	-6: clblastOpenCLOutOfHostMemory,
	-11: clblastOpenCLBuildProgramFailure,
	-30: clblastInvalidValue,
	-36: clblastInvalidCommandQueue,
	-38: clblastInvalidMemObject,
	-42: clblastInvalidBinary,
	-43: clblastInvalidBuildOptions,
	-44: clblastInvalidProgram,
	-45: clblastInvalidProgramExecutable,
	-46: clblastInvalidKernelName,
	-47: clblastInvalidKernelDefinition,
	-48: clblastInvalidKernel,
	-49: clblastInvalidArgIndex,
	-50: clblastInvalidArgValue,
	-51: clblastInvalidArgSize,
	-52: clblastInvalidKernelArgs,
	-53: clblastInvalidLocalNumDimensions,
	-54: clblastInvalidLocalThreadsTotal,
	-55: clblastInvalidLocalThreadsDim,
	-56: clblastInvalidGlobalOffset,
	-57: clblastInvalidEventWaitList,
	-58: clblastInvalidEvent,
	-59: clblastInvalidOperation,
	-61: clblastInvalidBufferSize,
	-63: clblastInvalidGlobalWorkSize,

	-1024: clblastNotImplemented,
	-1022: clblastInvalidMatrixA,
	-1021: clblastInvalidMatrixB,
	-1020: clblastInvalidMatrixC,
	-1019: clblastInvalidVectorX,
	-1018: clblastInvalidVectorY,
	-1017: clblastInvalidDimension,
	-1016: clblastInvalidLeadDimA,
	-1015: clblastInvalidLeadDimB,
	-1014: clblastInvalidLeadDimC,
	-1013: clblastInvalidIncrementX,
	-1012: clblastInvalidIncrementY,
	-1011: clblastInsufficientMemoryA,
	-1010: clblastInsufficientMemoryB,
	-1009: clblastInsufficientMemoryC,
	-1008: clblastInsufficientMemoryX,
	-1007: clblastInsufficientMemoryY,

	-2049: clblastInvalidBatchCount,
	-2048: clblastInvalidOverrideKernel,
	-2047: clblastMissingOverrideParameter,
	-2046: clblastInvalidLocalMemUsage,
	-2045: clblastNoHalfPrecision,
	-2044: clblastNoDoublePrecision,
	-2043: clblastInvalidVectorScalar,
	-2042: clblastInsufficientMemoryScalar,
	-2041: clblastDatabaseError,
	-2040: clblastUnknownError,
	-2039: clblastUnexpectedError
}


clblastLayout = {
	"r": 101,
	"c": 102
}

clblastTranspose = {
	"n": 111,
	"t": 112,
	"c": 113
}


def clblastCheckStatus(status):
	if status != 0:
		try:
			raise clblastExceptions[status]
		except KeyError:
			raise clblastError


_libclblast.CLBlastClearCache.restype = int
_libclblast.CLBlastClearCache.argtypes = []
def clblastClearCache():
	status = _libclblast.CLBlastClearCache()
	clblastCheckStatus(status)


_libclblast.CLBlastSaxpy.restype = int
_libclblast.CLBlastSaxpy.argtypes = [ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
									 ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
									 ctypes.c_void_p]
def clblastSaxpy(queue, n, alpha, X, offx, incx, Y, offy, incy):
	queue = ctypes.c_void_p(queue)

	status = _libclblast.CLBlastSaxpy(n, alpha, X, offx, incx, Y, offy, incy, ctypes.byref(queue), None)
	clblastCheckStatus(status)


_libclblast.CLBlastSasum.restype = int
_libclblast.CLBlastSasum.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
									 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
def clblastSasum(queue, n, asum, offAsum, X, offx, incx):
	queue = ctypes.c_void_p(queue)

	status = _libclblast.CLBlastSasum(n, asum, offAsum, X, offx, incx, ctypes.byref(queue), None)
	clblastCheckStatus(status)


_libclblast.CLBlastSnrm2.restype = int
_libclblast.CLBlastSnrm2.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
									 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
def clblastSnrm2(queue, n, nrm2, offNrm2, X, offx, incx):
	queue = ctypes.c_void_p(queue)

	status = _libclblast.CLBlastSnrm2(n, nrm2, offNrm2, X, offx, incx, ctypes.byref(queue), None)
	clblastCheckStatus(status)


_libclblast.CLBlastSscal.restype = int
_libclblast.CLBlastSscal.argtypes = [ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
									 ctypes.c_void_p, ctypes.c_void_p]
def clblastSscal(queue, n, alpha, X, offx, incx):
	queue = ctypes.c_void_p(queue)

	status = _libclblast.CLBlastSscal(n, alpha, X, offx, incx, ctypes.byref(queue), None)
	clblastCheckStatus(status)


_libclblast.CLBlastSdot.restype = int
_libclblast.CLBlastSdot.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
									ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
									ctypes.c_void_p]
def clblastSdot(queue, n, dotp, offDotp, X, offx, incx, Y, offy, incy):
	queue = ctypes.c_void_p(queue)

	status = _libclblast.CLBlastSdot(n, dotp, offDotp, X, offx, incx, Y, offy, incy, ctypes.byref(queue), None)
	clblastCheckStatus(status)


_libclblast.CLBlastSgemv.restype = int
_libclblast.CLBlastSgemv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
									 ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
									 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t,
									 ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
def clblastSgemv(queue, layout, transA, m, n, alpha, A, offA, lda, x, offx, incx, beta, y, offy, incy):
	queue = ctypes.c_void_p(queue)

	status = _libclblast.CLBlastSgemv(clblastLayout[layout], clblastTranspose[transA], m, n, alpha, A, offA, lda,
									  x, offx, incx, beta, y, offy, incy, ctypes.byref(queue), None)
	clblastCheckStatus(status)


_libclblast.CLBlastSgemm.restype = int
_libclblast.CLBlastSgemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t,
									 ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
									 ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_void_p,
									 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
def clblastSgemm(queue, layout, transA, transB, m, n, k, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc):
	queue = ctypes.c_void_p(queue)

	status = _libclblast.CLBlastSgemm(clblastLayout[layout], clblastTranspose[transA], clblastTranspose[transB],
									  m, n, k, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc,
									  ctypes.byref(queue), None)
	clblastCheckStatus(status)


_libclblast.CLBlastSgemmBatched.restype = int
_libclblast.CLBlastSgemmBatched.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t,
											ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
											ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_void_p,
											ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
											ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
											ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
											ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
def clblastSgemmBatched(queue, layout, transA, transB, m, n, k, alphas, A, offsetsA, lda, B, offsetsB, ldb, betas,
						C, offsetsC, ldc):
	queue = ctypes.c_void_p(queue)

	batchCount = len(alphas)

	alphas = (ctypes.c_float * batchCount)(*alphas)

	offsetsA = (ctypes.c_size_t * batchCount)(*offsetsA)
	offsetsB = (ctypes.c_size_t * batchCount)(*offsetsB)

	betas = (ctypes.c_float * batchCount)(*betas)
	offsetsC = (ctypes.c_size_t * batchCount)(*offsetsC)

	status = _libclblast.CLBlastSgemmBatched(clblastLayout[layout], clblastTranspose[transA], clblastTranspose[transB],
											 m, n, k, alphas, A, offsetsA, lda, B, offsetsB, ldb, betas, C, offsetsC,
											 ldc, batchCount, ctypes.byref(queue), None)
	clblastCheckStatus(status)


_libclblast.CLBlastSomatcopy.restype = int
_libclblast.CLBlastSomatcopy.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
										 ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
										 ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
def clblastSomatcopy(queue, layout, transA, m, n, alpha, A, offA, lda, B, offB, ldb):
	queue = ctypes.c_void_p(queue)

	status = _libclblast.CLBlastSomatcopy(clblastLayout[layout], clblastTranspose[transA], m, n, alpha, A, offA, lda,
										  B, offB, ldb, ctypes.byref(queue), None)
	clblastCheckStatus(status)
