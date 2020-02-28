import sys, ctypes, warnings, itertools


_version_list = ["10.2.1.89", "10.2.0.243", "10.1.168", 10.0, 9.2, 9.1, 9.0, 8.0]
if sys.platform == "linux":
	_libnppc_libname_list = [
		"libnppc.so.%s" % v for v in _version_list
	]

	_libnppi_libname_list = [
		"libnppi%s.so.%s" % ("g" if isinstance(v, str) or v >= 9.0 else "", v) for v in _version_list
	]

elif sys.platform == "darwin":
	_libnppc_libname_list = ["libnppc.dylib"]
	_libnppi_libname_list = ["libnppi.dylib"]

elif sys.platform == "win32":
	_libnppc_libname_list = ["nppc64_10.dll"] + [
		"nppc64_%s.dll" % int(10 * v) for v in _version_list[-5:]
	]

	_libnppi_libname_list = ["nppig64_10.dll"] + [
		"nppi%s64_%s.dll" % ("g" if v >= 9.0 else "", int(10 * v)) for v in _version_list[-5:]
	]

else:
	raise RuntimeError("Unsupported platform for npp")


_libnppc, _libnppi = None, None
for _libnppc_libname, _libnppi_libname in itertools.zip_longest(_libnppc_libname_list, _libnppi_libname_list):
	try:
		if sys.platform == "win32":
			_libnppc = ctypes.windll.LoadLibrary(_libnppc_libname)
			_libnppi = ctypes.windll.LoadLibrary(_libnppi_libname)
		else:
			_libnppc = ctypes.cdll.LoadLibrary(_libnppc_libname)
			_libnppi = ctypes.cdll.LoadLibrary(_libnppi_libname)
	except OSError:
		pass
	else:
		break
if _libnppc is None or _libnppi is None:
	raise OSError("npp library not found (searched for following version(s): %s)" % _version_list)


class nppError(Exception):
	pass

class nppNotSupportedModeError(nppError):
	pass

class nppInvalidHostPointerError(nppError):
	pass

class nppInvalidDevicePointerError(nppError):
	pass

class nppLutPaletteBitsizeError(nppError):
	pass

class nppZcModeNotSupportedError(nppError):
	pass

class nppNotSufficientComputeCapability(nppError):
	pass

class nppTextureBindError(nppError):
	pass

class nppWrongIntersectionRoiError(nppError):
	pass

class nppHaarClassifierPixelMatchError(nppError):
	pass

class nppMemfreeError(nppError):
	pass

class nppMemsetError(nppError):
	pass

class nppMemcpyError(nppError):
	pass

class nppAlignmentError(nppError):
	pass

class nppCudaKernelExecutionError(nppError):
	pass

class nppRoundModeNotSupportedError(nppError):
	pass

class nppQualityIndexError(nppError):
	pass

class nppResizeNoOperationError(nppError):
	pass

class nppOverflowError(nppError):
	pass

class nppNotEvenStepError(nppError):
	pass

class nppHistogramNumberOfLevelsError(nppError):
	pass

class nppLutNumberOfLevelsError(nppError):
	pass

class nppCorruptedDataError(nppError):
	pass

class nppChannelOrderError(nppError):
	pass

class nppZeroMaskValueError(nppError):
	pass

class nppQuadrangleError(nppError):
	pass

class nppRectangleError(nppError):
	pass

class nppCoefficientError(nppError):
	pass

class nppNumberOfChannelsError(nppError):
	pass

class nppCoiError(nppError):
	pass

class nppDivisorError(nppError):
	pass

class nppChannelError(nppError):
	pass

class nppStrideError(nppError):
	pass

class nppAnchorError(nppError):
	pass

class nppMaskSizeError(nppError):
	pass

class nppResizeFactorError(nppError):
	pass

class nppInterpolationError(nppError):
	pass

class nppMirrorFlipError(nppError):
	pass

class nppMoment00ZeroError(nppError):
	pass

class nppThresholdNegativeLevelError(nppError):
	pass

class nppThresholdError(nppError):
	pass

class nppContextMatchError(nppError):
	pass

class nppFftFlagError(nppError):
	pass

class nppFftOrderError(nppError):
	pass

class nppStepError(nppError):
	pass

class nppScaleRangeError(nppError):
	pass

class nppDataTypeError(nppError):
	pass

class nppOutOfRangeError(nppError):
	pass

class nppDivideByZeroError(nppError):
	pass

class nppMemoryAllocationError(nppError):
	pass

class nppNullPointerError(nppError):
	pass

class nppRangeError(nppError):
	pass

class nppSizeError(nppError):
	pass

class nppBadArgumentError(nppError):
	pass

class nppNoMemoryError(nppError):
	pass

class nppNotImplementedError(nppError):
	pass

class nppErrorReserved(nppError):
	pass


class nppWarning(Warning):
	pass

class nppNoOperationWarning(nppWarning):
	pass

class nppDivideByZeroWarning(nppWarning):
	pass

class nppAffineQuadIncorrectWarning(nppWarning):
	pass

class nppWrongIntersectionRoiWarning(nppWarning):
	pass

class nppWrongIntersectionQuadWarning(nppWarning):
	pass

class nppDoubleSizeWarning(nppWarning):
	pass

class nppMisalignedDstRoiWarning(nppWarning):
	pass


nppExceptions = {
	-9999: nppNotSupportedModeError,
	-1032: nppInvalidHostPointerError,
	-1031: nppInvalidDevicePointerError,
	-1030: nppLutPaletteBitsizeError,
	-1028: nppZcModeNotSupportedError,
	-1027: nppNotSufficientComputeCapability,
	-1024: nppTextureBindError,
	-1020: nppWrongIntersectionRoiError,
	-1006: nppHaarClassifierPixelMatchError,
	-1005: nppMemfreeError,
	-1004: nppMemsetError,
	-1003: nppMemcpyError,
	-1002: nppAlignmentError,
	-1000: nppCudaKernelExecutionError,
	-213: nppRoundModeNotSupportedError,
	-210: nppQualityIndexError,
	-201: nppResizeNoOperationError,
	-109: nppOverflowError,
	-108: nppNotEvenStepError,
	-107: nppHistogramNumberOfLevelsError,
	-106: nppLutNumberOfLevelsError,
	-61: nppCorruptedDataError,
	-60: nppChannelOrderError,
	-59: nppZeroMaskValueError,
	-58: nppQuadrangleError,
	-57: nppRectangleError,
	-56: nppCoefficientError,
	-53: nppNumberOfChannelsError,
	-52: nppCoiError,
	-51: nppDivisorError,
	-47: nppChannelError,
	-37: nppStrideError,
	-34: nppAnchorError,
	-33: nppMaskSizeError,
	-23: nppResizeFactorError,
	-22: nppInterpolationError,
	-21: nppMirrorFlipError,
	-20: nppMoment00ZeroError,
	-19: nppThresholdNegativeLevelError,
	-18: nppThresholdError,
	-17: nppContextMatchError,
	-16: nppFftFlagError,
	-15: nppFftOrderError,
	-14: nppStepError,
	-13: nppScaleRangeError,
	-12: nppDataTypeError,
	-11: nppOutOfRangeError,
	-10: nppDivideByZeroError,
	-9: nppMemoryAllocationError,
	-8: nppNullPointerError,
	-7: nppRangeError,
	-6: nppSizeError,
	-5: nppBadArgumentError,
	-4: nppNoMemoryError,
	-3: nppNotImplementedError,
	-2: nppError,
	-1: nppErrorReserved
}


nppWarnings = {
	1: nppNoOperationWarning,
	6: nppDivideByZeroWarning,
	28: nppAffineQuadIncorrectWarning,
	29: nppWrongIntersectionRoiWarning,
	30: nppWrongIntersectionQuadWarning,
	35: nppDoubleSizeWarning,
	10000: nppMisalignedDstRoiWarning
}


NppiInterpolationMode = {
	"NPPI_INTER_UNDEFINED": 0,
	"NPPI_INTER_NN": 1,
	"NPPI_INTER_LINEAR": 2,
	"NPPI_INTER_CUBIC": 4,
	"NPPI_INTER_CUBIC2P_BSPLINE": 5,
	"NPPI_INTER_CUBIC2P_CATMULLROM": 6,
	"NPPI_INTER_CUBIC2P_B05C03": 7,
	"NPPI_INTER_SUPER": 8,
	"NPPI_INTER_LANCZOS": 16,
	"NPPI_INTER_LANCZOS3_ADVANCED": 17,
	"NPPI_SMOOTH_EDGE": 1 << 31
}

class NppLibraryVersion(ctypes.Structure):
	_fields_ = [
		("major", ctypes.c_int),
		("minor", ctypes.c_int),
		("build", ctypes.c_int)
	]

class NppiSize(ctypes.Structure):
	_fields_ = [
		("width", ctypes.c_int),
		("height", ctypes.c_int)
	]

class NppiRect(ctypes.Structure):
	_fields_ = [
		("x", ctypes.c_int),
		("y", ctypes.c_int),
		("width", ctypes.c_int),
		("height", ctypes.c_int)
	]


def nppCheckStatus(status):
	if status < 0:
		try:
			raise nppExceptions[status]
		except KeyError:
			raise nppError
	elif status > 0:
		warnings.warn(str(nppWarnings[status]))


_libnppc.nppGetLibVersion.restype = ctypes.POINTER(NppLibraryVersion)
_libnppc.nppGetLibVersion.argtypes = []
def nppGetLibVersion():
	libversion = _libnppc.nppGetLibVersion().contents
	return libversion.major, libversion.minor, libversion.build


_libnppc.nppGetStream.restype = ctypes.c_void_p
_libnppc.nppGetStream.argtypes = []
def nppGetStream():
	streamId = _libnppc.nppGetStream()
	return streamId.value


_libnppc.nppSetStream.restype = None
_libnppc.nppSetStream.argtypes = [ctypes.c_void_p]
def nppSetStream(streamId):
	_libnppc.nppSetStream(streamId)


_libnppi.nppiGetResizeRect.restype = ctypes.c_int
_libnppi.nppiGetResizeRect.argtypes = [NppiRect, ctypes.POINTER(NppiRect), ctypes.c_double, ctypes.c_double,
									   ctypes.c_double, ctypes.c_double, ctypes.c_int]
def nppiGetResizeRect(rect, nXFactor, nYFactor, nXShift, nYShift, interpolation):
	x, y, width, height = rect

	inrect = NppiRect(x, y, width, height)
	outrect = NppiRect()

	status = _libnppi.nppiGetResizeRect(inrect, ctypes.byref(outrect), nXFactor, nYFactor, nXShift, nYShift,
										interpolation)
	nppCheckStatus(status)

	return outrect.x, outrect.y, outrect.width, outrect.height


nppiResizeSqrPixelResType = ctypes.c_int
nppiResizeSqrPixelArgTypes = [ctypes.c_void_p, NppiSize, ctypes.c_int, NppiRect, ctypes.c_void_p, ctypes.c_int,
							  NppiRect, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
							  ctypes.c_int]
def nppiResizeSqrPixel(dataType, memoryType, src, srcSize, nSrcStep, srcROI, dst, nDstStep, dstROI, nXFactor, nYFactor,
					   nXShift, nYShift, interpolation):
	inSize = NppiSize(*srcSize)
	inROI, outROI = NppiRect(*srcROI), NppiRect(*dstROI)

	f = _libnppi["%s_%s_%s" % (nppiResizeSqrPixel.__name__, dataType, memoryType)]
	f.restype = nppiResizeSqrPixelResType
	f.argtypes = nppiResizeSqrPixelArgTypes

	status = f(src, inSize, nSrcStep, inROI, dst, nDstStep, outROI, nXFactor, nYFactor, nXShift, nYShift, interpolation)
	nppCheckStatus(status)


_libnppi.nppiGetRotateQuad.restype = ctypes.c_int
_libnppi.nppiGetRotateQuad.argtypes = [NppiRect, ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_double,
									   ctypes.c_double]
def nppiGetRotateQuad(rect, nAngle, nShiftX, nShiftY):
	x, y, width, height = rect
	inrect = NppiRect(x, y, width, height)

	aQuad = (ctypes.c_double * 8)()

	status = _libnppi.nppiGetRotateQuad(inrect, aQuad, nAngle, nShiftX, nShiftY)
	nppCheckStatus(status)

	return list(aQuad)


_libnppi.nppiGetRotateBound.restype = ctypes.c_int
_libnppi.nppiGetRotateBound.argtypes = [NppiRect, ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_double,
										ctypes.c_double]
def nppiGetRotateBound(rect, nAngle, nShiftX, nShiftY):
	x, y, width, height = rect
	inrect = NppiRect(x, y, width, height)

	aBoundingBox = (ctypes.c_double * 4)()

	status = _libnppi.nppiGetRotateBound(inrect, aBoundingBox, nAngle, nShiftX, nShiftY)
	nppCheckStatus(status)

	return list(aBoundingBox)


nppiRotateResType = ctypes.c_int
nppiRotateArgTypes = [ctypes.c_void_p, NppiSize, ctypes.c_int, NppiRect, ctypes.c_void_p, ctypes.c_int, NppiRect,
					  ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
def nppiRotate(dataType, memoryType, src, srcSize, nSrcStep, srcROI, dst, nDstStep, dstROI, nAngle, nShiftX, nShiftY,
			   interpolation):
	inSize = NppiSize(*srcSize)
	inROI, outROI = NppiRect(*srcROI), NppiRect(*dstROI)

	f = _libnppi["%s_%s_%s" % (nppiRotate.__name__, dataType, memoryType)]
	f.restype = nppiRotateResType
	f.argtypes = nppiRotateArgTypes

	status = f(src, inSize, nSrcStep, inROI, dst, nDstStep, outROI, nAngle, nShiftX, nShiftY, interpolation)
	nppCheckStatus(status)


nppiWarpAffineResType = ctypes.c_int
nppiWarpAffineArgTypes = [ctypes.c_void_p, NppiSize, ctypes.c_int, NppiRect, ctypes.c_void_p, ctypes.c_int, NppiRect,
						  ctypes.c_void_p, ctypes.c_int]
def nppiWarpAffine(dataType, memoryType, src, srcSize, nSrcStep, srcROI, dst, nDstStep, dstROI, coeffs, interpolation):
	inSize = NppiSize(*srcSize)
	inROI, outROI = NppiRect(*srcROI), NppiRect(*dstROI)

	f = _libnppi["%s_%s_%s" % (nppiWarpAffine.__name__, dataType, memoryType)]
	f.restype = nppiWarpAffineResType
	f.argtypes = nppiWarpAffineArgTypes

	aCoeffs = (ctypes.c_double * 6)(*coeffs)

	status = f(src, inSize, nSrcStep, inROI, dst, nDstStep, outROI, aCoeffs, interpolation)
	nppCheckStatus(status)


nppiWarpAffineBackResType = ctypes.c_int
nppiWarpAffineBackArgTypes = [ctypes.c_void_p, NppiSize, ctypes.c_int, NppiRect, ctypes.c_void_p, ctypes.c_int,
							  NppiRect, ctypes.c_void_p, ctypes.c_int]
def nppiWarpAffineBack(dataType, memoryType, src, srcSize, nSrcStep, srcROI, dst, nDstStep, dstROI, coeffs,
					   interpolation):
	inSize = NppiSize(*srcSize)
	inROI, outROI = NppiRect(*srcROI), NppiRect(*dstROI)

	f = _libnppi["%s_%s_%s" % (nppiWarpAffineBack.__name__, dataType, memoryType)]
	f.restype = nppiWarpAffineBackResType
	f.argtypes = nppiWarpAffineBackArgTypes

	aCoeffs = (ctypes.c_double * 6)(*coeffs)

	status = f(src, inSize, nSrcStep, inROI, dst, nDstStep, outROI, aCoeffs, interpolation)
	nppCheckStatus(status)


nppiWarpAffineQuadResType = ctypes.c_int
nppiWarpAffineQuadArgTypes = [ctypes.c_void_p, NppiSize, ctypes.c_int, NppiRect, ctypes.c_void_p, ctypes.c_void_p,
							  ctypes.c_int, NppiRect, ctypes.c_void_p, ctypes.c_int]
def nppiWarpAffineQuad(dataType, memoryType, src, srcSize, nSrcStep, srcROI, srcQuad, dst, nDstStep, dstROI, dstQuad,
					   interpolation):
	inSize = NppiSize(*srcSize)
	inROI, outROI = NppiRect(*srcROI), NppiRect(*dstROI)

	f = _libnppi["%s_%s_%s" % (nppiWarpAffineQuad.__name__, dataType, memoryType)]
	f.restype = nppiWarpAffineQuadResType
	f.argtypes = nppiWarpAffineQuadArgTypes

	aSrcQuad = (ctypes.c_double * 8)(*srcQuad)
	aDstQuad = (ctypes.c_double * 8)(*dstQuad)

	status = f(src, inSize, nSrcStep, inROI, aSrcQuad, dst, nDstStep, outROI, aDstQuad, interpolation)
	nppCheckStatus(status)


_libnppi.nppiGetAffineTransform.restype = ctypes.c_int
_libnppi.nppiGetAffineTransform.argtypes = [NppiRect, ctypes.c_void_p, ctypes.c_void_p]
def nppiGetAffineTransform(srcROI, quad):
	inROI = NppiRect(*srcROI)
	aQuad = (ctypes.c_double * 8)(*quad)
	coeffs = (ctypes.c_double * 6)()

	status = _libnppi.nppiGetAffineTransform(inROI, aQuad, coeffs)
	nppCheckStatus(status)

	return list(coeffs)
