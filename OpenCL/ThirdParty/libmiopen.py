import sys, os, ctypes


_version_list = [1]
if sys.platform == "linux":
	_libmiopen_libname_list = ["libMIOpen.so.%s" % v for v in _version_list]
elif sys.platform == "win32":
	_libmiopen_libname_list = [os.path.join(os.path.dirname(__file__), "../Libs/miopen.dll")]
else:
	raise RuntimeError("Unsupported platform for MIOpen")


_libmiopen = None
for _libmiopen_libname in _libmiopen_libname_list:
	try:
		if sys.platform == "win32":
			_libmiopen = ctypes.windll.LoadLibrary(_libmiopen_libname)
		else:
			_libmiopen = ctypes.cdll.LoadLibrary(_libmiopen_libname)
	except OSError:
		pass
	else:
		break
if _libmiopen is None:
	raise OSError("MIOpen library not found (searched for following version(s): %s)" % _version_list)


class miopenError(Exception):
	pass

class miopenStatusNotInitialized(miopenError):
	pass

class miopenStatusInvalidValue(miopenError):
	pass

class miopenStatusBadParam(miopenError):
	pass

class miopenStatusAllocFailed(miopenError):
	pass

class miopenStatusInternalError(miopenError):
	pass

class miopenStatusNotImplemented(miopenError):
	pass

class miopenStatusUnknownError(miopenError):
	pass


miopenExceptions = {
	1: miopenStatusNotInitialized,
	2: miopenStatusInvalidValue,
	3: miopenStatusBadParam,
	4: miopenStatusAllocFailed,
	5: miopenStatusInternalError,
	6: miopenStatusNotImplemented,
	7: miopenStatusUnknownError
}


miopenDataType = {
	"miopenHalf": 0,
	"miopenFloat": 1
}

miopenTensorOp = {
	"miopenTensorOpAdd": 0,
	"miopenTensorOpMul": 1,
	"miopenTensorOpMin": 2,
	"miopenTensorOpMax": 3
}

miopenConvolutionMode = {
	"miopenConvolution": 0,
	"miopenTranspose": 1,
	"miopenGroupConv": 2,
	"miopenDepthWise": 3
}

miopenPaddingMode = {
	"miopenPaddingDefault": 0,
	"miopenPaddingSame": 1,
	"miopenPaddingValid": 2
}

miopenPoolingMode = {
	"miopenPoolingMax": 0,
	"miopenPoolingAverage": 1
}

miopenLRNMode = {
	"miopenLRNWithinChannel": 0,
	"miopenLRNCrossChannel": 1
}

miopenBatchNormMode = {
	"miopenBNPerActivation": 0,
	"miopenBNSpatial": 1
}

miopenActivationMode = {
	"miopenActivationPASTHRU": 0,
	"miopenActivationLOGISTIC": 1,
	"miopenActivationTANH": 2,
	"miopenActivationRELU": 3,
	"miopenActivationSOFTRELU": 4,
	"miopenActivationABS": 5,
	"miopenActivationPOWER": 6,
	"miopenActivationCLIPPEDRELU": 7,
	"miopenActivationLEAKYRELU": 8,
	"miopenActivationELU": 9
}

miopenConvFwdAlgorithm = {
	"miopenConvolutionFwdAlgoGEMM": 0,
	"miopenConvolutionFwdAlgoDirect": 1,
	"miopenConvolutionFwdAlgoFFT": 2,
	"miopenConvolutionFwdAlgoWinograd": 3,
	"miopenConvolutionFwdAlgoImplicitGEMM": 5,
	"miopenConvolutionFwdAlgoStaticCompiledGEMM": 6
}

miopenConvBwdWeightsAlgorithm = {
	"miopenConvolutionBwdWeightsAlgoGEMM": 0,
	"miopenConvolutionBwdWeightsAlgoDirect": 1,
	"miopenConvolutionBwdWeightsAlgoWinograd": 3,
	"miopenConvolutionBwdWeightsAlgoImplicitGEMM": 5
}

miopenConvBwdDataAlgorithm = {
	"miopenConvolutionBwdDataAlgoGEMM": 0,
	"miopenConvolutionBwdDataAlgoDirect": 1,
	"miopenConvolutionBwdDataAlgoFFT": 2,
	"miopenConvolutionBwdDataAlgoWinograd": 3,
	"miopenTransposeBwdDataAlgoGEMM": 4,
	"miopenConvolutionBwdDataAlgoImplicitGEMM": 5
}

class miopenConvAlgoPerf(ctypes.Structure):
	_fields_ = [
		("algo", ctypes.c_int),
		("time", ctypes.c_float),
		("memory", ctypes.c_size_t)
	]

miopenRNNMode = {
	"miopenRNNRELU": 0,
	"miopenRNNTANH": 1,
	"miopenLSTM": 2,
	"miopenGRU": 3
}

miopenRNNInputMode = {
	"miopenRNNlinear": 0,
	"miopenRNNskip": 1
}

miopenRNNAlgo = {
	"miopenRNNdefault": 0
}

miopenRNNDirectionMode = {
	"miopenRNNunidirection": 0,
	"miopenRNNbidirection": 1
}

miopenRNNBiasMode = {
	"miopenRNNNoBias": 0,
	"miopenRNNwithBias": 1
}

miopenRNNGEMMalgoMode = {
	"miopenRNNAlgoGEMM": 0
}


def miopenCheckStatus(status):
	if status != 0:
		try:
			raise miopenExceptions[status]
		except KeyError:
			raise miopenError


_libmiopen.miopenCreate.restype = int
_libmiopen.miopenCreate.argtypes = [ctypes.c_void_p]
def miopenCreate():
	handle = ctypes.c_void_p()
	status = _libmiopen.miopenCreate(ctypes.byref(handle))
	miopenCheckStatus(status)

	return handle.value


_libmiopen.miopenCreateWithStream.restype = int
_libmiopen.miopenCreateWithStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def miopenCreateWithStream(stream):
	handle = ctypes.c_void_p()
	status = _libmiopen.miopenCreateWithStream(ctypes.byref(handle), stream)
	miopenCheckStatus(status)

	return handle.value


_libmiopen.miopenDestroy.restype = int
_libmiopen.miopenDestroy.argtypes = [ctypes.c_void_p]
def miopenDestroy(handle):
	status = _libmiopen.miopenDestroy(ctypes.c_void_p(handle))
	miopenCheckStatus(status)


_libmiopen.miopenSetStream.restype = int
_libmiopen.miopenSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def miopenSetStream(handle, streamId):
	status = _libmiopen.miopenSetStream(handle, streamId)
	miopenCheckStatus(status)


_libmiopen.miopenGetStream.restype = int
_libmiopen.miopenGetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def miopenGetStream(handle):
	streamId = ctypes.c_void_p()
	status = _libmiopen.miopenGetStream(handle, ctypes.byref(streamId))
	miopenCheckStatus(status)

	return streamId.value


_libmiopen.miopenGetKernelTime.restype = int
_libmiopen.miopenGetKernelTime.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def miopenGetKernelTime(handle):
	time = ctypes.c_float()
	status = _libmiopen.miopenGetKernelTime(handle, ctypes.byref(time))
	miopenCheckStatus(status)

	return time.value


_libmiopen.miopenEnableProfiling.restype = int
_libmiopen.miopenEnableProfiling.argtypes = [ctypes.c_void_p, ctypes.c_bool]
def miopenEnableProfiling(handle, enable):
	status = _libmiopen.miopenEnableProfiling(handle, enable)
	miopenCheckStatus(status)


_libmiopen.miopenCreateTensorDescriptor.restype = int
_libmiopen.miopenCreateTensorDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateTensorDescriptor():
	tensorDesc = ctypes.c_void_p()
	status = _libmiopen.miopenCreateTensorDescriptor(ctypes.byref(tensorDesc))
	miopenCheckStatus(status)

	return tensorDesc.value


_libmiopen.miopenSet4dTensorDescriptor.restype = int
_libmiopen.miopenSet4dTensorDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int,
												   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
def miopenSet4dTensorDescriptor(tensorDesc, dataType, n, c, h, w):
	status = _libmiopen.miopenSet4dTensorDescriptor(tensorDesc, dataType, n, c, h, w)
	miopenCheckStatus(status)


_libmiopen.miopenGet4dTensorDescriptor.restype = int
_libmiopen.miopenGet4dTensorDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												   ctypes.c_void_p]
def miopenGet4dTensorDescriptor(tensorDesc):
	dataType = ctypes.c_int()
	n, c, h, w = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
	nStride, cStride, hStride, wStride = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()

	status = _libmiopen.miopenGet4dTensorDescriptor(tensorDesc, ctypes.byref(dataType), ctypes.byref(n),
													ctypes.byref(c), ctypes.byref(h), ctypes.byref(w),
													ctypes.byref(nStride), ctypes.byref(cStride),
													ctypes.byref(hStride), ctypes.byref(wStride))
	miopenCheckStatus(status)

	return dataType.value, n.value, c.value, h.value, w.value, \
		   nStride.value, cStride.value, hStride.value, wStride.value


_libmiopen.miopenSetTensorDescriptor.restype = int
_libmiopen.miopenSetTensorDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
												 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
def miopenSetTensorDescriptor(tensorDesc, dataType, dims, strides):
	nbDims = len(dims)
	dimA = (ctypes.c_int * nbDims)(*dims)
	strideA = (ctypes.c_int * nbDims)(*strides)

	status = _libmiopen.miopenSetTensorDescriptor(tensorDesc, dataType, nbDims, dimA, strideA)
	miopenCheckStatus(status)


_libmiopen.miopenGetTensorDescriptorSize.restype = int
_libmiopen.miopenGetTensorDescriptorSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
def miopenGetTensorDescriptorSize(tensorDesc):
	size = ctypes.c_int()

	status = _libmiopen.miopenGetTensorDescriptorSize(tensorDesc, ctypes.byref(size))
	miopenCheckStatus(status)

	return size.value


_libmiopen.miopenGetTensorDescriptor.restype = int
_libmiopen.miopenGetTensorDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
												 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
def miopenGetTensorDescriptor(tensorDesc):
	size = miopenGetTensorDescriptorSize(tensorDesc)

	dataType = ctypes.c_int()
	dimA = (ctypes.c_int * size)()
	strideA = (ctypes.c_int * size)()

	status = _libmiopen.miopenGetTensorDescriptor(tensorDesc, ctypes.byref(dataType), dimA, strideA)
	miopenCheckStatus(status)

	return dataType.value, dimA[:], strideA[:]


_libmiopen.miopenDestroyTensorDescriptor.restype = int
_libmiopen.miopenDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyTensorDescriptor(tensorDesc):
	status = _libmiopen.miopenDestroyTensorDescriptor(tensorDesc)
	miopenCheckStatus(status)


_libmiopen.miopenOpTensor.restype = int
_libmiopen.miopenOpTensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
									  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
									  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenOpTensor(handle, tensorOp, alpha1, aDesc, A, alpha2, bDesc, B, beta, cDesc, C):
	alpha1Ref = ctypes.byref(ctypes.c_float(alpha1))
	alpha2Ref = ctypes.byref(ctypes.c_float(alpha2))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenOpTensor(handle, tensorOp, alpha1Ref, aDesc, A, alpha2Ref, bDesc, B, betaRef, cDesc, C)
	miopenCheckStatus(status)


_libmiopen.miopenSetTensor.restype = int
_libmiopen.miopenSetTensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenSetTensor(handle, yDesc, y, value):
	status = _libmiopen.miopenSetTensor(handle, yDesc, y, ctypes.byref(ctypes.c_float(value)))
	miopenCheckStatus(status)


_libmiopen.miopenScaleTensor.restype = int
_libmiopen.miopenScaleTensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenScaleTensor(handle, yDesc, y, alpha):
	status = _libmiopen.miopenScaleTensor(handle, yDesc, y, ctypes.byref(ctypes.c_float(alpha)))
	miopenCheckStatus(status)


_libmiopen.miopenGetTensorNumBytes.restype = int
_libmiopen.miopenGetTensorNumBytes.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def miopenGetTensorNumBytes(tensorDesc):
	numBytes = ctypes.c_size_t()

	status = _libmiopen.miopenGetTensorSizeInBytes(tensorDesc, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenCreateConvolutionDescriptor.restype = int
_libmiopen.miopenCreateConvolutionDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateConvolutionDescriptor():
	convDesc = ctypes.c_void_p()

	status = _libmiopen.miopenCreateConvolutionDescriptor(ctypes.byref(convDesc))
	miopenCheckStatus(status)

	return convDesc


_libmiopen.miopenInitConvolutionDescriptor.restype = int
_libmiopen.miopenInitConvolutionDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
													   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
def miopenInitConvolutionDescriptor(convDesc, mode, pad_h, pad_w, u, v, dilation_h, dilation_w):
	status = _libmiopen.miopenInitConvolutionDescriptor(convDesc, mode, pad_h, pad_w, u, v, dilation_h, dilation_w)
	miopenCheckStatus(status)


_libmiopen.miopenGetConvolutionDescriptor.restype = int
_libmiopen.miopenGetConvolutionDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
													  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
													  ctypes.c_void_p, ctypes.c_void_p]
def miopenGetConvolutionDescriptor(convDesc):
	pad_h, pad_w = ctypes.c_int(), ctypes.c_int()
	u, v = ctypes.c_int(), ctypes.c_int()
	dilation_h, dilation_w = ctypes.c_int(), ctypes.c_int()
	mode = ctypes.c_int()

	status = _libmiopen.miopenGetConvolutionDescriptor(convDesc, ctypes.byref(mode), ctypes.byref(pad_h),
													   ctypes.byref(pad_w), ctypes.byref(u), ctypes.byref(v),
													   ctypes.byref(dilation_h), ctypes.byref(dilation_w))
	miopenCheckStatus(status)

	return pad_h.value, pad_w.value, u.value, v.value, dilation_h.value, dilation_w.value, mode.value


_libmiopen.miopenGetConvolutionForwardOutputDim.restype = int
_libmiopen.miopenGetConvolutionForwardOutputDim.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
															ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
															ctypes.c_void_p]
def miopenGetConvolutionForwardOutputDim(convDesc, inputTensorDesc, filterDesc):
	n, c, h, w = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()

	status = _libmiopen.miopenGetConvolutionForwardOutputDim(convDesc, inputTensorDesc, filterDesc, ctypes.byref(n),
															 ctypes.byref(c), ctypes.byref(h), ctypes.byref(w))
	miopenCheckStatus(status)

	return n.value, c.value, h.value, w.value


_libmiopen.miopenDestroyConvolutionDescriptor.restype = int
_libmiopen.miopenDestroyConvolutionDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyConvolutionDescriptor(convDesc):
	status = _libmiopen.miopenDestroyConvolutionDescriptor(convDesc)
	miopenCheckStatus(status)


_libmiopen.miopenConvolutionForwardGetWorkSpaceSize.restype = int
_libmiopen.miopenConvolutionForwardGetWorkSpaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
																ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc, convDesc, yDesc):
	workspaceSize = ctypes.c_size_t()

	status = _libmiopen.miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc, convDesc, yDesc,
																 ctypes.byref(workspaceSize))
	miopenCheckStatus(status)

	return workspaceSize.value


_libmiopen.miopenFindConvolutionForwardAlgorithm.restype = int
_libmiopen.miopenFindConvolutionForwardAlgorithm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
															 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
															 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
															 ctypes.c_void_p, ctypes.POINTER(miopenConvAlgoPerf),
															 ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool]
def miopenFindConvolutionForwardAlgorithm(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestAlgoCount,
										  workSpace, workSpaceSize, exhaustiveSearch):
	returnedAlgoCount = ctypes.c_int()
	perfResults = (miopenConvAlgoPerf * requestAlgoCount)()

	status = _libmiopen.miopenFindConvolutionForwardAlgorithm(handle, xDesc, x, wDesc, w, convDesc, yDesc, y,
															  requestAlgoCount, ctypes.byref(returnedAlgoCount),
															  perfResults, workSpace, workSpaceSize, exhaustiveSearch)
	miopenCheckStatus(status)

	return perfResults[:returnedAlgoCount.value]


_libmiopen.miopenConvolutionForward.restype = int
_libmiopen.miopenConvolutionForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
												ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												ctypes.c_size_t]
def miopenConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, beta, yDesc, y, workSpace,
							 workSpaceSize):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenConvolutionForward(handle, alphaRef, xDesc, x, wDesc, w, convDesc, algo, betaRef, yDesc,
												 y, workSpace, workSpaceSize)
	miopenCheckStatus(status)


_libmiopen.miopenConvolutionForwardBias.restype = int
_libmiopen.miopenConvolutionForwardBias.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
													ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenConvolutionForwardBias(handle, alpha, bDesc, b, beta, yDesc, y):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenConvolutionForwardBias(handle, alphaRef, bDesc, b, betaRef, yDesc, y)
	miopenCheckStatus(status)


_libmiopen.miopenConvolutionBackwardDataGetWorkSpaceSize.restype = int
_libmiopen.miopenConvolutionBackwardDataGetWorkSpaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
																	 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dyDesc, wDesc, convDesc, dxDesc):
	workspaceSize = ctypes.c_size_t()

	status = _libmiopen.miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dyDesc, wDesc, convDesc, dxDesc,
																	  ctypes.byref(workspaceSize))
	miopenCheckStatus(status)

	return workspaceSize.value


_libmiopen.miopenFindConvolutionBackwardDataAlgorithm.restype = int
_libmiopen.miopenFindConvolutionBackwardDataAlgorithm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
																  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
																  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
																  ctypes.c_void_p, ctypes.POINTER(miopenConvAlgoPerf),
																  ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool]
def miopenFindConvolutionBackwardDataAlgorithm(handle, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx, requestAlgoCount,
											   workSpace, workSpaceSize, exhaustiveSearch):
	returnedAlgoCount = ctypes.c_int()
	perfResults = (miopenConvAlgoPerf * requestAlgoCount)()

	status = _libmiopen.miopenFindConvolutionBackwardDataAlgorithm(handle, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx,
																   requestAlgoCount, ctypes.byref(returnedAlgoCount),
																   perfResults, workSpace, workSpaceSize,
																   exhaustiveSearch)
	miopenCheckStatus(status)

	return perfResults[:returnedAlgoCount.value]


_libmiopen.miopenConvolutionBackwardData.restype = int
_libmiopen.miopenConvolutionBackwardData.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
													 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
													 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
													 ctypes.c_size_t]
def miopenConvolutionBackwardData(handle, alpha, dyDesc, dy, wDesc, w, convDesc, algo, beta, dxDesc, dx, workSpace,
								  workSpaceSize):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenConvolutionBackwardData(handle, alphaRef, dyDesc, dy, wDesc, w, convDesc, algo, betaRef,
													  dxDesc, dx, workSpace, workSpaceSize)
	miopenCheckStatus(status)


_libmiopen.miopenConvolutionBackwardWeightsGetWorkSpaceSize.restype = int
_libmiopen.miopenConvolutionBackwardWeightsGetWorkSpaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
																		ctypes.c_void_p, ctypes.c_void_p,
																		ctypes.c_void_p, ctypes.c_void_p]
def miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dyDesc, xDesc, convDesc, dwDesc):
	workspaceSize = ctypes.c_size_t()

	status = _libmiopen.miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dyDesc, xDesc, convDesc, dwDesc,
																		 ctypes.byref(workspaceSize))
	miopenCheckStatus(status)

	return workspaceSize.value


_libmiopen.miopenFindConvolutionBackwardWeightsAlgorithm.restype = int
_libmiopen.miopenFindConvolutionBackwardWeightsAlgorithm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
																	 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
																	 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
																	 ctypes.c_void_p,
																	 ctypes.POINTER(miopenConvAlgoPerf),
																	 ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool]
def miopenFindConvolutionBackwardWeightsAlgorithm(handle, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, requestAlgoCount,
												  workSpace, workSpaceSize, exhaustiveSearch):
	returnedAlgoCount = ctypes.c_int()
	perfResults = (miopenConvAlgoPerf * requestAlgoCount)()

	status = _libmiopen.miopenFindConvolutionBackwardWeightsAlgorithm(handle, dyDesc, dy, xDesc, x, convDesc, dwDesc,
																	  dw, requestAlgoCount,
																	  ctypes.byref(returnedAlgoCount), perfResults,
																	  workSpace, workSpaceSize, exhaustiveSearch)
	miopenCheckStatus(status)

	return perfResults[:returnedAlgoCount.value]


_libmiopen.miopenConvolutionBackwardWeights.restype = int
_libmiopen.miopenConvolutionBackwardWeights.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
														ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
														ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
														ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
def miopenConvolutionBackwardWeights(handle, alpha, dyDesc, dy, xDesc, x, convDesc, algo, beta, dwDesc, dw, workSpace,
									 workSpaceSize):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenConvolutionBackwardWeights(handle, alphaRef, dyDesc, dy, xDesc, x, convDesc, algo,
														 betaRef, dwDesc, dw, workSpace, workSpaceSize)
	miopenCheckStatus(status)


_libmiopen.miopenConvolutionBackwardBias.restype = int
_libmiopen.miopenConvolutionBackwardBias.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
													 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenConvolutionBackwardBias(handle, alphaRef, dyDesc, dy, betaRef, dbDesc, db)
	miopenCheckStatus(status)


_libmiopen.miopenCreatePoolingDescriptor.restype = int
_libmiopen.miopenCreatePoolingDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreatePoolingDescriptor():
	poolDesc = ctypes.c_void_p()

	status = _libmiopen.miopenCreatePoolingDescriptor(ctypes.byref(poolDesc))
	miopenCheckStatus(status)

	return poolDesc


_libmiopen.miopenSet2dPoolingDescriptor.restype = int
_libmiopen.miopenSet2dPoolingDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
													ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
def miopenSet2dPoolingDescriptor(poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, u, v):
	status = _libmiopen.miopenSet2dPoolingDescriptor(poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, u, v)
	miopenCheckStatus(status)


_libmiopen.miopenGet2dPoolingDescriptor.restype = int
_libmiopen.miopenGet2dPoolingDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
													ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenGet2dPoolingDescriptor(poolDesc):
	mode = ctypes.c_int()
	windowHeight, windowWidth = ctypes.c_int(), ctypes.c_int()
	pad_h, pad_w = ctypes.c_int(), ctypes.c_int()
	u, v = ctypes.c_int(), ctypes.c_int()

	status = _libmiopen.miopenGet2dPoolingDescriptor(poolDesc, ctypes.byref(mode), ctypes.byref(windowHeight),
													 ctypes.byref(windowWidth), ctypes.byref(pad_h),
													 ctypes.byref(pad_w), ctypes.byref(u), ctypes.byref(v))
	miopenCheckStatus(status)

	return mode.value, windowHeight.value, windowWidth.value, pad_h.value, pad_w.value, u.value, v.value


_libmiopen.miopenGetPoolingForwardOutputDim.restype = int
_libmiopen.miopenGetPoolingForwardOutputDim.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
														ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenGetPoolingForwardOutputDim(poolDesc, tensorDesc):
	n, c, h, w = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()

	status = _libmiopen.miopenGetPoolingForwardOutputDim(poolDesc, tensorDesc, ctypes.byref(n), ctypes.byref(c),
														 ctypes.byref(h), ctypes.byref(w))
	miopenCheckStatus(status)

	return n.value, c.value, h.value, w.value


_libmiopen.miopenPoolingGetWorkSpaceSize.restype = int
_libmiopen.miopenPoolingGetWorkSpaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def miopenPoolingGetWorkSpaceSize(yDesc):
	workSpaceSize = ctypes.c_size_t()

	status = _libmiopen.miopenPoolingGetWorkSpaceSize(yDesc, ctypes.byref(workSpaceSize))
	miopenCheckStatus(status)

	return workSpaceSize.value


_libmiopen.miopenPoolingForward.restype = int
_libmiopen.miopenPoolingForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											ctypes.c_bool, ctypes.c_void_p, ctypes.c_size_t]
def miopenPoolingForward(handle, poolDesc, alpha, xDesc, x, beta, yDesc, y, do_backward, workSpace, workSpaceSize):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenPoolingForward(handle, poolDesc, alphaRef, xDesc, x, betaRef, yDesc, y, do_backward,
											 workSpace, workSpaceSize)
	miopenCheckStatus(status)


_libmiopen.miopenPoolingBackward.restype = int
_libmiopen.miopenPoolingBackward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											 ctypes.c_void_p]
def miopenPoolingBackward(handle, poolDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenPoolingBackward(handle, poolDesc, alphaRef, yDesc, y, dyDesc, dy, xDesc, x, betaRef,
											  dxDesc, dx, workSpace)
	miopenCheckStatus(status)


_libmiopen.miopenDestroyPoolingDescriptor.restype = int
_libmiopen.miopenDestroyPoolingDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyPoolingDescriptor(poolDesc):
	status = _libmiopen.miopenDestroyPoolingDescriptor(poolDesc)
	miopenCheckStatus(status)


_libmiopen.miopenCreateLRNDescriptor.restype = int
_libmiopen.miopenCreateLRNDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateLRNDescriptor():
	lrnDesc = ctypes.c_void_p()

	status = _libmiopen.miopenCreateLRNDescriptor(ctypes.byref(lrnDesc))
	miopenCheckStatus(status)

	return lrnDesc


_libmiopen.miopenSetLRNDescriptor.restype = int
_libmiopen.miopenSetLRNDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_double,
											  ctypes.c_double, ctypes.c_double]
def miopenSetLRNDescriptor(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK):
	status = _libmiopen.miopenSetLRNDescriptor(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK)
	miopenCheckStatus(status)


_libmiopen.miopenGetLRNDescriptor.restype = int
_libmiopen.miopenGetLRNDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											  ctypes.c_void_p, ctypes.c_void_p]
def miopenGetLRNDescriptor(lrnDesc):
	mode = ctypes.c_int()
	lrnN = ctypes.c_uint()
	lrnAlpha = ctypes.c_double()
	lrnBeta = ctypes.c_double()
	lrnK = ctypes.c_double()

	status = _libmiopen.miopenGetLRNDescriptor(lrnDesc, ctypes.byref(mode), ctypes.byref(lrnN), ctypes.byref(lrnAlpha),
											   ctypes.byref(lrnBeta), ctypes.byref(lrnK))
	miopenCheckStatus(status)

	return mode.value, lrnN.value, lrnAlpha.value, lrnBeta.value, lrnK.value


_libmiopen.miopenLRNGetWorkSpaceSize.restype = int
_libmiopen.miopenLRNGetWorkSpaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def miopenLRNGetWorkSpaceSize(yDesc):
	workSpaceSize = ctypes.c_size_t()

	status = _libmiopen.miopenLRNGetWorkSpaceSize(yDesc, ctypes.byref(workSpaceSize))
	miopenCheckStatus(status)

	return workSpaceSize.value


_libmiopen.miopenLRNForward.restype = int
_libmiopen.miopenLRNForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
										ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
										ctypes.c_bool, ctypes.c_void_p]
def miopenLRNForward(handle, lrnDesc, alpha, xDesc, x, beta, yDesc, y, do_backward, workSpace):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenLRNForward(handle, lrnDesc, alphaRef, xDesc, x, betaRef, yDesc, y, do_backward, workSpace)
	miopenCheckStatus(status)


_libmiopen.miopenLRNBackward.restype = int
_libmiopen.miopenLRNBackward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
										 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
										 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
										 ctypes.c_void_p]
def miopenLRNBackward(handle, lrnDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenLRNBackward(handle, lrnDesc, alphaRef, yDesc, y, dyDesc, dy, xDesc, x, betaRef, dxDesc,
										  dx, workSpace)
	miopenCheckStatus(status)


_libmiopen.miopenDestroyLRNDescriptor.restype = int
_libmiopen.miopenDestroyLRNDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyLRNDescriptor(lrnDesc):
	status = _libmiopen.miopenDestroyLRNDescriptor(lrnDesc)
	miopenCheckStatus(status)


_libmiopen.miopenDeriveBNTensorDescriptor.restype = int
_libmiopen.miopenDeriveBNTensorDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
def miopenDeriveBNTensorDescriptor(derivedBnDesc, xDesc, bn_mode):
	status = _libmiopen.miopenDeriveBNTensorDescriptor(derivedBnDesc, xDesc, bn_mode)
	miopenCheckStatus(status)


_libmiopen.miopenBatchNormalizationForwardTraining.restype = int
_libmiopen.miopenBatchNormalizationForwardTraining.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
															   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
															   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
															   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
															   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
															   ctypes.c_void_p, ctypes.c_void_p]
def miopenBatchNormalizationForwardTraining(handle, bn_mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
											bnScale, bnBias, expAvgFactor, resultRunningMean, resultRunningVariance,
											epsilon, resultSaveMean, resultSaveInvVariance):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenBatchNormalizationForwardTraining(handle, bn_mode, alphaRef, betaRef, xDesc, x, yDesc, y,
																bnScaleBiasMeanVarDesc, bnScale, bnBias, expAvgFactor,
																resultRunningMean, resultRunningVariance, epsilon,
																resultSaveMean, resultSaveInvVariance)
	miopenCheckStatus(status)


_libmiopen.miopenBatchNormalizationForwardInference.restype = int
_libmiopen.miopenBatchNormalizationForwardInference.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
																ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
																ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
																ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
																ctypes.c_void_p, ctypes.c_double]
def miopenBatchNormalizationForwardInference(handle, bn_mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
											 bnScale, bnBias, estimatedMean, estimatedVariance, epsilon):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenBatchNormalizationForwardInference(handle, bn_mode, alphaRef, betaRef, xDesc, x, yDesc, y,
																 bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean,
																 estimatedVariance, epsilon)
	miopenCheckStatus(status)


_libmiopen.miopenBatchNormalizationBackward.restype = int
_libmiopen.miopenBatchNormalizationBackward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
														ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
														ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
														ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
														ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
														ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
def miopenBatchNormalizationBackward(handle, bn_mode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff,
									 xDesc, x, dyDesc, dy, dxDesc, dx, bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff,
									 resultBnBiasDiff, epsilon, savedMean, savedInvVariance):
	alphaDataRef = ctypes.byref(ctypes.c_float(alphaDataDiff))
	betaDataRef = ctypes.byref(ctypes.c_float(betaDataDiff))

	alphaParamRef = ctypes.byref(ctypes.c_float(alphaParamDiff))
	betaParamRef = ctypes.byref(ctypes.c_float(betaParamDiff))

	status = _libmiopen.miopenBatchNormalizationBackward(handle, bn_mode, alphaDataRef, betaDataRef, alphaParamRef,
														 betaParamRef, xDesc, x, dyDesc, dy, dxDesc, dx,
														 bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff,
														 resultBnBiasDiff, epsilon, savedMean, savedInvVariance)
	miopenCheckStatus(status)


_libmiopen.miopenCreateActivationDescriptor.restype = int
_libmiopen.miopenCreateActivationDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateActivationDescriptor():
	activationDesc = ctypes.c_void_p()

	status = _libmiopen.miopenCreateActivationDescriptor(ctypes.byref(activationDesc))
	miopenCheckStatus(status)

	return activationDesc.value


_libmiopen.miopenSetActivationDescriptor.restype = int
_libmiopen.miopenSetActivationDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double,
													 ctypes.c_double]
def miopenSetActivationDescriptor(activationDesc, mode, alpha, beta, power):
	status = _libmiopen.miopenSetActivationDescriptor(activationDesc, mode, alpha, beta, power)
	miopenCheckStatus(status)


_libmiopen.miopenGetActivationDescriptor.restype = int
_libmiopen.miopenGetActivationDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
													 ctypes.c_void_p]
def miopenGetActivationDescriptor(activationDesc):
	mode = ctypes.c_void_p()
	alpha, beta, power = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()

	status = _libmiopen.miopenGetActivationDescriptor(activationDesc, ctypes.byref(mode), ctypes.byref(alpha),
													  ctypes.byref(beta), ctypes.byref(power))
	miopenCheckStatus(status)

	return mode.value, alpha.value, beta.value, power.value


_libmiopen.miopenActivationForward.restype = int
_libmiopen.miopenActivationForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											   ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenActivationForward(handle, activationDesc, alphaRef, xDesc, x, betaRef, yDesc, y)
	miopenCheckStatus(status)


_libmiopen.miopenActivationBackward.restype = int
_libmiopen.miopenActivationBackward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenActivationBackward(handle, activationDesc, alphaRef, yDesc, y, dyDesc, dy, xDesc, x,
												 betaRef, dxDesc, dx)
	miopenCheckStatus(status)


_libmiopen.miopenDestroyActivationDescriptor.restype = int
_libmiopen.miopenDestroyActivationDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyActivationDescriptor(activationDesc):
	status = _libmiopen.miopenDestroyActivationDescriptor(activationDesc)
	miopenCheckStatus(status)


_libmiopen.miopenSoftmaxForward.restype = int
_libmiopen.miopenSoftmaxForward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenSoftmaxForward(handle, alpha, xDesc, x, beta, yDesc, y):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenSoftmaxForward(handle, alphaRef, xDesc, x, betaRef, yDesc, y)
	miopenCheckStatus(status)


_libmiopen.miopenSoftmaxBackward.restype = int
_libmiopen.miopenSoftmaxBackward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											 ctypes.c_void_p]
def miopenSoftmaxBackward(handle, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx):
	alphaRef = ctypes.byref(ctypes.c_float(alpha))
	betaRef = ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenSoftmaxBackward(handle, alphaRef, yDesc, y, dyDesc, dy, betaRef, dxDesc, dx)
	miopenCheckStatus(status)


_libmiopen.miopenCreateRNNDescriptor.restype = int
_libmiopen.miopenCreateRNNDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateRNNDescriptor():
	rnnDesc = ctypes.c_void_p()

	status = _libmiopen.miopenCreateRNNDescriptor(ctypes.byref(rnnDesc))
	miopenCheckStatus(status)

	return rnnDesc.value


_libmiopen.miopenGetRNNDescriptor.restype = int
_libmiopen.miopenGetRNNDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
def miopenGetRNNDescriptor(rnnDesc):
	rnnMode, algoMode = ctypes.c_int(), ctypes.c_int()
	inputMode, dirMode, biasMode = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
	hiddenSize, layer = ctypes.c_int(), ctypes.c_int()

	status = _libmiopen.miopenGetRNNDescriptor(rnnDesc, ctypes.byref(rnnMode), ctypes.byref(algoMode),
											   ctypes.byref(inputMode), ctypes.byref(dirMode), ctypes.byref(biasMode),
											   ctypes.byref(hiddenSize), ctypes.byref(layer))
	miopenCheckStatus(status)

	return rnnMode.value, algoMode.value, inputMode.value, dirMode.value, biasMode.value, hiddenSize.value, layer.value


_libmiopen.miopenDestroyRNNDescriptor.restype = int
_libmiopen.miopenDestroyRNNDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyRNNDescriptor(rnnDesc):
	status = _libmiopen.miopenDestroyRNNDescriptor(rnnDesc)
	miopenCheckStatus(status)


_libmiopen.miopenSetRNNDescriptor.restype = int
_libmiopen.miopenSetRNNDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
											  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
def miopenSetRNNDescriptor(rnnDesc, hsize, nlayers, inMode, direction, rnnMode, biasMode, algo, dataType):
	status = _libmiopen.miopenSetRNNDescriptor(rnnDesc, hsize, nlayers, inMode, direction, rnnMode, biasMode, algo,
											   dataType)
	miopenCheckStatus(status)


_libmiopen.miopenGetRNNWorkspaceSize.restype = int
_libmiopen.miopenGetRNNWorkspaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
												 ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
def miopenGetRNNWorkspaceSize(handle, rnnDesc, sequenceLen, xDesc):
	numBytes = ctypes.c_size_t()
	xDesc = (ctypes.c_void_p * len(xDesc))(*xDesc)

	status = _libmiopen.miopenGetRNNWorkspaceSize(handle, rnnDesc, sequenceLen, xDesc, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNTrainingReserveSize.restype = int
_libmiopen.miopenGetRNNTrainingReserveSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
													   ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
def miopenGetRNNTrainingReserveSize(handle, rnnDesc, sequenceLen, xDesc):
	numBytes = ctypes.c_size_t()
	xDesc = (ctypes.c_void_p * len(xDesc))(*xDesc)

	status = _libmiopen.miopenGetRNNTrainingReserveSize(handle, rnnDesc, sequenceLen, xDesc, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNParamsSize.restype = int
_libmiopen.miopenGetRNNParamsSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											  ctypes.c_int]
def miopenGetRNNParamsSize(handle, rnnDesc, xDesc, dtype):
	numBytes = ctypes.c_size_t()

	status = _libmiopen.miopenGetRNNParamsSize(handle, rnnDesc, xDesc, ctypes.byref(numBytes), dtype)
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNParamsDescriptor.restype = int
_libmiopen.miopenGetRNNParamsDescriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
													ctypes.c_int]
def miopenGetRNNParamsDescriptor(handle, rnnDesc, xDesc, wDesc, dtype):
	status = _libmiopen.miopenGetRNNParamsDescriptor(handle, rnnDesc, xDesc, wDesc, dtype)
	miopenCheckStatus(status)


_libmiopen.miopenGetRNNInputTensorSize.restype = int
_libmiopen.miopenGetRNNInputTensorSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
												   ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
def miopenGetRNNInputTensorSize(handle, rnnDesc, seqLen, xDesc):
	numBytes = ctypes.c_size_t()
	xDesc = (ctypes.c_void_p * len(xDesc))(*xDesc)

	status = _libmiopen.miopenGetRNNInputTensorSize(handle, rnnDesc, seqLen, xDesc, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNHiddenTensorSize.restype = int
_libmiopen.miopenGetRNNHiddenTensorSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
													ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
def miopenGetRNNHiddenTensorSize(handle, rnnDesc, seqLen, xDesc):
	numBytes = ctypes.c_size_t()
	xDesc = (ctypes.c_void_p * len(xDesc))(*xDesc)

	status = _libmiopen.miopenGetRNNHiddenTensorSize(handle, rnnDesc, seqLen, xDesc, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNLayerParamSize.restype = int
_libmiopen.miopenGetRNNLayerParamSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
												  ctypes.c_int, ctypes.c_void_p]
def miopenGetRNNLayerParamSize(handle, rnnDesc, layer, xDesc, paramID):
	numBytes = ctypes.c_size_t()

	status = _libmiopen.miopenGetRNNLayerParamSize(handle, rnnDesc, layer, xDesc, paramID, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNLayerBiasSize.restype = int
_libmiopen.miopenGetRNNLayerBiasSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
												 ctypes.c_void_p]
def miopenGetRNNLayerBiasSize(handle, rnnDesc, layer, biasID):
	numBytes = ctypes.c_size_t()

	status = _libmiopen.miopenGetRNNLayerBiasSize(handle, rnnDesc, layer, biasID, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNLayerParam.restype = int
_libmiopen.miopenGetRNNLayerParam.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
											  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
											  ctypes.c_void_p]
def miopenGetRNNLayerParam(handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam):
	status = _libmiopen.miopenGetRNNLayerParam(handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam)
	miopenCheckStatus(status)


_libmiopen.miopenGetRNNLayerBias.restype = int
_libmiopen.miopenGetRNNLayerBias.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
											 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
											 ctypes.c_void_p]
def miopenGetRNNLayerBias(handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias):
	status = _libmiopen.miopenGetRNNLayerBias(handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias)
	miopenCheckStatus(status)


_libmiopen.miopenSetRNNLayerParam.restype = int
_libmiopen.miopenSetRNNLayerParam.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
											  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
											  ctypes.c_void_p]
def miopenSetRNNLayerParam(handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam):
	status = _libmiopen.miopenSetRNNLayerParam(handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam)
	miopenCheckStatus(status)


_libmiopen.miopenSetRNNLayerBias.restype = int
_libmiopen.miopenSetRNNLayerBias.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
											 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
											 ctypes.c_void_p]
def miopenSetRNNLayerBias(handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias):
	status = _libmiopen.miopenSetRNNLayerBias(handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias)
	miopenCheckStatus(status)


_libmiopen.miopenRNNForwardTraining.restype = int
_libmiopen.miopenRNNForwardTraining.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
												ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
												ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
												ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
def miopenRNNForwardTraining(handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
							 cyDesc, cy, workSpace, workSpaceNumBytes, reserveSpace, reserveSpaceNumBytes):
	xDesc = (ctypes.c_void_p * len(xDesc))(*xDesc)
	yDesc = (ctypes.c_void_p * len(yDesc))(*yDesc)

	status = _libmiopen.miopenRNNForwardTraining(handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
												 yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceNumBytes,
												 reserveSpace, reserveSpaceNumBytes)
	miopenCheckStatus(status)


_libmiopen.miopenRNNBackwardData.restype = int
_libmiopen.miopenRNNBackwardData.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
											 ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
											 ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
											 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											 ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
											 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
											 ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
def miopenRNNBackwardData(handle, rnnDesc, seqLen, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w,
						  hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceNumBytes,
						  reserveSpace, reserveSpaceNumBytes):
	yDesc = (ctypes.c_void_p * len(yDesc))(*yDesc)
	dyDesc = (ctypes.c_void_p * len(dyDesc))(*dyDesc)
	dxDesc = (ctypes.c_void_p * len(dxDesc))(*dxDesc)

	status = _libmiopen.miopenRNNBackwardData(handle, rnnDesc, seqLen, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy,
											  wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx,
											  workSpace, workSpaceNumBytes, reserveSpace, reserveSpaceNumBytes)
	miopenCheckStatus(status)


_libmiopen.miopenRNNBackwardWeights.restype = int
_libmiopen.miopenRNNBackwardWeights.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
												ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
												ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
												ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
												ctypes.c_void_p, ctypes.c_size_t]
def miopenRNNBackwardWeights(handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, yDesc, y, dwDesc, dw, workSpace,
							 workSpaceNumBytes, reserveSpace, reserveSpaceNumBytes):
	xDesc = (ctypes.c_void_p * len(xDesc))(*xDesc)
	yDesc = (ctypes.c_void_p * len(yDesc))(*yDesc)

	status = _libmiopen.miopenRNNBackwardWeights(handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, yDesc, y, dwDesc, dw,
												 workSpace, workSpaceNumBytes, reserveSpace, reserveSpaceNumBytes)
	miopenCheckStatus(status)


_libmiopen.miopenRNNForwardInference.restype = int
_libmiopen.miopenRNNForwardInference.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
												 ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
												 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												 ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
												 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
												 ctypes.c_void_p, ctypes.c_size_t]
def miopenRNNForwardInference(handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
							  cyDesc, cy, workSpace, workSpaceNumBytes):
	xDesc = (ctypes.c_void_p * len(xDesc))(*xDesc)
	yDesc = (ctypes.c_void_p * len(yDesc))(*yDesc)

	status = _libmiopen.miopenRNNForwardInference(handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w,
												  yDesc, y, hyDesc, hy, cyDesc, cy, workSpace, workSpaceNumBytes)
	miopenCheckStatus(status)
