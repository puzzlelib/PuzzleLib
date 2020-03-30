import sys, ctypes


_version_list = [1]
if sys.platform == "linux":
	_libmiopen_libname_list = [("libMIOpen.so.%s" % v, v) for v in _version_list]
else:
	raise RuntimeError("Unsupported platform for MIOpen")


_libmiopen, version = None, None
for _libmiopen_libname, v in _libmiopen_libname_list:
	try:
		_libmiopen = ctypes.cdll.LoadLibrary(_libmiopen_libname)
		version = v
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

class miopenStatusUnsupportedOp(miopenError):
	pass


miopenExceptions = {
	1: miopenStatusNotInitialized,
	2: miopenStatusInvalidValue,
	3: miopenStatusBadParam,
	4: miopenStatusAllocFailed,
	5: miopenStatusInternalError,
	6: miopenStatusNotImplemented,
	7: miopenStatusUnknownError,
	8: miopenStatusUnsupportedOp
}


miopenDataType = {
	"miopenHalf": 0,
	"miopenFloat": 1
}

miopenConvolutionMode = {
	"miopenConvolution": 0,
	"miopenTranspose": 1,
	"miopenGroupConv": 2,
	"miopenDepthWise": 3
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

miopenPoolingMode = {
	"miopenPoolingMax": 0,
	"miopenPoolingAverage": 1,
	"miopenPoolingAverageInclusive": 2
}

miopenSoftmaxAlgorithm = {
	"MIOPEN_SOFTMAX_FAST": 0,
	"MIOPEN_SOFTMAX_ACCURATE": 1,
	"MIOPEN_SOFTMAX_LOG": 2
}

miopenSoftmaxMode = {
	"MIOPEN_SOFTMAX_MODE_INSTANCE": 0,
	"MIOPEN_SOFTMAX_MODE_CHANNEL": 1
}

miopenBatchNormMode = {
	"miopenBNPerActivation": 0,
	"miopenBNSpatial": 1
}

miopenLRNMode = {
	"miopenLRNWithinChannel": 0,
	"miopenLRNCrossChannel": 1
}

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
	"miopenRNNdefault": 0,
	"miopenRNNfundamental": 1
}

miopenRNNDirectionMode = {
	"miopenRNNunidirection": 0,
	"miopenRNNbidirection": 1
}

miopenRNNBiasMode = {
	"miopenRNNNoBias": 0,
	"miopenRNNwithBias": 1
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


_libmiopen.miopenDestroy.restype = int
_libmiopen.miopenDestroy.argtypes = [ctypes.c_void_p]
def miopenDestroy(handle):
	status = _libmiopen.miopenDestroy(ctypes.c_void_p(handle))
	miopenCheckStatus(status)


_libmiopen.miopenCreateTensorDescriptor.restype = int
_libmiopen.miopenCreateTensorDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateTensorDescriptor():
	tensorDesc = ctypes.c_void_p()
	status = _libmiopen.miopenCreateTensorDescriptor(ctypes.byref(tensorDesc))
	miopenCheckStatus(status)

	return tensorDesc.value


_libmiopen.miopenSetTensorDescriptor.restype = int
_libmiopen.miopenSetTensorDescriptor.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
]
def miopenSetTensorDescriptor(tensorDesc, dataType, dims, strides):
	nbDims = len(dims)
	dimA, strideA = (ctypes.c_int * nbDims)(*dims), (ctypes.c_int * nbDims)(*strides)

	status = _libmiopen.miopenSetTensorDescriptor(tensorDesc, dataType, nbDims, dimA, strideA)
	miopenCheckStatus(status)


_libmiopen.miopenDestroyTensorDescriptor.restype = int
_libmiopen.miopenDestroyTensorDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyTensorDescriptor(tensorDesc):
	status = _libmiopen.miopenDestroyTensorDescriptor(tensorDesc)
	miopenCheckStatus(status)


_libmiopen.miopenCreateConvolutionDescriptor.restype = int
_libmiopen.miopenCreateConvolutionDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateConvolutionDescriptor():
	convDesc = ctypes.c_void_p()

	status = _libmiopen.miopenCreateConvolutionDescriptor(ctypes.byref(convDesc))
	miopenCheckStatus(status)

	return convDesc


_libmiopen.miopenInitConvolutionNdDescriptor.restype = int
_libmiopen.miopenInitConvolutionNdDescriptor.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
	ctypes.POINTER(ctypes.c_int), ctypes.c_int
]
def miopenInitConvolutionNdDescriptor(convDesc, pad, stride, dilation, mode):
	nbDims = len(pad)
	padA, strideA = (ctypes.c_int * nbDims)(*pad), (ctypes.c_int * nbDims)(*stride)
	dilationA = (ctypes.c_int * nbDims)(*dilation)

	status = _libmiopen.miopenInitConvolutionNdDescriptor(convDesc, nbDims, padA, strideA, dilationA, mode)
	miopenCheckStatus(status)


_libmiopen.miopenSetConvolutionGroupCount.restype = int
_libmiopen.miopenSetConvolutionGroupCount.argtypes = [ctypes.c_void_p, ctypes.c_int]
def miopenSetConvolutionGroupCount(convDesc, groupCount):
	status = _libmiopen.miopenSetConvolutionGroupCount(convDesc, groupCount)
	miopenCheckStatus(status)


_libmiopen.miopenGetConvolutionNdForwardOutputDim.restype = int
_libmiopen.miopenGetConvolutionNdForwardOutputDim.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
def miopenGetConvolutionNdForwardOutputDim(convDesc, inputTensorDesc, filterDesc, nbDims):
	nDim = ctypes.c_int()
	tensorOutputDimA = (ctypes.c_int * nbDims)()

	status = _libmiopen.miopenGetConvolutionNdForwardOutputDim(
		convDesc, inputTensorDesc, filterDesc, ctypes.byref(nDim), tensorOutputDimA
	)
	miopenCheckStatus(status)

	return tuple(tensorOutputDimA)


_libmiopen.miopenDestroyConvolutionDescriptor.restype = int
_libmiopen.miopenDestroyConvolutionDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyConvolutionDescriptor(convDesc):
	status = _libmiopen.miopenDestroyConvolutionDescriptor(convDesc)
	miopenCheckStatus(status)


_libmiopen.miopenConvolutionForwardGetWorkSpaceSize.restype = int
_libmiopen.miopenConvolutionForwardGetWorkSpaceSize.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
def miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc, convDesc, yDesc):
	workspaceSize = ctypes.c_size_t()

	status = _libmiopen.miopenConvolutionForwardGetWorkSpaceSize(
		handle, wDesc, xDesc, convDesc, yDesc, ctypes.byref(workspaceSize)
	)
	miopenCheckStatus(status)

	return workspaceSize.value


_libmiopen.miopenFindConvolutionForwardAlgorithm.restype = int
_libmiopen.miopenFindConvolutionForwardAlgorithm.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(miopenConvAlgoPerf),
	ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool
]
def miopenFindConvolutionForwardAlgorithm(handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestAlgoCount,
										  workSpace, workSpaceSize, exhaustiveSearch):
	returnedAlgoCount = ctypes.c_int()
	perfResults = (miopenConvAlgoPerf * requestAlgoCount)()

	status = _libmiopen.miopenFindConvolutionForwardAlgorithm(
		handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestAlgoCount, ctypes.byref(returnedAlgoCount),
		perfResults, workSpace, workSpaceSize, exhaustiveSearch
	)
	miopenCheckStatus(status)

	return perfResults[:returnedAlgoCount.value]


_libmiopen.miopenConvolutionForward.restype = int
_libmiopen.miopenConvolutionForward.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
]
def miopenConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, beta, yDesc, y, workSpace,
							 workSpaceSize):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenConvolutionForward(
		handle, alphaRef, xDesc, x, wDesc, w, convDesc, algo, betaRef, yDesc, y, workSpace, workSpaceSize
	)
	miopenCheckStatus(status)


_libmiopen.miopenConvolutionForwardBias.restype = int
_libmiopen.miopenConvolutionForwardBias.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
def miopenConvolutionForwardBias(handle, alpha, bDesc, b, beta, yDesc, y):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenConvolutionForwardBias(handle, alphaRef, bDesc, b, betaRef, yDesc, y)
	miopenCheckStatus(status)


_libmiopen.miopenConvolutionBackwardDataGetWorkSpaceSize.restype = int
_libmiopen.miopenConvolutionBackwardDataGetWorkSpaceSize.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
def miopenConvolutionBackwardDataGetWorkSpaceSize(handle, dyDesc, wDesc, convDesc, dxDesc):
	workspaceSize = ctypes.c_size_t()

	status = _libmiopen.miopenConvolutionBackwardDataGetWorkSpaceSize(
		handle, dyDesc, wDesc, convDesc, dxDesc, ctypes.byref(workspaceSize)
	)
	miopenCheckStatus(status)

	return workspaceSize.value


_libmiopen.miopenFindConvolutionBackwardDataAlgorithm.restype = int
_libmiopen.miopenFindConvolutionBackwardDataAlgorithm.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(miopenConvAlgoPerf),
	ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool
]
def miopenFindConvolutionBackwardDataAlgorithm(handle, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx, requestAlgoCount,
											   workSpace, workSpaceSize, exhaustiveSearch):
	returnedAlgoCount = ctypes.c_int()
	perfResults = (miopenConvAlgoPerf * requestAlgoCount)()

	status = _libmiopen.miopenFindConvolutionBackwardDataAlgorithm(
		handle, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx, requestAlgoCount, ctypes.byref(returnedAlgoCount),
		perfResults, workSpace, workSpaceSize, exhaustiveSearch
	)
	miopenCheckStatus(status)

	return perfResults[:returnedAlgoCount.value]


_libmiopen.miopenConvolutionBackwardData.restype = int
_libmiopen.miopenConvolutionBackwardData.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
]
def miopenConvolutionBackwardData(handle, alpha, dyDesc, dy, wDesc, w, convDesc, algo, beta, dxDesc, dx, workSpace,
								  workSpaceSize):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenConvolutionBackwardData(
		handle, alphaRef, dyDesc, dy, wDesc, w, convDesc, algo, betaRef, dxDesc, dx, workSpace, workSpaceSize
	)
	miopenCheckStatus(status)


_libmiopen.miopenConvolutionBackwardWeightsGetWorkSpaceSize.restype = int
_libmiopen.miopenConvolutionBackwardWeightsGetWorkSpaceSize.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
def miopenConvolutionBackwardWeightsGetWorkSpaceSize(handle, dyDesc, xDesc, convDesc, dwDesc):
	workspaceSize = ctypes.c_size_t()

	status = _libmiopen.miopenConvolutionBackwardWeightsGetWorkSpaceSize(
		handle, dyDesc, xDesc, convDesc, dwDesc, ctypes.byref(workspaceSize)
	)
	miopenCheckStatus(status)

	return workspaceSize.value


_libmiopen.miopenFindConvolutionBackwardWeightsAlgorithm.restype = int
_libmiopen.miopenFindConvolutionBackwardWeightsAlgorithm.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(miopenConvAlgoPerf),
	ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool
]
def miopenFindConvolutionBackwardWeightsAlgorithm(handle, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, requestAlgoCount,
												  workSpace, workSpaceSize, exhaustiveSearch):
	returnedAlgoCount = ctypes.c_int()
	perfResults = (miopenConvAlgoPerf * requestAlgoCount)()

	status = _libmiopen.miopenFindConvolutionBackwardWeightsAlgorithm(
		handle, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, requestAlgoCount, ctypes.byref(returnedAlgoCount),
		perfResults, workSpace, workSpaceSize, exhaustiveSearch
	)
	miopenCheckStatus(status)

	return perfResults[:returnedAlgoCount.value]


_libmiopen.miopenConvolutionBackwardWeights.restype = int
_libmiopen.miopenConvolutionBackwardWeights.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
]
def miopenConvolutionBackwardWeights(handle, alpha, dyDesc, dy, xDesc, x, convDesc, algo, beta, dwDesc, dw, workSpace,
									 workSpaceSize):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenConvolutionBackwardWeights(
		handle, alphaRef, dyDesc, dy, xDesc, x, convDesc, algo, betaRef, dwDesc, dw, workSpace, workSpaceSize
	)
	miopenCheckStatus(status)


_libmiopen.miopenConvolutionBackwardBias.restype = int
_libmiopen.miopenConvolutionBackwardBias.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
]
def miopenConvolutionBackwardBias(handle, alpha, dyDesc, dy, beta, dbDesc, db):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

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
_libmiopen.miopenSet2dPoolingDescriptor.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
def miopenSet2dPoolingDescriptor(poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, u, v):
	status = _libmiopen.miopenSet2dPoolingDescriptor(poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, u, v)
	miopenCheckStatus(status)


_libmiopen.miopenPoolingGetWorkSpaceSize.restype = int
_libmiopen.miopenPoolingGetWorkSpaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def miopenPoolingGetWorkSpaceSize(yDesc):
	workSpaceSize = ctypes.c_size_t()

	status = _libmiopen.miopenPoolingGetWorkSpaceSize(yDesc, ctypes.byref(workSpaceSize))
	miopenCheckStatus(status)

	return workSpaceSize.value


_libmiopen.miopenPoolingForward.restype = int
_libmiopen.miopenPoolingForward.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p, ctypes.c_size_t
]
def miopenPoolingForward(handle, poolDesc, alpha, xDesc, x, beta, yDesc, y, do_backward, workSpace, workSpaceSize):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenPoolingForward(
		handle, poolDesc, alphaRef, xDesc, x, betaRef, yDesc, y, do_backward, workSpace, workSpaceSize
	)
	miopenCheckStatus(status)


_libmiopen.miopenPoolingBackward.restype = int
_libmiopen.miopenPoolingBackward.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p
]
def miopenPoolingBackward(handle, poolDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenPoolingBackward(
		handle, poolDesc, alphaRef, yDesc, y, dyDesc, dy, xDesc, x, betaRef, dxDesc, dx, workSpace
	)
	miopenCheckStatus(status)


_libmiopen.miopenDestroyPoolingDescriptor.restype = int
_libmiopen.miopenDestroyPoolingDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyPoolingDescriptor(poolDesc):
	status = _libmiopen.miopenDestroyPoolingDescriptor(poolDesc)
	miopenCheckStatus(status)


_libmiopen.miopenSoftmaxForward_V2.restype = int
_libmiopen.miopenSoftmaxForward_V2.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_int, ctypes.c_int
]
def miopenSoftmaxForward(handle, alpha, xDesc, x, beta, yDesc, y, algorithm, mode):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenSoftmaxForward_V2(handle, alphaRef, xDesc, x, betaRef, yDesc, y, algorithm, mode)
	miopenCheckStatus(status)


_libmiopen.miopenSoftmaxBackward_V2.restype = int
_libmiopen.miopenSoftmaxBackward_V2.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
]
def miopenSoftmaxBackward(handle, alpha, yDesc, y, dyDesc, dy, beta, dxDesc, dx, algorithm, mode):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenSoftmaxBackward_V2(
		handle, alphaRef, yDesc, y, dyDesc, dy, betaRef, dxDesc, dx, algorithm, mode
	)
	miopenCheckStatus(status)


_libmiopen.miopenBatchNormalizationForwardTraining.restype = int
_libmiopen.miopenBatchNormalizationForwardTraining.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p
]
def miopenBatchNormalizationForwardTraining(handle, bn_mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
											bnScale, bnBias, expAvgFactor, resultRunningMean, resultRunningVariance,
											epsilon, resultSaveMean, resultSaveInvVariance):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenBatchNormalizationForwardTraining(
		handle, bn_mode, alphaRef, betaRef, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, expAvgFactor,
		resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance
	)
	miopenCheckStatus(status)


_libmiopen.miopenBatchNormalizationForwardInference.restype = int
_libmiopen.miopenBatchNormalizationForwardInference.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_double
]
def miopenBatchNormalizationForwardInference(handle, bn_mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc,
											 bnScale, bnBias, estimatedMean, estimatedVariance, epsilon):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenBatchNormalizationForwardInference(
		handle, bn_mode, alphaRef, betaRef, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean,
		estimatedVariance, epsilon
	)
	miopenCheckStatus(status)


_libmiopen.miopenBatchNormalizationBackward.restype = int
_libmiopen.miopenBatchNormalizationBackward.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p
]
def miopenBatchNormalizationBackward(handle, bn_mode, alphaDataDiff, betaDataDiff, alphaParmDiff, betaParmDiff,
									 xDesc, x, dyDesc, dy, dxDesc, dx, bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff,
									 resultBnBiasDiff, epsilon, savedMean, savedInvVariance):
	alphaDataRef, betaDataRef = ctypes.byref(ctypes.c_float(alphaDataDiff)), ctypes.byref(ctypes.c_float(betaDataDiff))
	alphaParmRef, betaParmRef = ctypes.byref(ctypes.c_float(alphaParmDiff)), ctypes.byref(ctypes.c_float(betaParmDiff))

	status = _libmiopen.miopenBatchNormalizationBackward(
		handle, bn_mode, alphaDataRef, betaDataRef, alphaParmRef, betaParmRef, xDesc, x, dyDesc, dy, dxDesc, dx,
		bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean, savedInvVariance
	)
	miopenCheckStatus(status)


_libmiopen.miopenCreateLRNDescriptor.restype = int
_libmiopen.miopenCreateLRNDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateLRNDescriptor():
	lrnDesc = ctypes.c_void_p()

	status = _libmiopen.miopenCreateLRNDescriptor(ctypes.byref(lrnDesc))
	miopenCheckStatus(status)

	return lrnDesc


_libmiopen.miopenSetLRNDescriptor.restype = int
_libmiopen.miopenSetLRNDescriptor.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_double, ctypes.c_double, ctypes.c_double
]
def miopenSetLRNDescriptor(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK):
	status = _libmiopen.miopenSetLRNDescriptor(lrnDesc, mode, lrnN, lrnAlpha, lrnBeta, lrnK)
	miopenCheckStatus(status)


_libmiopen.miopenLRNGetWorkSpaceSize.restype = int
_libmiopen.miopenLRNGetWorkSpaceSize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
def miopenLRNGetWorkSpaceSize(yDesc):
	workSpaceSize = ctypes.c_size_t()

	status = _libmiopen.miopenLRNGetWorkSpaceSize(yDesc, ctypes.byref(workSpaceSize))
	miopenCheckStatus(status)

	return workSpaceSize.value


_libmiopen.miopenLRNForward.restype = int
_libmiopen.miopenLRNForward.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p
]
def miopenLRNForward(handle, lrnDesc, alpha, xDesc, x, beta, yDesc, y, do_backward, workSpace):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenLRNForward(handle, lrnDesc, alphaRef, xDesc, x, betaRef, yDesc, y, do_backward, workSpace)
	miopenCheckStatus(status)


_libmiopen.miopenLRNBackward.restype = int
_libmiopen.miopenLRNBackward.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p
]
def miopenLRNBackward(handle, lrnDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx, workSpace):
	alphaRef, betaRef = ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(beta))

	status = _libmiopen.miopenLRNBackward(
		handle, lrnDesc, alphaRef, yDesc, y, dyDesc, dy, xDesc, x, betaRef, dxDesc, dx, workSpace
	)
	miopenCheckStatus(status)


_libmiopen.miopenDestroyLRNDescriptor.restype = int
_libmiopen.miopenDestroyLRNDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyLRNDescriptor(lrnDesc):
	status = _libmiopen.miopenDestroyLRNDescriptor(lrnDesc)
	miopenCheckStatus(status)


_libmiopen.miopenCreateRNNDescriptor.restype = int
_libmiopen.miopenCreateRNNDescriptor.argtypes = [ctypes.c_void_p]
def miopenCreateRNNDescriptor():
	rnnDesc = ctypes.c_void_p()

	status = _libmiopen.miopenCreateRNNDescriptor(ctypes.byref(rnnDesc))
	miopenCheckStatus(status)

	return rnnDesc.value


_libmiopen.miopenSetRNNDescriptor.restype = int
_libmiopen.miopenSetRNNDescriptor.argtypes = [
	ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
	ctypes.c_int
]
def miopenSetRNNDescriptor(rnnDesc, hsize, nlayers, inMode, direction, rnnMode, biasMode, algo, dataType):
	status = _libmiopen.miopenSetRNNDescriptor(
		rnnDesc, hsize, nlayers, inMode, direction, rnnMode, biasMode, algo, dataType
	)
	miopenCheckStatus(status)


_libmiopen.miopenDestroyRNNDescriptor.restype = int
_libmiopen.miopenDestroyRNNDescriptor.argtypes = [ctypes.c_void_p]
def miopenDestroyRNNDescriptor(rnnDesc):
	status = _libmiopen.miopenDestroyRNNDescriptor(rnnDesc)
	miopenCheckStatus(status)


_libmiopen.miopenGetRNNParamsSize.restype = int
_libmiopen.miopenGetRNNParamsSize.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
]
def miopenGetRNNParamsSize(handle, rnnDesc, xDesc, dtype):
	numBytes = ctypes.c_size_t()

	status = _libmiopen.miopenGetRNNParamsSize(handle, rnnDesc, xDesc, ctypes.byref(numBytes), dtype)
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNLayerParamSize.restype = int
_libmiopen.miopenGetRNNLayerParamSize.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
]
def miopenGetRNNLayerParamSize(handle, rnnDesc, layer, xDesc, paramID):
	numBytes = ctypes.c_size_t()

	status = _libmiopen.miopenGetRNNLayerParamSize(handle, rnnDesc, layer, xDesc, paramID, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNLayerBiasSize.restype = int
_libmiopen.miopenGetRNNLayerBiasSize.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
]
def miopenGetRNNLayerBiasSize(handle, rnnDesc, layer, biasID):
	numBytes = ctypes.c_size_t()

	status = _libmiopen.miopenGetRNNLayerBiasSize(handle, rnnDesc, layer, biasID, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNLayerParam.restype = int
_libmiopen.miopenGetRNNLayerParam.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
	ctypes.c_void_p, ctypes.c_void_p
]
def miopenGetRNNLayerParam(handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam):
	status = _libmiopen.miopenGetRNNLayerParam(handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam)
	miopenCheckStatus(status)


_libmiopen.miopenGetRNNLayerBias.restype = int
_libmiopen.miopenGetRNNLayerBias.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
	ctypes.c_void_p, ctypes.c_void_p
]
def miopenGetRNNLayerBias(handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias):
	status = _libmiopen.miopenGetRNNLayerBias(handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias)
	miopenCheckStatus(status)


_libmiopen.miopenSetRNNLayerParam.restype = int
_libmiopen.miopenSetRNNLayerParam.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
	ctypes.c_void_p, ctypes.c_void_p
]
def miopenSetRNNLayerParam(handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam):
	status = _libmiopen.miopenSetRNNLayerParam(handle, rnnDesc, layer, xDesc, wDesc, w, paramID, paramDesc, layerParam)
	miopenCheckStatus(status)


_libmiopen.miopenSetRNNLayerBias.restype = int
_libmiopen.miopenSetRNNLayerBias.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
	ctypes.c_void_p, ctypes.c_void_p
]
def miopenSetRNNLayerBias(handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias):
	status = _libmiopen.miopenSetRNNLayerBias(handle, rnnDesc, layer, xDesc, wDesc, w, biasID, biasDesc, layerBias)
	miopenCheckStatus(status)


_libmiopen.miopenGetRNNWorkspaceSize.restype = int
_libmiopen.miopenGetRNNWorkspaceSize.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
]
def miopenGetRNNWorkspaceSize(handle, rnnDesc, sequenceLen, xDesc):
	numBytes = ctypes.c_size_t()
	xDesc = (ctypes.c_void_p * len(xDesc))(*xDesc)

	status = _libmiopen.miopenGetRNNWorkspaceSize(handle, rnnDesc, sequenceLen, xDesc, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenGetRNNTrainingReserveSize.restype = int
_libmiopen.miopenGetRNNTrainingReserveSize.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
]
def miopenGetRNNTrainingReserveSize(handle, rnnDesc, sequenceLen, xDesc):
	numBytes = ctypes.c_size_t()
	xDesc = (ctypes.c_void_p * len(xDesc))(*xDesc)

	status = _libmiopen.miopenGetRNNTrainingReserveSize(handle, rnnDesc, sequenceLen, xDesc, ctypes.byref(numBytes))
	miopenCheckStatus(status)

	return numBytes.value


_libmiopen.miopenRNNForwardTraining.restype = int
_libmiopen.miopenRNNForwardTraining.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t
]
def miopenRNNForwardTraining(handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
							 cyDesc, cy, workSpace, workSpaceNumBytes, reserveSpace, reserveSpaceNumBytes):
	xDesc, yDesc = (ctypes.c_void_p * len(xDesc))(*xDesc), (ctypes.c_void_p * len(yDesc))(*yDesc)

	status = _libmiopen.miopenRNNForwardTraining(
		handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy,
		workSpace, workSpaceNumBytes, reserveSpace, reserveSpaceNumBytes
	)
	miopenCheckStatus(status)


_libmiopen.miopenRNNBackwardData.restype = int
_libmiopen.miopenRNNBackwardData.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
	ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t
]
def miopenRNNBackwardData(handle, rnnDesc, seqLen, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w,
						  hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceNumBytes,
						  reserveSpace, reserveSpaceNumBytes):
	yDesc = (ctypes.c_void_p * len(yDesc))(*yDesc)
	dyDesc, dxDesc = (ctypes.c_void_p * len(dyDesc))(*dyDesc), (ctypes.c_void_p * len(dxDesc))(*dxDesc)

	status = _libmiopen.miopenRNNBackwardData(
		handle, rnnDesc, seqLen, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx,
		dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workSpace, workSpaceNumBytes, reserveSpace, reserveSpaceNumBytes
	)
	miopenCheckStatus(status)


_libmiopen.miopenRNNBackwardWeights.restype = int
_libmiopen.miopenRNNBackwardWeights.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t
]
def miopenRNNBackwardWeights(handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, yDesc, y, dwDesc, dw, workSpace,
							 workSpaceNumBytes, reserveSpace, reserveSpaceNumBytes):
	xDesc, yDesc = (ctypes.c_void_p * len(xDesc))(*xDesc), (ctypes.c_void_p * len(yDesc))(*yDesc)

	status = _libmiopen.miopenRNNBackwardWeights(
		handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, yDesc, y, dwDesc, dw, workSpace, workSpaceNumBytes,
		reserveSpace, reserveSpaceNumBytes
	)
	miopenCheckStatus(status)


_libmiopen.miopenRNNForwardInference.restype = int
_libmiopen.miopenRNNForwardInference.argtypes = [
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
	ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
]
def miopenRNNForwardInference(handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy,
							  cyDesc, cy, workSpace, workSpaceNumBytes):
	xDesc, yDesc = (ctypes.c_void_p * len(xDesc))(*xDesc), (ctypes.c_void_p * len(yDesc))(*yDesc)

	status = _libmiopen.miopenRNNForwardInference(
		handle, rnnDesc, seqLen, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy,
		workSpace, workSpaceNumBytes
	)
	miopenCheckStatus(status)
