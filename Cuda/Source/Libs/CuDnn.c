#include "Libs.h"
#include "../TraceMalloc/TraceMalloc.gen.h"


PyObject *CuDnn_Error = NULL;


PyDoc_STRVAR(CuDnn_getVersion_doc, "getVersion() -> int");
static PyObject *CuDnn_getVersion(PyObject *self, PyObject *args)
{
	(void)self, (void)args;

	size_t version = cudnnGetVersion();
	return Py_BuildValue("n", (Py_ssize_t)version);
}


static PyObject *CuDnn_Context_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)args, (void)kwds;

	CuDnn_Context *self = (CuDnn_Context *)type->tp_alloc(type, 0);
	if (self == NULL)
		goto error_1;

	CUDNN_CHECK(cudnnCreate(&self->handle), goto error_2);
	self->mathType = CUDNN_DEFAULT_MATH;

#if defined(TRACE_CUDA_CUDNN)
	fprintf(stderr, "[" CUDNN_CONTEXT_OBJNAME "] (0x%" PRIXMAX ") Allocated context\n", (size_t)self);
#endif

	return (PyObject *)self;

error_2:
	self->handle = NULL;
	Py_DECREF(self);

error_1:
	return NULL;
}


static void CuDnn_Context_dealloc(PyObject *self)
{
	CuDnn_Context *pyctx = (CuDnn_Context *)self;

	if (pyctx->handle != NULL)
	{
		CUDNN_ASSERT(cudnnDestroy(pyctx->handle));

#if defined(TRACE_CUDA_CUDNN)
		fprintf(stderr, "[" CUDNN_CONTEXT_OBJNAME "] (0x%" PRIXMAX ") Deallocated context\n", (size_t)self);
#endif
	}

	Py_TYPE(self)->tp_free(self);
}


PyDoc_STRVAR(CuDnn_Context_enableTensorOps_doc, "enableTensorOps(self, enable)");
static PyObject *CuDnn_Context_enableTensorOps(PyObject *self, PyObject *args)
{
	CuDnn_Context *pyctx = (CuDnn_Context *)self;
	int enable;

	if (!PyArg_ParseTuple(args, "p", &enable))
		return NULL;

	pyctx->mathType = enable ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
	Py_RETURN_NONE;
}


Cuda_GPUArray *CuDnn_enforceAllocated(Cuda_GPUArray *out, Cuda_MemoryPool *allocator, const size_t *shape, size_t ndim,
									  Cuda_DataType dtype, bool zeroOut)
{
	if (out == NULL)
	{
		Cuda_ArraySpec spec;
		Cuda_fillShapeAsContiguous(spec.shape, spec.strides, shape, ndim, Cuda_dtypeSize(dtype));

		spec.ndim = ndim, spec.size = spec.strides[0] * spec.shape[0] / Cuda_dtypeSize(dtype);
		spec.dtype = dtype, spec.contiguous = true;

		out = Cuda_GPUArray_newWithAllocator(allocator, NULL, &spec);
		if (out == NULL)
			return NULL;
	}
	else
	{
		bool shapesAreEqual = true;

		if (out->ndim != ndim)
			shapesAreEqual = false;

		if (shapesAreEqual)
		{
			const size_t *outshape = CUDA_GPUARRAY_SHAPE(out);

			for (size_t i = 0; i < out->ndim; i += 1)
				if (outshape[i] != shape[i])
				{
					shapesAreEqual = false;
					break;
				}
		}

		if (!shapesAreEqual || out->dtype != dtype)
		{
			PyErr_SetString(PyExc_ValueError, "invalid output gpuarray data layout");
			return NULL;
		}

		Py_INCREF(out);
	}

	if (zeroOut)
	{
		assert(out->contiguous);
		CU_CHECK(cuMemsetD8((CUdeviceptr)out->gpudata->ptr, 0x00, out->size * Cuda_dtypeSize(out->dtype)), goto error);
	}

	return out;

error:
	Py_DECREF(out);
	return NULL;
}


bool CuDnn_describeTensorFromShape(cudnnTensorDescriptor_t *desc, const size_t *shape, const size_t *strides,
								   size_t ndim, Cuda_DataType dtype)
{
	CUDNN_CHECK(cudnnCreateTensorDescriptor(desc), goto error_1);

	assert(CuDnn_isValidDescribedDim(ndim));
	int dimA[GPUTENSOR_DIM_MAX], strideA[GPUTENSOR_DIM_MAX];

	size_t itemsize; itemsize = Cuda_dtypeSize(dtype);

	for (size_t i = 0; i < ndim; i += 1)
		dimA[i] = (int)shape[i], strideA[i] = (int)(strides[i] / itemsize);

	CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc, CuDnn_dtypeToDnn(dtype), (int)ndim, dimA, strideA), goto error_2);
	return true;

error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(*desc));

error_1:
	return false;
}


bool CuDnn_describeTensor(cudnnTensorDescriptor_t *desc, const Cuda_GPUArray *tensor)
{
	return CuDnn_describeTensorFromShape(
		desc, CUDA_GPUARRAY_SHAPE(tensor), CUDA_GPUARRAY_STRIDES(tensor), tensor->ndim, tensor->dtype
	);
}


bool CuDnn_describe1DTensor(cudnnTensorDescriptor_t *desc, const Cuda_GPUArray *tn, size_t ndim)
{
	CUDNN_CHECK(cudnnCreateTensorDescriptor(desc), goto error_1);

	assert(tn->ndim == 1 && tn->contiguous);
	const size_t *shape; shape = CUDA_GPUARRAY_SHAPE(tn);

	int dimA[GPUTENSOR_DIM_MAX], strideA[GPUTENSOR_DIM_MAX];

	for (size_t i = 0; i < ndim; i += 1)
		dimA[i] = (i == 1) ? (int)shape[0] : 1, strideA[i] = (i < 1) ? (int)shape[0] : 1;

	CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc, CuDnn_dtypeToDnn(tn->dtype), (int)ndim, dimA, strideA), goto error_2);
	return true;

error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(*desc));

error_1:
	return false;
}


bool CuDnn_describeFilterFromShape(cudnnFilterDescriptor_t *desc, const size_t *shape, size_t ndim, Cuda_DataType dtype)
{
	CUDNN_CHECK(cudnnCreateFilterDescriptor(desc), goto error_1);

	assert(CuDnn_isValidDescribedDim(ndim));
	int filterDimA[GPUTENSOR_DIM_MAX];

	for (size_t i = 0; i < ndim; i += 1)
		filterDimA[i] = (int)shape[i];

	CUDNN_CHECK(cudnnSetFilterNdDescriptor(
		*desc, CuDnn_dtypeToDnn(dtype), CUDNN_TENSOR_NCHW, (int)ndim, filterDimA
	), goto error_2);
	return true;

error_2:
	CUDNN_ASSERT(cudnnDestroyFilterDescriptor(*desc));

error_1:
	return false;
}


inline static bool CuDnn_describeFilter(cudnnFilterDescriptor_t *desc, const Cuda_GPUArray *W)
{
	assert(W->contiguous);
	return CuDnn_describeFilterFromShape(desc, CUDA_GPUARRAY_SHAPE(W), W->ndim, W->dtype);
}


typedef struct CuDnn_ConvParams
{
	size_t stride[GPUTENSOR_DIM_MAX - 2], pad[GPUTENSOR_DIM_MAX - 2], dilation[GPUTENSOR_DIM_MAX - 2];
	size_t groups, ndim;
}
CuDnn_ConvParams;


inline static bool CuDnn_unpackConvParams(PyObject *pystride, PyObject *pypad, PyObject *pydil,
										  size_t groups, size_t ndim, CuDnn_ConvParams *params)
{
	if (!CuDnn_unpackIntSequence(pystride, params->stride, ndim, 1, "stride"))  return false;
	if (!CuDnn_unpackIntSequence(pypad, params->pad, ndim, 0, "pad"))           return false;
	if (!CuDnn_unpackIntSequence(pydil, params->dilation, ndim, 1, "dilation")) return false;

	params->groups = groups, params->ndim = ndim;
	return true;
}


inline static bool CuDnn_convNd_outshape(size_t *outshape, CuDnn_ConvParams params,
										 const size_t *inshape, const size_t *Wshape)
{
	if (inshape[1] != Wshape[1] * params.groups)
	{
		PyErr_SetString(PyExc_ValueError, "invalid number of input maps");
		return false;
	}

	for (size_t i = 0; i < params.ndim; i += 1)
	{
		size_t size = inshape[2 + i] + 2 * params.pad[i], fsize = params.dilation[i] * (Wshape[2 + i] - 1) + 1;

		if (size < fsize)
		{
			PyErr_Format(PyExc_ValueError, "invalid input map size on dim #%d", (int)(i + 1));
			return false;
		}

		outshape[2 + i] = (size - fsize) / params.stride[i] + 1;
	}

	outshape[0] = inshape[0], outshape[1] = Wshape[0];
	return true;
}


inline static bool CuDnn_convNd_inshape(size_t *inshape, CuDnn_ConvParams params,
										const size_t *outshape, const size_t *Wshape)
{
	if (outshape[1] != Wshape[0])
	{
		PyErr_SetString(PyExc_ValueError, "invalid number of output maps");
		return false;
	}

	for (size_t i = 0; i < params.ndim; i += 1)
		inshape[2 + i] = params.stride[i] * (outshape[2 + i] - 1) +
						 params.dilation[i] * (Wshape[2 + i] - 1) - 2 * params.pad[i] + 1;

	inshape[0] = outshape[0], inshape[1] = Wshape[1] * params.groups;
	return true;
}


inline static bool CuDnn_convNd_isMapFullyCovered(bool *isCovered, size_t *inshape, CuDnn_ConvParams params,
												  const Cuda_GPUArray *tensor, size_t batchsize, const size_t *Wshape)
{
	const size_t *shape = CUDA_GPUARRAY_SHAPE(tensor);

	if (tensor->ndim != params.ndim + 2 || shape[0] != batchsize)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return false;
	}

	inshape[0] = shape[0];

	if (shape[1] != Wshape[1] * params.groups)
	{
		PyErr_SetString(PyExc_ValueError, "invalid number of input maps");
		return false;
	}

	inshape[1] = shape[1];
	bool covered = true;

	for (size_t i = 0; i < params.ndim; i += 1)
	{
		size_t size = shape[2 + i] + 2 * params.pad[i], fsize = params.dilation[i] * (Wshape[2 + i] - 1) + 1;

		if (size < fsize)
		{
			PyErr_Format(PyExc_ValueError, "invalid input map size on dim #%d", (int)(i + 1));
			return false;
		}

		if ((size - fsize) % params.stride[i] != 0)
			covered = false;

		inshape[i + 2] = shape[i + 2];
	}

	*isCovered = covered;
	return true;
}


inline static bool CuDnn_Context_describeConv(CuDnn_Context *self, cudnnConvolutionDescriptor_t *desc,
											  CuDnn_ConvParams params)
{
	CUDNN_CHECK(cudnnCreateConvolutionDescriptor(desc), goto error_1);

	assert(CuDnn_isValidDim(params.ndim + 2));
	int strideA[GPUTENSOR_DIM_MAX - 2], padA[GPUTENSOR_DIM_MAX - 2], dilationA[GPUTENSOR_DIM_MAX - 2];

	for (size_t i = 0; i < params.ndim; i += 1)
		strideA[i] = (int)params.stride[i], padA[i] = (int)params.pad[i], dilationA[i] = (int)params.dilation[i];

	CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
		*desc, (int)params.ndim, padA, strideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
	), goto error_2);

	CUDNN_ASSERT(cudnnSetConvolutionMathType(*desc, self->mathType));
	CUDNN_ASSERT(cudnnSetConvolutionGroupCount(*desc, (int)params.groups));

	return true;

error_2:
	CUDNN_ASSERT(cudnnDestroyConvolutionDescriptor(*desc));

error_1:
	return false;
}


static bool CuDnn_Context_toTensorAddBias(CuDnn_Context *self, cudnnTensorDescriptor_t desc,
										  Cuda_GPUArray *tensor, const Cuda_GPUArray *bias)
{
	if (!CuDnn_isValid1DTensor(bias, CUDA_GPUARRAY_SHAPE(tensor)[1], tensor->dtype, "bias"))
		return false;

	cudnnTensorDescriptor_t biasDesc;
	if (!CuDnn_describe1DTensor(&biasDesc, bias, tensor->ndim))
		return false;

	float alpha = 1.0f, beta = 1.0f;
	bool status = true;

	CUDNN_CHECK(cudnnAddTensor(
		self->handle, &alpha, biasDesc, bias->gpudata->ptr, &beta, desc, tensor->gpudata->ptr
	), status = false);

	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(biasDesc));
	return status;
}


static bool CuDnn_Context_convNdBackwardBias(CuDnn_Context *self, cudnnTensorDescriptor_t desc,
											 const Cuda_GPUArray *tensor, float scale, float momentum,
											 Cuda_GPUArray *bgrad)
{
	if (!CuDnn_isValid1DTensor(bgrad, CUDA_GPUARRAY_SHAPE(tensor)[1], tensor->dtype, "bgrad"))
		return false;

	cudnnTensorDescriptor_t bgradDesc;
	if (!CuDnn_describe1DTensor(&bgradDesc, bgrad, tensor->ndim))
		return false;

	bool status = true;

	CUDNN_CHECK(cudnnConvolutionBackwardBias(
		self->handle, &scale, desc, tensor->gpudata->ptr, &momentum, bgradDesc, bgrad->gpudata->ptr
	), status = false);

	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(bgradDesc));
	return status;
}


inline static bool CuDnn_Context_convNd(CuDnn_Context *self, const Cuda_GPUArray *data, const Cuda_GPUArray *W,
										const Cuda_GPUArray *bias, Cuda_GPUArray *out, CuDnn_ConvParams params,
										cudnnConvolutionFwdAlgo_t algo, Cuda_MemoryPool *allocator)
{
	bool status = false;
	float alpha = 1.0f, beta = 0.0f;

	cudnnTensorDescriptor_t dataDesc, outDesc;
	cudnnFilterDescriptor_t wDesc;
	cudnnConvolutionDescriptor_t convDesc;

	if (!CuDnn_describeTensor(&dataDesc, data))                goto error_1;
	if (!CuDnn_describeTensor(&outDesc, out))                  goto error_2;
	if (!CuDnn_describeFilter(&wDesc, W))                      goto error_3;
	if (!CuDnn_Context_describeConv(self, &convDesc, params))  goto error_4;

	size_t size;
	CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
		self->handle, dataDesc, wDesc, convDesc, outDesc, algo, &size
	), goto error_5);

	Cuda_Buffer *workspace; workspace = NULL;
	if (size > 0)
	{
		workspace = Cuda_Buffer_newWithAllocator(size, data->gpudata->device, allocator);
		if (workspace == NULL) goto error_5;
	}

	CUDNN_CHECK(cudnnConvolutionForward(
		self->handle, &alpha, dataDesc, data->gpudata->ptr, wDesc, W->gpudata->ptr, convDesc, algo,
		workspace == NULL ? NULL : workspace->ptr, size, &beta, outDesc, out->gpudata->ptr
	), goto error_6);

	if (bias != NULL && !CuDnn_Context_toTensorAddBias(self, outDesc, out, bias))
		goto error_6;

	status = true;

error_6:
	if (workspace != NULL)
		Py_DECREF(workspace);

error_5:
	CUDNN_ASSERT(cudnnDestroyConvolutionDescriptor(convDesc));
error_4:
	CUDNN_ASSERT(cudnnDestroyFilterDescriptor(wDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


PyDoc_STRVAR(
	CuDnn_Context_pyConvNd_doc,
	"convNd(self, data, W, bias=None, stride=1, pad=0, dilation=1, groups=1, "
	"algo=" CUDNN_BACKEND_NAME ".CONV_FWD_IMPLICIT_GEMM, out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
static PyObject *CuDnn_Context_pyConvNd(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {
		"data", "W", "bias", "stride", "pad", "dilation", "groups", "algo", "out", "allocator", NULL
	};

	Cuda_GPUArray *data, *W;
	PyObject *pybias = NULL, *pystride = NULL, *pypad = NULL, *pydil = NULL;

	int groups = 1, algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	PyObject *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!|OOOOiiOO", (char **)kwlist, Cuda_GPUArray_Type, &data, Cuda_GPUArray_Type, &W,
		&pybias, &pystride, &pypad, &pydil, &groups, &algo, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pybias, Cuda_GPUArray_Type, "bias"))         return NULL;
	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *bias = (Cuda_GPUArray *)pybias, *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidDim(data->ndim) || data->ndim != W->ndim)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(data->dtype) || data->dtype != W->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	CuDnn_ConvParams params;
	if (!CuDnn_unpackConvParams(pystride, pypad, pydil, groups, data->ndim - 2, &params))
		return NULL;

	size_t outshape[GPUTENSOR_DIM_MAX];
	if (!CuDnn_convNd_outshape(outshape, params, CUDA_GPUARRAY_SHAPE(data), CUDA_GPUARRAY_SHAPE(W)))
		return NULL;

	out = CuDnn_enforceAllocated(out, allocator, outshape, data->ndim, data->dtype, false);
	if (out == NULL) return NULL;

	if (!CuDnn_Context_convNd(
		(CuDnn_Context *)self, data, W, bias, out, params, (cudnnConvolutionFwdAlgo_t)algo, allocator
	))
	{
		Py_DECREF(out);
		out = NULL;
	}

	return (PyObject *)out;
}


inline static bool CuDnn_Context_convNdBackwardData(CuDnn_Context *self, const Cuda_GPUArray *grad,
													const Cuda_GPUArray *W, const Cuda_GPUArray *bias,
													Cuda_GPUArray *out, CuDnn_ConvParams params,
													cudnnConvolutionBwdDataAlgo_t algo, Cuda_MemoryPool *allocator)
{
	bool status = false;
	float alpha = 1.0f, beta = 0.0f;

	cudnnTensorDescriptor_t gradDesc, outDesc;
	cudnnFilterDescriptor_t wDesc;
	cudnnConvolutionDescriptor_t convDesc;

	if (!CuDnn_describeTensor(&gradDesc, grad))                goto error_1;
	if (!CuDnn_describeTensor(&outDesc, out))                  goto error_2;
	if (!CuDnn_describeFilter(&wDesc, W))                      goto error_3;
	if (!CuDnn_Context_describeConv(self, &convDesc, params))  goto error_4;

	size_t size;
	CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
		self->handle, wDesc, gradDesc, convDesc, outDesc, algo, &size
	), goto error_5);

	Cuda_Buffer *workspace; workspace = NULL;
	if (size > 0)
	{
		workspace = Cuda_Buffer_newWithAllocator(size, grad->gpudata->device, allocator);
		if (workspace == NULL)
			goto error_5;
	}

	CUDNN_CHECK(cudnnConvolutionBackwardData(
		self->handle, &alpha, wDesc, W->gpudata->ptr, gradDesc, grad->gpudata->ptr, convDesc, algo,
		workspace == NULL ? NULL : workspace->ptr, size, &beta, outDesc, out->gpudata->ptr
	), goto error_6);

	if (bias != NULL && !CuDnn_Context_toTensorAddBias(self, outDesc, out, bias))
		goto error_6;

	status = true;

error_6:
	if (workspace != NULL)
		Py_DECREF(workspace);

error_5:
	CUDNN_ASSERT(cudnnDestroyConvolutionDescriptor(convDesc));
error_4:
	CUDNN_ASSERT(cudnnDestroyFilterDescriptor(wDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(gradDesc));
error_1:
	return status;
}


PyDoc_STRVAR(
	CuDnn_Context_pyConvNdBackwardData_doc,
	"convNdBackwardData(self, grad, W, bias=None, data=None, stride=1, pad=0, dilation=1, groups=1, "
	"algo=" CUDNN_BACKEND_NAME ".CONV_BWD_DATA_ALGO_0, out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
static PyObject *CuDnn_Context_pyConvNdBackwardData(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {
		"grad", "W", "bias", "data", "stride", "pad", "dilation", "groups", "algo", "out", "allocator", NULL
	};

	Cuda_GPUArray *grad, *W;
	PyObject *pystride = NULL, *pypad = NULL, *pydil = NULL;

	int groups = 1, algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
	PyObject *pybias = NULL, *pydata = NULL, *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!|OOOOOiiOO", (char **)kwlist, Cuda_GPUArray_Type, &grad, Cuda_GPUArray_Type, &W,
		&pybias, &pydata, &pystride, &pypad, &pydil, &groups, &algo, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pybias, Cuda_GPUArray_Type, "bias"))         return NULL;
	if (!unpackPyOptional(&pydata, Cuda_GPUArray_Type, "data"))         return NULL;
	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *bias = (Cuda_GPUArray *)pybias, *data = (Cuda_GPUArray *)pydata, *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidDim(grad->ndim) || grad->ndim != W->ndim)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(grad->dtype) || grad->dtype != W->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	CuDnn_ConvParams params;
	if (!CuDnn_unpackConvParams(pystride, pypad, pydil, groups, grad->ndim - 2, &params))
		return NULL;

	bool isMapCovered = true;

	const size_t *gradshape = CUDA_GPUARRAY_SHAPE(grad), *Wshape = CUDA_GPUARRAY_SHAPE(W);
	size_t inshape[GPUTENSOR_DIM_MAX];

	if (data == NULL)
	{
		if (!CuDnn_convNd_inshape(inshape, params, gradshape, Wshape))
			return NULL;
	}
	else if (!CuDnn_convNd_isMapFullyCovered(&isMapCovered, inshape, params, data, gradshape[0], Wshape))
		return NULL;

	out = CuDnn_enforceAllocated(out, allocator, inshape, grad->ndim, grad->dtype, !isMapCovered);
	if (out == NULL) return NULL;

	if (!CuDnn_Context_convNdBackwardData(
		(CuDnn_Context *)self, grad, W, bias, out, params, (cudnnConvolutionBwdDataAlgo_t)algo, allocator
	))
	{
		Py_DECREF(out);
		out = NULL;
	}

	return (PyObject *)out;
}


inline static bool CuDnn_Context_convNdBackwardParams(CuDnn_Context *self, const Cuda_GPUArray *data,
													  const Cuda_GPUArray *grad, Cuda_GPUArray *wgrad,
													  Cuda_GPUArray *bgrad, CuDnn_ConvParams params, bool deconv,
													  float scale, float momentum, cudnnConvolutionBwdFilterAlgo_t algo,
													  Cuda_MemoryPool *allocator)
{
	bool status = false;

	cudnnTensorDescriptor_t dataDesc, gradDesc;
	cudnnFilterDescriptor_t wDesc;
	cudnnConvolutionDescriptor_t convDesc;

	if (!CuDnn_describeTensor(&dataDesc, data))                goto error_1;
	if (!CuDnn_describeTensor(&gradDesc, grad))                goto error_2;
	if (!CuDnn_describeFilter(&wDesc, wgrad))                  goto error_3;
	if (!CuDnn_Context_describeConv(self, &convDesc, params))  goto error_4;

	size_t size;
	CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
		self->handle, dataDesc, gradDesc, convDesc, wDesc, algo, &size
	), goto error_5);

	Cuda_Buffer *workspace; workspace = NULL;
	if (size > 0)
	{
		workspace = Cuda_Buffer_newWithAllocator(size, grad->gpudata->device, allocator);
		if (workspace == NULL)
			goto error_5;
	}

	CUDNN_CHECK(cudnnConvolutionBackwardFilter(
		self->handle, &scale, dataDesc, data->gpudata->ptr, gradDesc, grad->gpudata->ptr, convDesc, algo,
		workspace == NULL ? NULL : workspace->ptr, size, &momentum, wDesc, wgrad->gpudata->ptr
	), goto error_6);

	if (bgrad != NULL)
	{
		cudnnTensorDescriptor_t desc = deconv ? dataDesc : gradDesc;
		const Cuda_GPUArray *tensor = deconv ? data : grad;

		if (!CuDnn_Context_convNdBackwardBias(self, desc, tensor, scale, momentum, bgrad))
			goto error_6;
	}

	status = true;

error_6:
	if (workspace != NULL)
		Py_DECREF(workspace);

error_5:
	CUDNN_ASSERT(cudnnDestroyConvolutionDescriptor(convDesc));
error_4:
	CUDNN_ASSERT(cudnnDestroyFilterDescriptor(wDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(gradDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


PyDoc_STRVAR(
	CuDnn_Context_pyConvNdBackwardParams_doc,
	"convNdBackwardParams(self, data, grad, W, stride=1, pad=0, dilation=1, groups=1, withbias=False, deconv=False, "
	"wgrad=None, bgrad=None, scale=1.0, momentum=0.0, algo=" CUDNN_BACKEND_NAME ".CONV_BWD_PARAM_ALGO_0, "
	"allocator=None) -> "
	"Union[" CUDA_GPUARRAY_FULLNAME ", Tuple[" CUDA_GPUARRAY_FULLNAME ", " CUDA_GPUARRAY_FULLNAME "]]"
);
static PyObject *CuDnn_Context_pyConvNdBackwardParams(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {
		"data", "grad", "W", "stride", "pad", "dilation", "groups", "withbias", "deconv", "wgrad", "bgrad",
		"scale", "momentum", "algo", "allocator", NULL
	};

	Cuda_GPUArray *data, *grad, *W;
	PyObject *pystride = NULL, *pypad = NULL, *pydil = NULL, *pywgrad = NULL, *pybgrad = NULL, *pyalloc = NULL;

	float scale = 1.0f, momentum = 0.0f;
	int withbias = 0, deconv = 0, groups = 1, algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!O!|OOOippOOffiO", (char **)kwlist,
		Cuda_GPUArray_Type, &data, Cuda_GPUArray_Type, &grad, Cuda_GPUArray_Type, &W,
		&pystride, &pypad, &pydil, &groups, &withbias, &deconv, &pywgrad, &pybgrad, &scale, &momentum, &algo, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pywgrad, Cuda_GPUArray_Type, "wgrad"))       return NULL;
	if (!unpackPyOptional(&pybgrad, Cuda_GPUArray_Type, "bgrad"))       return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *wgrad = (Cuda_GPUArray *)pywgrad, *bgrad = (Cuda_GPUArray *)pybgrad;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	const size_t *datashape = CUDA_GPUARRAY_SHAPE(data), *gradshape = CUDA_GPUARRAY_SHAPE(grad);
	const size_t *Wshape = CUDA_GPUARRAY_SHAPE(W);

	if (!CuDnn_isValidDim(data->ndim) || data->ndim != grad->ndim || grad->ndim != W->ndim ||
		gradshape[1] != Wshape[0] || datashape[1] != Wshape[1] * groups)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(data->dtype) || data->dtype != grad->dtype || grad->dtype != W->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	CuDnn_ConvParams params;
	if (!CuDnn_unpackConvParams(pystride, pypad, pydil, groups, data->ndim - 2, &params))
		return NULL;

	wgrad = CuDnn_enforceAllocated(wgrad, allocator, Wshape, W->ndim, W->dtype, wgrad == NULL && momentum != 0.0f);
	if (wgrad == NULL) return NULL;

	if (withbias)
	{
		size_t maps = deconv ? datashape[1] : gradshape[1];
		bgrad = CuDnn_enforceAllocated(bgrad, allocator, &maps, 1, grad->dtype, bgrad == NULL && momentum != 0.0f);

		if (bgrad == NULL)
		{
			Py_DECREF(wgrad);
			return NULL;
		}
	}

	if (!CuDnn_Context_convNdBackwardParams(
		(CuDnn_Context *)self, data, grad, wgrad, bgrad, params, deconv, scale, momentum,
		(cudnnConvolutionBwdFilterAlgo_t)algo, allocator
	))
	{
		Py_DECREF(wgrad);
		wgrad = NULL;

		if (withbias)
		{
			Py_DECREF(bgrad);
			bgrad = NULL;
		}
	}

	return withbias ? Py_BuildValue("NN", wgrad, bgrad) : (PyObject *)wgrad;
}


inline static void CuDnn_contiguousStridesForShape(size_t *strides, const size_t *shape, size_t ndim,
												   Cuda_DataType dtype)
{
	size_t laststride = Cuda_dtypeSize(dtype);

	for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; i -= 1)
	{
		strides[i] = laststride;
		laststride *= shape[i];
	}
}


PyDoc_STRVAR(
	CuDnn_Context_convNdbenchmark_doc,
	"convNdbenchmark(self, datashape, Wshape, dtype, stride=1, pad=0, dilation=1, groups=1, algoCount=10) -> "
	"List[Tuple[int, float, int, int, int]]"
);
static PyObject *CuDnn_Context_convNdbenchmark(PyObject *self, PyObject *args, PyObject *kwds)
{
	bool status = false;
	PyObject *pyFwdResults = NULL, *pyBwdDataResults = NULL, *pyBwdParamResults = NULL;

	CuDnn_Context *context = (CuDnn_Context *)self;
	const char *kwlist[] = {"datashape", "Wshape", "dtype", "stride", "pad", "dilation", "groups", "algoCount", NULL};

	PyObject *pydatashape, *pywshape, *pystride = NULL, *pypad = NULL, *pydil = NULL;
	PyArray_Descr *pytype;
	int groups = 1, algoCount = 10;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!O&|OOOii", (char **)kwlist, &PyTuple_Type, &pydatashape, &PyTuple_Type, &pywshape,
		PyArray_DescrConverter2, &pytype, &pystride, &pypad, &pydil, &groups, &algoCount
	))
		goto error_1;

	Cuda_DataType dtype; dtype = Cuda_toDataType(pytype->type_num);
	if (dtype == DTYPE_INVALID)
		goto error_2;

	size_t ndim; ndim = PyTuple_GET_SIZE(pydatashape);
	if (!CuDnn_isValidDim(ndim))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input shape dims");
		goto error_2;
	}

	size_t datashape[GPUTENSOR_DIM_MAX], Wshape[GPUTENSOR_DIM_MAX];
	if (!CuDnn_unpackIntTuple(pydatashape, datashape, ndim, "stride")) goto error_2;
	if (!CuDnn_unpackIntTuple(pywshape, Wshape, ndim, "pad"))          goto error_2;

	CuDnn_ConvParams params;
	size_t outshape[GPUTENSOR_DIM_MAX];

	if (!CuDnn_unpackConvParams(pystride, pypad, pydil, groups, ndim - 2, &params)) goto error_2;
	if (!CuDnn_convNd_outshape(outshape, params, datashape, Wshape))                goto error_2;

	size_t strides[GPUTENSOR_DIM_MAX], outstrides[GPUTENSOR_DIM_MAX];
	CuDnn_contiguousStridesForShape(strides, datashape, ndim, dtype);
	CuDnn_contiguousStridesForShape(outstrides, outshape, ndim, dtype);

	cudnnTensorDescriptor_t dataDesc, outDesc;
	cudnnFilterDescriptor_t wDesc;
	cudnnConvolutionDescriptor_t convDesc;

	if (!CuDnn_describeTensorFromShape(&dataDesc, datashape, strides, ndim, dtype))  goto error_2;
	if (!CuDnn_describeTensorFromShape(&outDesc, outshape, outstrides, ndim, dtype)) goto error_3;
	if (!CuDnn_describeFilterFromShape(&wDesc, Wshape, ndim, dtype))                 goto error_4;
	if (!CuDnn_Context_describeConv(context, &convDesc, params))                     goto error_5;

	cudnnConvolutionFwdAlgoPerf_t *fwdResults;
	cudnnConvolutionBwdDataAlgoPerf_t *bwdDataResults;
	cudnnConvolutionBwdFilterAlgoPerf_t *bwdParamResults;

	fwdResults = (cudnnConvolutionFwdAlgoPerf_t *)TRACE_MALLOC(sizeof(*fwdResults) * algoCount);
	bwdDataResults = (cudnnConvolutionBwdDataAlgoPerf_t *)TRACE_MALLOC(sizeof(*bwdDataResults) * algoCount);
	bwdParamResults = (cudnnConvolutionBwdFilterAlgoPerf_t *)TRACE_MALLOC(sizeof(*bwdParamResults) * algoCount);

	float millisInSec; millisInSec = 1.0e-3f;
	int nalgo;

	CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
		context->handle, dataDesc, wDesc, convDesc, outDesc, algoCount, &nalgo, fwdResults
	), goto error_6);

	pyFwdResults = PyList_New(nalgo);
	if (pyFwdResults == NULL) goto error_6;

	for (int i = 0; i < nalgo; i += 1)
	{
		cudnnConvolutionFwdAlgoPerf_t perf = fwdResults[i];
		PyObject *item = Py_BuildValue(
			"ifnii", (int)perf.algo, (perf.time >= 0.0f ? perf.time * millisInSec : perf.time), (Py_ssize_t)perf.memory,
			(int)perf.determinism, (int)perf.mathType
		);

		if (item == NULL) goto error_7;
		PyList_SET_ITEM(pyFwdResults, i, item);
	}

	CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
		context->handle, wDesc, outDesc, convDesc, dataDesc, algoCount, &nalgo, bwdDataResults
	), goto error_7);

	pyBwdDataResults = PyList_New(nalgo);
	if (pyBwdDataResults == NULL) goto error_7;

	for (int i = 0; i < nalgo; i += 1)
	{
		cudnnConvolutionBwdDataAlgoPerf_t perf = bwdDataResults[i];
		PyObject *item = Py_BuildValue(
			"ifnii", (int)perf.algo, (perf.time >= 0.0f ? perf.time * millisInSec : perf.time), (Py_ssize_t)perf.memory,
			(int)perf.determinism, (int)perf.mathType
		);

		if (item == NULL) goto error_8;
		PyList_SET_ITEM(pyBwdDataResults, i, item);
	}

	CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
		context->handle, dataDesc, outDesc, convDesc, wDesc, algoCount, &nalgo, bwdParamResults
	), goto error_8);

	pyBwdParamResults = PyList_New(nalgo);
	if (pyBwdParamResults == NULL) goto error_8;

	for (int i = 0; i < nalgo; i += 1)
	{
		cudnnConvolutionBwdFilterAlgoPerf_t perf = bwdParamResults[i];
		PyObject *item = Py_BuildValue(
			"ifnii", (int)perf.algo, (perf.time >= 0.0f ? perf.time * millisInSec : perf.time), (Py_ssize_t)perf.memory,
			(int)perf.determinism, (int)perf.mathType
		);

		if (item == NULL) goto error_9;
		PyList_SET_ITEM(pyBwdParamResults, i, item);
	}

	status = true;

error_9:
	if (!status)
		Py_DECREF(pyBwdParamResults);

error_8:
	if (!status)
		Py_DECREF(pyBwdDataResults);

error_7:
	if (!status)
		Py_DECREF(pyFwdResults);

error_6:
	TRACE_FREE(fwdResults);
	TRACE_FREE(bwdDataResults);
	TRACE_FREE(bwdParamResults);

	CUDNN_ASSERT(cudnnDestroyConvolutionDescriptor(convDesc));
error_5:
	CUDNN_ASSERT(cudnnDestroyFilterDescriptor(wDesc));
error_4:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_2:
	Py_DECREF(pytype);
error_1:
	return status ? Py_BuildValue("NNN", pyFwdResults, pyBwdDataResults, pyBwdParamResults) : NULL;
}


inline static bool CuDnn_Context_softmaxNd(CuDnn_Context *self, const Cuda_GPUArray *data, Cuda_GPUArray *out,
										   cudnnSoftmaxMode_t mode, cudnnSoftmaxAlgorithm_t algo)
{
	bool status = false;
	float alpha = 1.0f, beta = 0.0f;
	cudnnTensorDescriptor_t dataDesc, outDesc;

	if (!CuDnn_describeTensor(&dataDesc, data)) goto error_1;
	if (!CuDnn_describeTensor(&outDesc, out))   goto error_2;

	CUDNN_CHECK(cudnnSoftmaxForward(
		self->handle, algo, mode, &alpha, dataDesc, data->gpudata->ptr, &beta, outDesc, out->gpudata->ptr
	), goto error_3);

	status = true;

error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


PyDoc_STRVAR(
	CuDnn_Context_pySoftmaxNd_doc,
	"softmaxNd(self, data, "
	"mode=" CUDNN_BACKEND_NAME ".SOFTMAX_MODE_SPATIAL, algo=" CUDNN_BACKEND_NAME ".SOFTMAX_ACCURATE, "
	"out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
static PyObject *CuDnn_Context_pySoftmaxNd(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"data", "mode", "algo", "out", "allocator", NULL};

	Cuda_GPUArray *data;
	PyObject *pyout = NULL, *pyalloc = NULL;
	int mode = CUDNN_SOFTMAX_MODE_CHANNEL, algo = CUDNN_SOFTMAX_ACCURATE;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!|iiOO", (char **)kwlist, Cuda_GPUArray_Type, &data, &mode, &algo, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidDim(data->ndim))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(data->dtype))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	out = CuDnn_enforceAllocated(out, allocator, CUDA_GPUARRAY_SHAPE(data), data->ndim, data->dtype, false);
	if (out == NULL) return NULL;

	if (!CuDnn_Context_softmaxNd(
		(CuDnn_Context *)self, data, out, (cudnnSoftmaxMode_t)mode, (cudnnSoftmaxAlgorithm_t)algo
	))
	{
		Py_DECREF(out);
		out = NULL;
	}

	return (PyObject *)out;
}


inline static bool CuDnn_Context_softmaxNdBackward(CuDnn_Context *self, const Cuda_GPUArray *grad,
												   const Cuda_GPUArray *outdata, Cuda_GPUArray *out,
												   cudnnSoftmaxMode_t mode, cudnnSoftmaxAlgorithm_t algo)
{
	bool status = false;
	float alpha = 1.0f, beta = 0.0f;
	cudnnTensorDescriptor_t gradDesc, outdataDesc, outDesc;

	if (!CuDnn_describeTensor(&gradDesc, grad))       goto error_1;
	if (!CuDnn_describeTensor(&outdataDesc, outdata)) goto error_2;
	if (!CuDnn_describeTensor(&outDesc, out))         goto error_3;

	CUDNN_CHECK(cudnnSoftmaxBackward(
		self->handle, algo, mode, &alpha, outdataDesc, outdata->gpudata->ptr, gradDesc, grad->gpudata->ptr,
		&beta, outDesc, out->gpudata->ptr
	), goto error_4);

	status = true;

error_4:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outdataDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(gradDesc));
error_1:
	return status;
}


PyDoc_STRVAR(
	CuDnn_Context_pySoftmaxNdBackward_doc,
	"softmaxNdBackward(self, grad, outdata, "
	"mode=" CUDNN_BACKEND_NAME ".SOFTMAX_MODE_SPATIAL, algo=" CUDNN_BACKEND_NAME ".SOFTMAX_ACCURATE, "
	"out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
static PyObject *CuDnn_Context_pySoftmaxNdBackward(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"grad", "outdata", "mode", "algo", "out", "allocator", NULL};

	Cuda_GPUArray *grad, *outdata;
	PyObject *pyout = NULL, *pyalloc = NULL;
	int mode = CUDNN_SOFTMAX_MODE_CHANNEL, algo = CUDNN_SOFTMAX_ACCURATE;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!|iiOO", (char **)kwlist, Cuda_GPUArray_Type, &grad, Cuda_GPUArray_Type, &outdata,
		&mode, &algo, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidDim(grad->ndim) || grad->ndim != outdata->ndim)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(grad->dtype) || grad->dtype != outdata->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	out = CuDnn_enforceAllocated(out, allocator, CUDA_GPUARRAY_SHAPE(grad), grad->ndim, grad->dtype, false);
	if (out == NULL) return NULL;

	if (!CuDnn_Context_softmaxNdBackward(
		(CuDnn_Context *)self, grad, outdata, out, (cudnnSoftmaxMode_t)mode, (cudnnSoftmaxAlgorithm_t)algo
	))
	{
		Py_DECREF(out);
		out = NULL;
	}

	return (PyObject *)out;
}


static PyObject *CuDnn_Context_getHandle(PyObject *self, void *closure)
{
	(void)closure;

	CuDnn_Context *pyctx = (CuDnn_Context *)self;
	return PyLong_FromSize_t((size_t)pyctx->handle);
}


static PyGetSetDef CuDnn_Context_getset[] = {
	{(char *)"handle", CuDnn_Context_getHandle, NULL, NULL, NULL},
	{NULL, NULL, NULL, NULL, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic push
	#if __GNUC__ >= 8
		#pragma GCC diagnostic ignored "-Wcast-function-type"
	#endif
#endif

static PyMethodDef CuDnn_Context_methods[] = {
	{"enableTensorOps", CuDnn_Context_enableTensorOps, METH_VARARGS, CuDnn_Context_enableTensorOps_doc},

	{"convNd", (PyCFunction)CuDnn_Context_pyConvNd, METH_VARARGS | METH_KEYWORDS, CuDnn_Context_pyConvNd_doc},
	{
		"convNdBackwardData", (PyCFunction)CuDnn_Context_pyConvNdBackwardData, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_pyConvNdBackwardData_doc
	},
	{
		"convNdBackwardParams", (PyCFunction)CuDnn_Context_pyConvNdBackwardParams, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_pyConvNdBackwardParams_doc
	},
	{
		"convNdbenchmark", (PyCFunction)CuDnn_Context_convNdbenchmark, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_convNdbenchmark_doc
	},

	{"poolNd", (PyCFunction)CuDnn_Context_pyPoolNd, METH_VARARGS | METH_KEYWORDS, CuDnn_Context_pyPoolNd_doc},
	{
		"poolNdBackward", (PyCFunction)CuDnn_Context_pyPoolNdBackward, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_pyPoolNdBackward_doc
	},

	{"softmaxNd", (PyCFunction)CuDnn_Context_pySoftmaxNd, METH_VARARGS | METH_KEYWORDS, CuDnn_Context_pySoftmaxNd_doc},
	{
		"softmaxNdBackward", (PyCFunction)CuDnn_Context_pySoftmaxNdBackward, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_pySoftmaxNdBackward_doc
	},

	{
		"batchNormNd", (PyCFunction)CuDnn_Context_pyBatchNormNd, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_pyBatchNormNd_doc
	},
	{
		"batchNormNdBackward", (PyCFunction)CuDnn_Context_pyBatchNormNdBackward, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_pyBatchNormNdBackward_doc
	},
	{"mapLRN", (PyCFunction)CuDnn_Context_pyMapLRN, METH_VARARGS | METH_KEYWORDS, CuDnn_Context_pyMapLRN_doc},
	{
		"mapLRNBackward", (PyCFunction)CuDnn_Context_pyMapLRNBackward, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_pyMapLRNBackward_doc
	},
	{
		"crossMapLRN", (PyCFunction)CuDnn_Context_pyCrossMapLRN, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_pyCrossMapLRN_doc
	},
	{
		"crossMapLRNBackward", (PyCFunction)CuDnn_Context_pyCrossMapLRNBackward, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_pyCrossMapLRNBackward_doc
	},

	{"transpose", (PyCFunction)CuDnn_Context_pyTranspose, METH_VARARGS | METH_KEYWORDS, CuDnn_Context_pyTranspose_doc},
	{"moveaxis", (PyCFunction)CuDnn_Context_moveaxis, METH_VARARGS | METH_KEYWORDS, CuDnn_Context_moveaxis_doc},
	{"swapaxes", (PyCFunction)CuDnn_Context_swapaxes, METH_VARARGS | METH_KEYWORDS, CuDnn_Context_swapaxes_doc},
	{
		"depthConcat", (PyCFunction)CuDnn_Context_depthConcat, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_depthConcat_doc
	},
	{"depthSplit", (PyCFunction)CuDnn_Context_depthSplit, METH_VARARGS | METH_KEYWORDS, CuDnn_Context_depthSplit_doc},

#if defined(CUDA_BACKEND_IS_CUDA)
	{"spatialTf", (PyCFunction)CuDnn_Context_pySpatialTf, METH_VARARGS | METH_KEYWORDS, CuDnn_Context_pySpatialTf_doc},
	{
		"spatialTfBackward", (PyCFunction)CuDnn_Context_pySpatialTfBackward, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Context_pySpatialTfBackward_doc
	},
#endif

	{NULL, NULL, 0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif

static PyType_Slot CuDnn_Context_slots[] = {
	{Py_tp_new, (void *)CuDnn_Context_new},
	{Py_tp_dealloc, (void *)CuDnn_Context_dealloc},
	{Py_tp_getset, CuDnn_Context_getset},
	{Py_tp_methods, CuDnn_Context_methods},
	{0, NULL}
};

static PyType_Spec CuDnn_Context_TypeSpec = {
	CUDNN_BACKEND_NAME "." CUDNN_CONTEXT_OBJNAME,
	sizeof(CuDnn_Context),
	0,
	Py_TPFLAGS_DEFAULT,
	CuDnn_Context_slots
};


PyTypeObject *CuDnn_Context_Type = NULL;


static PyMethodDef CuDnn_methods[] = {
	{"getVersion", CuDnn_getVersion, METH_NOARGS, CuDnn_getVersion_doc},
	{NULL, NULL, 0, NULL}
};

static PyModuleDef CuDnn_module = {
	PyModuleDef_HEAD_INIT,
	CUDNN_BACKEND_NAME,
	NULL, 0,
	CuDnn_methods,
	NULL, NULL, NULL, NULL
};


bool CuDnn_moduleInit(PyObject *module)
{
	PyObject *m = PyModule_Create(&CuDnn_module);
	if (m == NULL)
		goto error_1;

	if (!createPyClass(m, CUDNN_CONTEXT_OBJNAME, &CuDnn_Context_TypeSpec, &CuDnn_Context_Type))   goto error_2;
	if (!createPyExc(m, CUDNN_ERROR_NAME, CUDNN_BACKEND_NAME "." CUDNN_ERROR_NAME, &CuDnn_Error)) goto error_3;
	if (!CuDnnRnn_moduleInit(m))                                                                  goto error_4;
	if (PyModule_AddObject(module, CUDNN_BACKEND_NAME, m) < 0)                                    goto error_5;

	PyModule_AddIntConstant(m, "CONV_FWD_IMPLICIT_GEMM", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
	PyModule_AddIntConstant(m, "CONV_FWD_IMPLICIT_PRECOMP_GEMM", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
	PyModule_AddIntConstant(m, "CONV_FWD_GEMM", CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
	PyModule_AddIntConstant(m, "CONV_FWD_DIRECT", CUDNN_CONVOLUTION_FWD_ALGO_DIRECT);
	PyModule_AddIntConstant(m, "CONV_FWD_FFT", CUDNN_CONVOLUTION_FWD_ALGO_FFT);
	PyModule_AddIntConstant(m, "CONV_FWD_FFT_TILING", CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING);
	PyModule_AddIntConstant(m, "CONV_FWD_WINOGRAD", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);
	PyModule_AddIntConstant(m, "CONV_FWD_WINOGRAD_NONFUSED", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);

	PyModule_AddIntConstant(m, "CONV_BWD_DATA_ALGO_0", CUDNN_CONVOLUTION_BWD_DATA_ALGO_0);
	PyModule_AddIntConstant(m, "CONV_BWD_DATA_ALGO_1", CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
	PyModule_AddIntConstant(m, "CONV_BWD_DATA_FFT", CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT);
	PyModule_AddIntConstant(m, "CONV_BWD_DATA_FFT_TILING", CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING);
	PyModule_AddIntConstant(m, "CONV_BWD_DATA_WINOGRAD", CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD);
	PyModule_AddIntConstant(m, "CONV_BWD_DATA_WINOGRAD_NONFUSED", CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);

	PyModule_AddIntConstant(m, "CONV_BWD_PARAM_ALGO_0", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0);
	PyModule_AddIntConstant(m, "CONV_BWD_PARAM_ALGO_1", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
	PyModule_AddIntConstant(m, "CONV_BWD_PARAM_FFT", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT);
	PyModule_AddIntConstant(m, "CONV_BWD_PARAM_ALGO_3", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3);
	PyModule_AddIntConstant(m, "CONV_BWD_PARAM_WINOGRAD", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD);
	PyModule_AddIntConstant(m, "CONV_BWD_PARAM_WINOGRAD_NONFUSED", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED);
	PyModule_AddIntConstant(m, "CONV_BWD_PARAM_FFT_TILING", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING);

	PyModule_AddIntConstant(m, "SOFTMAX_MODE_PER_ACTIVATION", CUDNN_SOFTMAX_MODE_INSTANCE);
	PyModule_AddIntConstant(m, "SOFTMAX_MODE_SPATIAL", CUDNN_SOFTMAX_MODE_CHANNEL);

	PyModule_AddIntConstant(m, "SOFTMAX_FAST", CUDNN_SOFTMAX_FAST);
	PyModule_AddIntConstant(m, "SOFTMAX_ACCURATE", CUDNN_SOFTMAX_ACCURATE);
	PyModule_AddIntConstant(m, "SOFTMAX_LOG", CUDNN_SOFTMAX_LOG);

	PyModule_AddIntConstant(m, "MATH_DEFAULT", CUDNN_DEFAULT_MATH);
	PyModule_AddIntConstant(m, "MATH_TENSOR_OP", CUDNN_TENSOR_OP_MATH);
	PyModule_AddIntConstant(m, "MATH_TENSOR_OP_ALLOW_CONVERSION", CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

	CuDnnPool_moduleInit(m);
	CuDnnNorm_moduleInit(m);

	return true;

error_5:
	CuDnnRnn_moduleDealloc();
error_4:
	REMOVE_PY_OBJECT(&CuDnn_Error);
error_3:
	REMOVE_PY_OBJECT(&CuDnn_Context_Type);
error_2:
	Py_DECREF(m);
error_1:
	return false;
}


void CuDnn_moduleDealloc(void)
{
	CuDnnRnn_moduleDealloc();
	REMOVE_PY_OBJECT(&CuDnn_Error);
	REMOVE_PY_OBJECT(&CuDnn_Context_Type);
}
