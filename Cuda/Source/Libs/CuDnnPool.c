#include "Libs.h"


typedef struct CuDnn_PoolParams
{
	size_t size[GPUTENSOR_MAP_MAX], stride[GPUTENSOR_MAP_MAX], pad[GPUTENSOR_MAP_MAX];
	size_t ndim;
}
CuDnn_PoolParams;


inline static bool CuDnn_unpackPoolParams(PyObject *pysize, PyObject *pystride, PyObject *pypad, size_t ndim,
										  CuDnn_PoolParams *params)
{
	if (!CuDnn_unpackIntSequence(pysize, params->size, ndim, 1, "size"))       return false;
	if (!CuDnn_unpackIntSequence(pystride, params->stride, ndim, 1, "stride")) return false;
	if (!CuDnn_unpackIntSequence(pypad, params->pad, ndim, 0, "pad"))          return false;

	params->ndim = ndim;
	return true;
}


inline static bool CuDnn_poolNd_outshape(size_t *outshape, CuDnn_PoolParams params, const size_t *inshape)
{
	for (size_t i = 0; i < params.ndim; i += 1)
	{
		size_t size = inshape[2 + i] + 2 * params.pad[i];

		if (size < params.size[i])
		{
			PyErr_Format(PyExc_ValueError, "invalid input map size on dim #%d", (int)(i + 1));
			return false;
		}

		outshape[2 + i] = (size - params.size[i]) / params.stride[i] + 1;
	}

	outshape[0] = inshape[0], outshape[1] = inshape[1];
	return true;
}


inline static bool CuDnn_describePool(cudnnPoolingDescriptor_t *desc, CuDnn_PoolParams params, cudnnPoolingMode_t mode)
{
	CUDNN_CHECK(cudnnCreatePoolingDescriptor(desc), goto error_1);

	assert(CuDnn_isValidDim(params.ndim + 2));
	int windowDimA[GPUTENSOR_MAP_MAX], paddingA[GPUTENSOR_MAP_MAX], strideA[GPUTENSOR_MAP_MAX];

	for (size_t i = 0; i < params.ndim; i += 1)
		windowDimA[i] = (int)params.size[i], paddingA[i] = (int)params.pad[i], strideA[i] = (int)params.stride[i];

	CUDNN_CHECK(cudnnSetPoolingNdDescriptor(
		*desc, mode, CUDNN_NOT_PROPAGATE_NAN, (int)params.ndim, windowDimA, paddingA, strideA
	), goto error_2);

	return true;

error_2:
	CUDNN_ASSERT(cudnnDestroyPoolingDescriptor(*desc));

error_1:
	return false;
}


inline static bool CuDnn_Context_poolNd(CuDnn_Context *self, const Cuda_GPUArray *data, Cuda_GPUArray *out,
										CuDnn_PoolParams params, cudnnPoolingMode_t mode)
{
	bool status = false;
	float alpha = 1.0f, beta = 0.0f;

	cudnnTensorDescriptor_t dataDesc, outDesc;
	cudnnPoolingDescriptor_t poolDesc;

	if (!CuDnn_describeTensor(&dataDesc, data))       goto error_1;
	if (!CuDnn_describeTensor(&outDesc, out))         goto error_2;
	if (!CuDnn_describePool(&poolDesc, params, mode)) goto error_3;

	CUDNN_CHECK(cudnnPoolingForward(
		self->handle, poolDesc, &alpha, dataDesc, data->gpudata->ptr, &beta, outDesc, out->gpudata->ptr
	), goto error_4);

	status = true;

error_4:
	CUDNN_ASSERT(cudnnDestroyPoolingDescriptor(poolDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


const char CuDnn_Context_pyPoolNd_doc[] = PyDoc_STR(
	"poolNd(self, data, size=2, stride=2, pad=0, mode=" CUDNN_BACKEND_NAME ".POOL_MODE_MAX, "
	"out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_pyPoolNd(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"data", "size", "stride", "pad", "mode", "out", "allocator", NULL};

	Cuda_GPUArray *data;
	PyObject *pysize = NULL, *pystride = NULL, *pypad = NULL, *pyout = NULL, *pyalloc = NULL;
	int mode = CUDNN_POOLING_MAX;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!|OOOiOO", (char **)kwlist, Cuda_GPUArray_Type, &data,
		&pysize, &pystride, &pypad, &mode, &pyout, &pyalloc
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

	CuDnn_PoolParams params;
	if (!CuDnn_unpackPoolParams(pysize, pystride, pypad, data->ndim - 2, &params))
		return NULL;

	size_t outshape[GPUTENSOR_DIM_MAX];
	if (!CuDnn_poolNd_outshape(outshape, params, CUDA_GPUARRAY_SHAPE(data)))
		return NULL;

	out = CuDnn_enforceAllocated(out, allocator, outshape, data->ndim, data->dtype, false);
	if (out == NULL) return NULL;

	if (!CuDnn_Context_poolNd((CuDnn_Context *)self, data, out, params, (cudnnPoolingMode_t)mode))
	{
		Py_DECREF(out);
		out = NULL;
	}

	return (PyObject *)out;
}


inline static bool CuDnn_Context_poolNdBackward(CuDnn_Context *self, const Cuda_GPUArray *grad,
												const Cuda_GPUArray *indata, const Cuda_GPUArray *outdata,
												Cuda_GPUArray *out, CuDnn_PoolParams params, cudnnPoolingMode_t mode)
{
	bool status = false;
	float alpha = 1.0f, beta = 0.0f;

	cudnnTensorDescriptor_t gradDesc, indataDesc, outdataDesc, outDesc;
	cudnnPoolingDescriptor_t poolDesc;

	if (!CuDnn_describeTensor(&gradDesc, grad))       goto error_1;
	if (!CuDnn_describeTensor(&indataDesc, indata))   goto error_2;
	if (!CuDnn_describeTensor(&outdataDesc, outdata)) goto error_3;
	if (!CuDnn_describeTensor(&outDesc, out))         goto error_4;
	if (!CuDnn_describePool(&poolDesc, params, mode)) goto error_5;

	CUDNN_CHECK(cudnnPoolingBackward(
		self->handle, poolDesc, &alpha, outdataDesc, outdata->gpudata->ptr, gradDesc, grad->gpudata->ptr,
		indataDesc, indata->gpudata->ptr, &beta, outDesc, out->gpudata->ptr
	), goto error_6);

	status = true;

error_6:
	CUDNN_ASSERT(cudnnDestroyPoolingDescriptor(poolDesc));
error_5:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_4:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outdataDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(indataDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(gradDesc));
error_1:
	return status;
}


const char CuDnn_Context_pyPoolNdBackward_doc[] = PyDoc_STR(
	"poolNdBackward(self, grad, indata, outdata, size=2, stride=2, pad=0, mode=" CUDNN_BACKEND_NAME ".POOL_MODE_MAX, "
	"out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_pyPoolNdBackward(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"grad", "indata", "outdata", "size", "stride", "pad", "mode", "out", "allocator", NULL};

	Cuda_GPUArray *grad, *indata, *outdata;
	PyObject *pysize = NULL, *pystride = NULL, *pypad = NULL, *pyout = NULL, *pyalloc = NULL;
	int mode = CUDNN_POOLING_MAX;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!O!|OOOiOO", (char **)kwlist, Cuda_GPUArray_Type, &grad, Cuda_GPUArray_Type, &indata,
		Cuda_GPUArray_Type, &outdata, &pysize, &pystride, &pypad, &mode, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidDim(grad->ndim) || grad->ndim != indata->ndim || grad->ndim != outdata->ndim)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(grad->dtype) || grad->dtype != indata->dtype || grad->dtype != outdata->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	CuDnn_PoolParams params;
	if (!CuDnn_unpackPoolParams(pysize, pystride, pypad, grad->ndim - 2, &params))
		return NULL;

	out = CuDnn_enforceAllocated(out, allocator, CUDA_GPUARRAY_SHAPE(indata), indata->ndim, indata->dtype, false);
	if (out == NULL) return NULL;

	if (!CuDnn_Context_poolNdBackward(
		(CuDnn_Context *)self, grad, indata, outdata, out, params, (cudnnPoolingMode_t)mode
	))
	{
		Py_DECREF(out);
		out = NULL;
	}

	return (PyObject *)out;
}


void CuDnnPool_moduleInit(PyObject *m)
{
	PyModule_AddIntConstant(m, "POOL_MODE_MAX", CUDNN_POOLING_MAX);
	PyModule_AddIntConstant(m, "POOL_MODE_AVG_WITH_PAD", CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
	PyModule_AddIntConstant(m, "POOL_MODE_AVG_NO_PAD", CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
	PyModule_AddIntConstant(m, "POOL_MODE_MAX_DETERMINISM", CUDNN_POOLING_MAX_DETERMINISTIC);
}
