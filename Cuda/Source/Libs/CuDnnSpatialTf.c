#include "Libs.h"

#if defined(CUDA_BACKEND_IS_CUDA)

inline static bool CuDnn_describeSpatialTf(cudnnSpatialTransformerDescriptor_t *desc, const Cuda_GPUArray *out)
{
	CUDNN_CHECK(cudnnCreateSpatialTransformerDescriptor(desc), goto error_1);

	assert(CuDnn_isValidDim(out->ndim) && out->contiguous);
	int dimA[GPUTENSOR_DIM_MAX];

	for (size_t i = 0; i < out->ndim; i += 1)
		dimA[i] = (int)CUDA_GPUARRAY_SHAPE(out)[i];

	CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(
		*desc, CUDNN_SAMPLER_BILINEAR, CuDnn_dtypeToDnn(out->dtype), (int)out->ndim, dimA
	), goto error_2);

	return true;

error_2:
	CUDNN_ASSERT(cudnnDestroySpatialTransformerDescriptor(*desc));

error_1:
	return false;
}


inline static bool CuDnn_Context_spatialTf(CuDnn_Context *self, const Cuda_GPUArray *data,
										   const Cuda_GPUArray *transform, Cuda_GPUArray *grid, Cuda_GPUArray *out)
{
	bool status = false;
	float alpha = 1.0f, beta = 0.0f;

	cudnnSpatialTransformerDescriptor_t stDesc;
	cudnnTensorDescriptor_t dataDesc, outDesc;

	if (!CuDnn_describeTensor(&dataDesc, data)) goto error_1;
	if (!CuDnn_describeTensor(&outDesc, out))   goto error_2;
	if (!CuDnn_describeSpatialTf(&stDesc, out)) goto error_3;

	CUDNN_CHECK(cudnnSpatialTfGridGeneratorForward(
		self->handle, stDesc, transform->gpudata->ptr, grid->gpudata->ptr
	), goto error_4);

	CUDNN_CHECK(cudnnSpatialTfSamplerForward(
		self->handle, stDesc, &alpha, dataDesc, data->gpudata->ptr, grid->gpudata->ptr,
		&beta, outDesc, out->gpudata->ptr
	), goto error_4);

	status = true;

error_4:
	CUDNN_ASSERT(cudnnDestroySpatialTransformerDescriptor(stDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


const char CuDnn_Context_pySpatialTf_doc[] = PyDoc_STR(
	"spatialTf(self, data, transform, outshape=None, getGrid=False, grid=None, out=None, allocator=None) -> "
	"Union[Tuple[" CUDA_GPUARRAY_FULLNAME ", " CUDA_GPUARRAY_FULLNAME "], " CUDA_GPUARRAY_FULLNAME "]"
);
PyObject *CuDnn_Context_pySpatialTf(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"data", "transform", "outshape", "getGrid", "grid", "out", "allocator", NULL};

	Cuda_GPUArray *data, *transform;
	int getGrid = 0;
	PyObject *pyoutshape = NULL, *pygrid = NULL, *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!|OpOOO", (char **)kwlist, Cuda_GPUArray_Type, &data, Cuda_GPUArray_Type, &transform,
		&pyoutshape, &getGrid, &pygrid, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyoutshape, &PyTuple_Type, "outshape"))      return NULL;
	if (!unpackPyOptional(&pygrid, Cuda_GPUArray_Type, "grid"))         return NULL;
	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *grid = (Cuda_GPUArray *)pygrid, *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	const size_t *shape = CUDA_GPUARRAY_SHAPE(data), *trshape = CUDA_GPUARRAY_SHAPE(transform);

	if (data->ndim != 4 || transform->ndim != 3 || shape[0] != trshape[0] || trshape[1] != 2 || trshape[2] != 3)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(data->dtype) || data->dtype != transform->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	size_t outshape[4];

	if (pyoutshape == NULL)
		for (size_t i = 0; i < data->ndim; i += 1) outshape[i] = shape[i];

	else
	{
		if (!CuDnn_unpackIntTuple(pyoutshape, outshape, 4, "outshape"))
			return NULL;

		if (outshape[0] != shape[0] || outshape[1] != shape[1])
		{
			PyErr_SetString(PyExc_ValueError, "invalid outshape");
			return NULL;
		}
	}

	size_t gridshape[4];
	gridshape[0] = shape[0], gridshape[1] = outshape[2], gridshape[2] = outshape[3], gridshape[3] = 2;

	grid = CuDnn_enforceAllocated(grid, allocator, gridshape, 4, data->dtype, false);
	if (grid == NULL) return NULL;

	out = CuDnn_enforceAllocated(out, allocator, outshape, 4, data->dtype, false);
	if (out == NULL)
	{
		Py_DECREF(grid);
		return NULL;
	}

	if (!CuDnn_Context_spatialTf((CuDnn_Context *)self, data, transform, grid, out))
	{
		Py_DECREF(grid);
		grid = NULL;

		Py_DECREF(out);
		out = NULL;
	}
	else if (!getGrid)
		Py_DECREF(grid);

	return getGrid ? Py_BuildValue("NN", out, grid) : (PyObject *)out;
}


typedef struct CuDnn_SpatialTfGrad
{
	Cuda_GPUArray *dgrid, *dtransform, *out;
}
CuDnn_SpatialTfGrad;


inline static bool CuDnn_spatialTfBackward_prepareInGrad(CuDnn_SpatialTfGrad *ingrad, const Cuda_GPUArray *grad,
														 const Cuda_GPUArray *indata, const Cuda_GPUArray *grid,
														 Cuda_MemoryPool *allocator)
{
	ingrad->dgrid = CuDnn_enforceAllocated(
		ingrad->dgrid, allocator, CUDA_GPUARRAY_SHAPE(grid), grid->ndim, grid->dtype, false
	);
	if (ingrad->dgrid == NULL) goto error_1;

	size_t trshape[3];
	trshape[0] = CUDA_GPUARRAY_SHAPE(grad)[0], trshape[1] = 2, trshape[2] = 3;

	ingrad->dtransform = CuDnn_enforceAllocated(ingrad->dtransform, allocator, trshape, 3, grad->dtype, false);
	if (ingrad->dtransform == NULL) goto error_2;

	ingrad->out = CuDnn_enforceAllocated(
		ingrad->out, allocator, CUDA_GPUARRAY_SHAPE(indata), indata->ndim, indata->dtype, false
	);
	if (ingrad->out == NULL) goto error_3;

	return true;

error_3:
	Py_DECREF(ingrad->dtransform);
error_2:
	Py_DECREF(ingrad->dgrid);
error_1:
	return false;
}


inline static bool CuDnn_Context_spatialTfBackward(CuDnn_Context *self, const Cuda_GPUArray *grad,
												   const Cuda_GPUArray *indata, const Cuda_GPUArray *grid,
												   Cuda_GPUArray *dgrid, Cuda_GPUArray *dtransform, Cuda_GPUArray *out)
{
	bool status = false;
	float alpha = 1.0f, beta = 0.0f;

	cudnnSpatialTransformerDescriptor_t stDesc;
	cudnnTensorDescriptor_t gradDesc, dataDesc, outDesc;

	if (!CuDnn_describeTensor(&gradDesc, grad))   goto error_1;
	if (!CuDnn_describeTensor(&dataDesc, indata)) goto error_2;
	if (!CuDnn_describeTensor(&outDesc, out))     goto error_3;
	if (!CuDnn_describeSpatialTf(&stDesc, out))   goto error_4;

	CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
		self->handle, stDesc, &alpha, dataDesc, indata->gpudata->ptr, &beta, outDesc, out->gpudata->ptr,
		&alpha, gradDesc, grad->gpudata->ptr, grid->gpudata->ptr, &beta, dgrid->gpudata->ptr
	), goto error_5);

	CUDNN_CHECK(cudnnSpatialTfGridGeneratorBackward(
		self->handle, stDesc, dgrid->gpudata->ptr, dtransform->gpudata->ptr
	), goto error_5);

	status = true;

error_5:
	CUDNN_ASSERT(cudnnDestroySpatialTransformerDescriptor(stDesc));
error_4:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(gradDesc));
error_1:
	return status;
}


const char CuDnn_Context_pySpatialTfBackward_doc[] = PyDoc_STR(
	"spatialTfBackward(self, grad, indata, grid, getDGrid=False, dgrid=None, dtransform=None, out=None, allocator=None)"
	" -> Union[Tuple[" CUDA_GPUARRAY_FULLNAME ", " CUDA_GPUARRAY_FULLNAME ", " CUDA_GPUARRAY_FULLNAME "], "
	"Tuple[" CUDA_GPUARRAY_FULLNAME ", " CUDA_GPUARRAY_FULLNAME "]]"
);
PyObject *CuDnn_Context_pySpatialTfBackward(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"grad", "indata", "grid", "getDGrid", "dgrid", "dtransform", "out", "allocator", NULL};

	Cuda_GPUArray *grad, *indata, *grid;
	int getDGrid = 0;
	PyObject *pydgrid = NULL, *pydtransform = NULL, *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!O!|pOOOO", (char **)kwlist,
		Cuda_GPUArray_Type, &grad, Cuda_GPUArray_Type, &indata, Cuda_GPUArray_Type, &grid,
		&getDGrid, &pydgrid, &pydtransform, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pydgrid, Cuda_GPUArray_Type, "dgrid"))           return NULL;
	if (!unpackPyOptional(&pydtransform, Cuda_GPUArray_Type, "dtransform")) return NULL;
	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))               return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator"))     return NULL;

	Cuda_GPUArray *dgrid = (Cuda_GPUArray *)pydgrid, *dtransform = (Cuda_GPUArray *)pydtransform;
	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (grad->ndim != 4 || grad->ndim != indata->ndim || grad->ndim != grid->ndim)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(grad->dtype) || grad->dtype != indata->dtype || grad->dtype != grid->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	CuDnn_SpatialTfGrad ingrad;
	ingrad.dgrid = dgrid, ingrad.dtransform = dtransform, ingrad.out = out;

	if (!CuDnn_spatialTfBackward_prepareInGrad(&ingrad, grad, indata, grid, allocator))
		return NULL;

	dgrid = ingrad.dgrid, dtransform = ingrad.dtransform, out = ingrad.out;

	if (!CuDnn_Context_spatialTfBackward((CuDnn_Context *)self, grad, indata, grid, dgrid, dtransform, out))
	{
		Py_DECREF(dgrid);
		Py_DECREF(dtransform);
		Py_DECREF(out);

		dgrid = NULL, dtransform = NULL, out = NULL;
	}

	return getDGrid ? Py_BuildValue("NNN", out, dtransform, dgrid) : Py_BuildValue("NN", out, dtransform);
}

#endif
