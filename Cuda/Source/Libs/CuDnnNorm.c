#include "Libs.h"


inline static bool CuDnn_allocateBatchNormTensors(Cuda_GPUArray **tensor1, Cuda_GPUArray **tensor2, size_t maps,
												  Cuda_MemoryPool *allocator)
{
	Cuda_ArraySpec spec;

	spec.shape[0] = maps, spec.strides[0] = Cuda_dtypeSize(DTYPE_FLOAT32);
	spec.ndim = 1, spec.size = maps, spec.dtype = DTYPE_FLOAT32, spec.contiguous = true;

	Cuda_GPUArray *tn1 = Cuda_GPUArray_newWithAllocator(allocator, NULL, &spec);
	if (tn1 == NULL)
		goto error_1;

	Cuda_GPUArray *tn2; tn2 = Cuda_GPUArray_newWithAllocator(allocator, NULL, &spec);
	if (tn2 == NULL)
		goto error_2;

	*tensor1 = tn1, *tensor2 = tn2;
	return true;

error_2:
	Py_DECREF(tn1);

error_1:
	return false;
}


inline static bool CuDnn_Context_batchNormNd(CuDnn_Context *self, const Cuda_GPUArray *data, const Cuda_GPUArray *mean,
											 const Cuda_GPUArray *var, const Cuda_GPUArray *scale,
											 const Cuda_GPUArray *bias, Cuda_GPUArray *out,
											 double epsilon, double factor, cudnnBatchNormMode_t mode,
											 Cuda_GPUArray *savemean, Cuda_GPUArray *saveinvvar)
{
	bool status = false;
	float alpha = 1.0f, beta = 0.0f;
	cudnnTensorDescriptor_t dataDesc, outDesc, scaleDesc;

	if (!CuDnn_describeTensor(&dataDesc, data))                 goto error_1;
	if (!CuDnn_describeTensor(&outDesc, out))                   goto error_2;
	if (!CuDnn_describe1DTensor(&scaleDesc, scale, data->ndim)) goto error_3;

	if (savemean != NULL)
	{
		CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
			self->handle, mode, &alpha, &beta, dataDesc, data->gpudata->ptr, outDesc, out->gpudata->ptr,
			scaleDesc, scale->gpudata->ptr, bias->gpudata->ptr, factor, mean->gpudata->ptr, var->gpudata->ptr, epsilon,
			savemean->gpudata->ptr, saveinvvar->gpudata->ptr
		), goto error_4);
	}
	else
	{
		CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
			self->handle, mode, &alpha, &beta, dataDesc, data->gpudata->ptr, outDesc, out->gpudata->ptr,
			scaleDesc, scale->gpudata->ptr, bias->gpudata->ptr, mean->gpudata->ptr, var->gpudata->ptr, epsilon
		), goto error_4);
	}

	status = true;

error_4:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(scaleDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


const char CuDnn_Context_pyBatchNormNd_doc[] = PyDoc_STR(
	"batchNormNd(self, data, mean, var, scale, bias, epsilon=1e-5, factor=1.0, test=False, "
	"mode=" CUDNN_BACKEND_NAME ".BATCHNORM_MODE_SPATIAL, out=None, allocator=None) -> "
	"Union[" CUDA_GPUARRAY_FULLNAME ", "
	"Tuple[" CUDA_GPUARRAY_FULLNAME ", " CUDA_GPUARRAY_FULLNAME ", " CUDA_GPUARRAY_FULLNAME"]]"
);
PyObject *CuDnn_Context_pyBatchNormNd(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {
		"data", "mean", "var", "scale", "bias", "epsilon", "factor", "test", "mode", "out", "allocator", NULL
	};

	Cuda_GPUArray *data, *mean, *var, *scale, *bias;
	double epsilon = 1.0e-5, factor = 1.0;
	int test = 0, mode = CUDNN_BATCHNORM_SPATIAL;
	PyObject *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!O!O!O!|ddpiOO", (char **)kwlist, Cuda_GPUArray_Type, &data, Cuda_GPUArray_Type, &mean,
		Cuda_GPUArray_Type, &var, Cuda_GPUArray_Type, &scale, Cuda_GPUArray_Type, &bias,
		&epsilon, &factor, &test, &mode, &pyout, &pyalloc
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

	size_t maps = CUDA_GPUARRAY_SHAPE(data)[1];

	if (!CuDnn_isValid1DTensor(mean, maps, DTYPE_FLOAT32, "mean"))   return NULL;
	if (!CuDnn_isValid1DTensor(var, maps, DTYPE_FLOAT32, "var"))     return NULL;
	if (!CuDnn_isValid1DTensor(scale, maps, DTYPE_FLOAT32, "scale")) return NULL;
	if (!CuDnn_isValid1DTensor(bias, maps, DTYPE_FLOAT32, "bias"))   return NULL;

	out = CuDnn_enforceAllocated(out, allocator, CUDA_GPUARRAY_SHAPE(data), data->ndim, data->dtype, false);
	if (out == NULL) return NULL;

	Cuda_GPUArray *savemean = NULL, *saveinvvar = NULL;

	if (!test)
	{
		if (!CuDnn_allocateBatchNormTensors(&savemean, &saveinvvar, maps, allocator))
		{
			Py_DECREF(out);
			return NULL;
		}
	}

	if (!CuDnn_Context_batchNormNd(
		(CuDnn_Context *)self, data, mean, var, scale, bias, out, epsilon, factor,
		(cudnnBatchNormMode_t)mode, savemean, saveinvvar
	))
	{
		Py_DECREF(out);
		out = NULL;

		if (!test)
		{
			Py_DECREF(savemean);
			Py_DECREF(saveinvvar);

			savemean = NULL, saveinvvar = NULL;
		}
	}

	return test ? (PyObject *)out : Py_BuildValue("NNN", out, savemean, saveinvvar);
}


inline static bool CuDnn_Context_batchNormNdBackward(CuDnn_Context *self, const Cuda_GPUArray *grad,
													 const Cuda_GPUArray *data, const Cuda_GPUArray *scale,
													 Cuda_GPUArray *scalegrad, Cuda_GPUArray *bgrad,
													 Cuda_GPUArray *out, double epsilon, cudnnBatchNormMode_t mode,
													 const Cuda_GPUArray *savemean, const Cuda_GPUArray *saveinvvar)
{
	bool status = false;
	float alpha = 1.0f, beta = 0.0f;
	cudnnTensorDescriptor_t gradDesc, dataDesc, outDesc, scaleDesc;

	if (!CuDnn_describeTensor(&gradDesc, grad))                 goto error_1;
	if (!CuDnn_describeTensor(&dataDesc, data))                 goto error_2;
	if (!CuDnn_describeTensor(&outDesc, out))                   goto error_3;
	if (!CuDnn_describe1DTensor(&scaleDesc, scale, grad->ndim)) goto error_4;

	const void *meanPtr; meanPtr = (savemean != NULL) ? savemean->gpudata->ptr : NULL;
	const void *invvarPtr; invvarPtr = (saveinvvar != NULL) ? saveinvvar->gpudata->ptr : NULL;

	CUDNN_CHECK(cudnnBatchNormalizationBackward(
		self->handle, mode, &alpha, &beta, &alpha, &beta, dataDesc, data->gpudata->ptr, gradDesc, grad->gpudata->ptr,
		outDesc, out->gpudata->ptr, scaleDesc, scale->gpudata->ptr,
		scalegrad->gpudata->ptr, bgrad->gpudata->ptr, epsilon, meanPtr, invvarPtr
	), goto error_5);

	status = true;

error_5:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(scaleDesc));
error_4:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(gradDesc));
error_1:
	return status;
}


const char CuDnn_Context_pyBatchNormNdBackward_doc[] = PyDoc_STR(
	"batchNormNdBackward(self, grad, data, scale, savemean=None, saveinvvar=None, epsilon=1e-5, "
	"mode=" CUDNN_BACKEND_NAME ".BATCHNORM_MODE_SPATIAL, out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_pyBatchNormNdBackward(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {
		"grad", "data", "scale", "savemean", "saveinvvar", "epsilon", "mode", "scalegrad", "bgrad",
		"out", "allocator", NULL
	};

	Cuda_GPUArray *grad, *data, *scale;
	double epsilon = 1.0e-5;
	int mode = CUDNN_BATCHNORM_SPATIAL;
	PyObject *pysavemean = NULL, *pysaveinvvar = NULL, *pyscalegrad = NULL, *pybgrad = NULL;
	PyObject *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!O!|OOdiOOOO", (char **)kwlist, Cuda_GPUArray_Type, &grad, Cuda_GPUArray_Type, &data,
		Cuda_GPUArray_Type, &scale, &pysavemean, &pysaveinvvar, &epsilon, &mode,
		&pyscalegrad, &pybgrad, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pysavemean, Cuda_GPUArray_Type, "savemean"))     return NULL;
	if (!unpackPyOptional(&pysaveinvvar, Cuda_GPUArray_Type, "saveinvvar")) return NULL;
	if (!unpackPyOptional(&pyscalegrad, Cuda_GPUArray_Type, "scalegrad"))   return NULL;
	if (!unpackPyOptional(&pybgrad, Cuda_GPUArray_Type, "bgrad"))           return NULL;
	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))               return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator"))     return NULL;

	Cuda_GPUArray *savemean = (Cuda_GPUArray *)pysavemean, *saveinvvar = (Cuda_GPUArray *)pysaveinvvar;
	Cuda_GPUArray *scalegrad = (Cuda_GPUArray *)pyscalegrad, *bgrad = (Cuda_GPUArray *)pybgrad;
	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidDim(grad->ndim) || grad->ndim != data->ndim)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(grad->dtype) || grad->dtype != data->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	size_t maps = CUDA_GPUARRAY_SHAPE(grad)[1];

	if (!CuDnn_isValid1DTensor(scale, maps, DTYPE_FLOAT32, "scale"))                                 return NULL;
	if (savemean != NULL && !CuDnn_isValid1DTensor(savemean, maps, DTYPE_FLOAT32, "savemean"))       return NULL;
	if (saveinvvar != NULL && !CuDnn_isValid1DTensor(saveinvvar, maps, DTYPE_FLOAT32, "saveinvvar")) return NULL;

	if ((scalegrad == NULL) ^ (bgrad == NULL))
	{
		PyErr_SetString(PyExc_ValueError, "invalid grad gpuarray configuration");
		return NULL;
	}

	if (scalegrad == NULL)
	{
		if (!CuDnn_allocateBatchNormTensors(&scalegrad, &bgrad, maps, allocator))
			return NULL;
	}
	else
	{
		if (!CuDnn_isValid1DTensor(scalegrad, maps, DTYPE_FLOAT32, "scalegrad")) return NULL;
		if (!CuDnn_isValid1DTensor(bgrad, maps, DTYPE_FLOAT32, "bgrad"))         return NULL;

		Py_INCREF(scalegrad);
		Py_INCREF(bgrad);
	}

	out = CuDnn_enforceAllocated(out, allocator, CUDA_GPUARRAY_SHAPE(grad), grad->ndim, grad->dtype, false);
	if (out == NULL)
	{
		Py_DECREF(scalegrad);
		Py_DECREF(bgrad);

		return NULL;
	}

	if (!CuDnn_Context_batchNormNdBackward(
		(CuDnn_Context *)self, grad, data, scale, scalegrad, bgrad, out,
		epsilon, (cudnnBatchNormMode_t)mode, savemean, saveinvvar
	))
	{
		Py_DECREF(out);
		Py_DECREF(scalegrad);
		Py_DECREF(bgrad);

		out = NULL, scalegrad = NULL, bgrad = NULL;
	}

	return Py_BuildValue("NNN", out, scalegrad, bgrad);
}


inline static bool CuDnn_describeLRN(cudnnLRNDescriptor_t *desc, unsigned N, double alpha, double beta, double K)
{
	CUDNN_CHECK(cudnnCreateLRNDescriptor(desc), goto error_1);
	CUDNN_CHECK(cudnnSetLRNDescriptor(*desc, N, alpha, beta, K), goto error_2);

	return true;

error_2:
	CUDNN_ASSERT(cudnnDestroyLRNDescriptor(*desc));
error_1:
	return false;
}


inline static bool CuDnn_allocateTempLRNBuffers(Cuda_Buffer **pTemp1, Cuda_Buffer **pTemp2, size_t size, int device,
												Cuda_MemoryPool *allocator)
{
	Cuda_Buffer *temp1 = Cuda_Buffer_newWithAllocator(size, device, allocator);
	if (temp1 == NULL) goto error_1;

	Cuda_Buffer *temp2; temp2 = Cuda_Buffer_newWithAllocator(size, device, allocator);
	if (temp2 == NULL) goto error_2;

	*pTemp1 = temp1, *pTemp2 = temp2;
	return true;

error_2:
	Py_DECREF(temp1);
error_1:
	return false;
}


inline static bool CuDnn_Context_mapLRN(CuDnn_Context *self, const Cuda_GPUArray *data, const Cuda_GPUArray *means,
										unsigned N, double alpha, double beta, double K, Cuda_GPUArray *out,
										Cuda_MemoryPool *allocator)
{
	bool status = false;
	float a = 1.0f, b = 0.0f;

	cudnnTensorDescriptor_t dataDesc;
	cudnnLRNDescriptor_t lrnDesc;

	if (!CuDnn_describeTensor(&dataDesc, data))          goto error_1;
	if (!CuDnn_describeLRN(&lrnDesc, N, alpha, beta, K)) goto error_2;

	Cuda_Buffer *temp1, *temp2;
	if (!CuDnn_allocateTempLRNBuffers(&temp1, &temp2, CUDA_GPUARRAY_NBYTES(data), data->gpudata->device, allocator))
		goto error_3;

	CUDNN_CHECK(cudnnDivisiveNormalizationForward(
		self->handle, lrnDesc, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &a, dataDesc, data->gpudata->ptr,
		means == NULL ? NULL : means->gpudata->ptr, temp1->ptr, temp2->ptr, &b, dataDesc, out->gpudata->ptr
	), goto error_4);

	status = true;

error_4:
	Py_DECREF(temp1);
	Py_DECREF(temp2);

error_3:
	CUDNN_ASSERT(cudnnDestroyLRNDescriptor(lrnDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


const char CuDnn_Context_pyMapLRN_doc[] = PyDoc_STR(
	"mapLRN(self, data, means=None, N=5, alpha=1.0e-4, beta=0.75, K=2.0, out=None, allocator=None) -> "
	CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_pyMapLRN(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"data", "means", "N", "alpha", "beta", "K", "out", "allocator", NULL};

	Cuda_GPUArray *data;
	PyObject *pymeans = NULL, *pyout = NULL, *pyalloc = NULL;

	unsigned N = 5;
	double alpha = 1.0e-4, beta = 0.75, K = 2.0;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!|OIdddOO", (char **)kwlist, Cuda_GPUArray_Type, &data, &pymeans,
		&N, &alpha, &beta, &K, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pymeans, Cuda_GPUArray_Type, "means"))       return NULL;
	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *means = (Cuda_GPUArray *)pymeans, *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidDim(data->ndim) || (means != NULL && data->ndim != means->ndim))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(data->dtype) || (means != NULL && data->dtype != means->dtype))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	out = CuDnn_enforceAllocated(out, allocator, CUDA_GPUARRAY_SHAPE(data), data->ndim, data->dtype, false);
	if (out == NULL) goto error;

	if (!CuDnn_Context_mapLRN((CuDnn_Context *)self, data, means, N, alpha, beta, K, out, allocator))
	{
		Py_DECREF(out);
		out = NULL;
	}

error:
	return (PyObject *)out;
}


inline static bool CuDnn_Context_mapLRNBackward(CuDnn_Context *self, const Cuda_GPUArray *grad,
												const Cuda_GPUArray *data, const Cuda_GPUArray *means,
												unsigned N, double alpha, double beta, double K,
												Cuda_GPUArray *out, Cuda_GPUArray *dmeans, Cuda_MemoryPool *allocator)
{
	bool status = false;
	float a = 1.0f, b = 0.0f;

	cudnnTensorDescriptor_t dataDesc;
	cudnnLRNDescriptor_t lrnDesc;

	if (!CuDnn_describeTensor(&dataDesc, data))          goto error_1;
	if (!CuDnn_describeLRN(&lrnDesc, N, alpha, beta, K)) goto error_2;

	Cuda_Buffer *temp1, *temp2;
	if (!CuDnn_allocateTempLRNBuffers(&temp1, &temp2, CUDA_GPUARRAY_NBYTES(data), data->gpudata->device, allocator))
		goto error_3;

	CUDNN_CHECK(cudnnDivisiveNormalizationBackward(
		self->handle, lrnDesc, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &a, dataDesc, data->gpudata->ptr,
		means == NULL ? NULL : means->gpudata->ptr, grad->gpudata->ptr, temp1->ptr, temp2->ptr,
		&b, dataDesc, out->gpudata->ptr, dmeans == NULL ? NULL : dmeans->gpudata->ptr
	), goto error_4);

	status = true;

error_4:
	Py_DECREF(temp1);
	Py_DECREF(temp2);

error_3:
	CUDNN_ASSERT(cudnnDestroyLRNDescriptor(lrnDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


const char CuDnn_Context_pyMapLRNBackward_doc[] = PyDoc_STR(
	"mapLRNBackward(self, data, grad, means=None, N=5, alpha=1.0e-4, beta=0.75, K=2.0, "
	"dmeans=None, out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_pyMapLRNBackward(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"data", "grad", "means", "N", "alpha", "beta", "K", "dmeans", "out", "allocator", NULL};

	Cuda_GPUArray *data, *grad;
	PyObject *pymeans = NULL, *pydmeans = NULL, *pyout = NULL, *pyalloc = NULL;

	unsigned N = 5;
	double alpha = 1.0e-4, beta = 0.75, K = 2.0;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!|OIdddOOO", (char **)kwlist, Cuda_GPUArray_Type, &data, Cuda_GPUArray_Type, &grad, &pymeans,
		&N, &alpha, &beta, &K, &pydmeans, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pymeans, Cuda_GPUArray_Type, "means"))       return NULL;
	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *means = (Cuda_GPUArray *)pymeans, *dmeans = (Cuda_GPUArray *)pydmeans, *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidDim(grad->ndim) || grad->ndim != data->ndim || (means != NULL && grad->ndim != means->ndim))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(grad->dtype) || grad->dtype != data->dtype ||
		(means != NULL && grad->dtype != means->dtype))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	out = CuDnn_enforceAllocated(out, allocator, CUDA_GPUARRAY_SHAPE(grad), grad->ndim, grad->dtype, false);
	if (out == NULL) goto error;

	if (means != NULL)
	{
		dmeans = CuDnn_enforceAllocated(
			dmeans, allocator, CUDA_GPUARRAY_SHAPE(means), means->ndim, means->dtype, false
		);
		if (dmeans == NULL)
		{
			Py_DECREF(out);
			goto error;
		}
	}

	if (!CuDnn_Context_mapLRNBackward(
		(CuDnn_Context *)self, grad, data, means, N, alpha, beta, K, out, dmeans, allocator
	))
	{
		Py_DECREF(out);
		out = NULL;

		if (means != NULL)
		{
			Py_DECREF(dmeans);
			dmeans = NULL;
		}
	}

error:
	return (means == NULL) ? (PyObject *)out : Py_BuildValue("NN", out, dmeans);
}


inline static bool CuDnn_Context_crossMapLRN(CuDnn_Context *self, const Cuda_GPUArray *data,
											 unsigned N, double alpha, double beta, double K, Cuda_GPUArray *out)
{
	bool status = false;
	float a = 1.0f, b = 0.0f;

	cudnnTensorDescriptor_t dataDesc;
	cudnnLRNDescriptor_t lrnDesc;

	if (!CuDnn_describeTensor(&dataDesc, data))          goto error_1;
	if (!CuDnn_describeLRN(&lrnDesc, N, alpha, beta, K)) goto error_2;

	CUDNN_CHECK(cudnnLRNCrossChannelForward(
		self->handle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &a, dataDesc, data->gpudata->ptr,
		&b, dataDesc, out->gpudata->ptr
	), goto error_3);

	status = true;

error_3:
	CUDNN_ASSERT(cudnnDestroyLRNDescriptor(lrnDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


const char CuDnn_Context_pyCrossMapLRN_doc[] = PyDoc_STR(
	"crossMapLRN(self, data, N=5, alpha=1.0e-4, beta=0.75, K=2.0, out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_pyCrossMapLRN(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"data", "N", "alpha", "beta", "K", "out", "allocator", NULL};

	Cuda_GPUArray *data;
	PyObject *pyout = NULL, *pyalloc = NULL;

	unsigned N = 5;
	double alpha = 1.0e-4, beta = 0.75, K = 2.0;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!|IdddOO", (char **)kwlist, Cuda_GPUArray_Type, &data, &N, &alpha, &beta, &K, &pyout, &pyalloc
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
	if (out == NULL) goto error;

	if (!CuDnn_Context_crossMapLRN((CuDnn_Context *)self, data, N, alpha, beta, K, out))
	{
		Py_DECREF(out);
		out = NULL;
	}

error:
	return (PyObject *)out;
}


inline static bool CuDnn_Context_crossMapLRNBackward(CuDnn_Context *self, const Cuda_GPUArray *grad,
													 const Cuda_GPUArray *indata, const Cuda_GPUArray *outdata,
													 unsigned N, double alpha, double beta, double K,
													 Cuda_GPUArray *out)
{
	bool status = false;
	float a = 1.0f, b = 0.0f;

	cudnnTensorDescriptor_t dataDesc;
	cudnnLRNDescriptor_t lrnDesc;

	if (!CuDnn_describeTensor(&dataDesc, indata))        goto error_1;
	if (!CuDnn_describeLRN(&lrnDesc, N, alpha, beta, K)) goto error_2;

	CUDNN_CHECK(cudnnLRNCrossChannelBackward(
		self->handle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &a, dataDesc, outdata->gpudata->ptr,
		dataDesc, grad->gpudata->ptr, dataDesc, indata->gpudata->ptr, &b, dataDesc, out->gpudata->ptr
	), goto error_3);

	status = true;

error_3:
	CUDNN_ASSERT(cudnnDestroyLRNDescriptor(lrnDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


const char CuDnn_Context_pyCrossMapLRNBackward_doc[] = PyDoc_STR(
	"crossMapLRNBackward(self, indata, outdata, grad, N=5, alpha=1.0e-4, beta=0.75, K=2.0, out=None, allocator=None)"
	" -> " CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_pyCrossMapLRNBackward(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"indata", "outdata", "grad", "N", "alpha", "beta", "K", "out", "allocator", NULL};

	Cuda_GPUArray *indata, *outdata, *grad;
	PyObject *pyout = NULL, *pyalloc = NULL;

	unsigned N = 5;
	double alpha = 1.0e-4, beta = 0.75, K = 2.0;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!O!|IdddOO", (char **)kwlist, Cuda_GPUArray_Type, &indata, Cuda_GPUArray_Type, &outdata,
		Cuda_GPUArray_Type, &grad, &N, &alpha, &beta, &K, &pyout, &pyalloc
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

	out = CuDnn_enforceAllocated(out, allocator, CUDA_GPUARRAY_SHAPE(grad), grad->ndim, grad->dtype, false);
	if (out == NULL) goto error;

	if (!CuDnn_Context_crossMapLRNBackward((CuDnn_Context *)self, grad, indata, outdata, N, alpha, beta, K, out))
	{
		Py_DECREF(out);
		out = NULL;
	}

error:
	return (PyObject *)out;
}


void CuDnnNorm_moduleInit(PyObject *m)
{
	PyModule_AddIntConstant(m, "BATCHNORM_MODE_PER_ACTIVATION", CUDNN_BATCHNORM_PER_ACTIVATION);
	PyModule_AddIntConstant(m, "BATCHNORM_MODE_SPATIAL", CUDNN_BATCHNORM_SPATIAL);
	PyModule_AddIntConstant(m, "BATCHNORM_MODE_SPATIAL_PERSISTENT", CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
}
