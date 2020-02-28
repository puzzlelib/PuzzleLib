#include "Libs.h"


typedef struct CuDnn_TransformParams
{
	size_t shape[GPUTENSOR_DIM_MAX], instrides[GPUTENSOR_DIM_MAX], outstrides[GPUTENSOR_DIM_MAX];
	size_t ndim;
}
CuDnn_TransformParams;


static bool CuDnn_Context_transform(CuDnn_Context *self, const void *dataPtr, void *outPtr,
									CuDnn_TransformParams params, size_t ndim, Cuda_DataType dtype)
{
	if (ndim < 3)
	{
		if (ndim == 2)
		{
			params.shape[2] = params.shape[1], params.shape[1] = params.shape[0], params.shape[0] = 1;

			params.instrides[2] = params.instrides[1], params.instrides[1] = params.instrides[0];
			params.outstrides[2] = params.outstrides[1], params.outstrides[1] = params.outstrides[0];
		}
		else
		{
			params.shape[2] = params.shape[0], params.shape[0] = 1, params.shape[1] = 1;

			params.instrides[2] = params.instrides[0], params.instrides[1] = params.instrides[0];
			params.outstrides[2] = params.outstrides[0], params.outstrides[1] = params.outstrides[0];
		}

		params.ndim = 3;
	}

	bool status = false;
	float alpha = 1.0f, beta = 0.0f;
	cudnnTensorDescriptor_t dataDesc, outDesc;

	if (!CuDnn_describeTensorFromShape(&dataDesc, params.shape, params.instrides, params.ndim, dtype))
		goto error_1;

	if (!CuDnn_describeTensorFromShape(&outDesc, params.shape, params.outstrides, params.ndim, dtype))
		goto error_2;

	CUDNN_CHECK(cudnnTransformTensor(self->handle, &alpha, dataDesc, dataPtr, &beta, outDesc, outPtr), goto error_3);
	status = true;

error_3:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


inline static Cuda_GPUArray *CuDnn_Context_transpose(CuDnn_Context *self, const size_t *axes, const Cuda_GPUArray *data,
													 Cuda_GPUArray *out, Cuda_MemoryPool *allocator)
{
	CuDnn_TransformParams params;
	params.ndim = data->ndim;

	const size_t *shape = CUDA_GPUARRAY_SHAPE(data);
	size_t outshape[GPUTENSOR_DIM_MAX];

	for (size_t i = 0; i < data->ndim; i += 1)
	{
		params.instrides[i] = CUDA_GPUARRAY_STRIDES(data)[i];
		params.shape[i] = shape[i], outshape[i] = shape[axes[i]];
	}

	out = CuDnn_enforceAllocated(out, allocator, outshape, params.ndim, data->dtype, false);
	if (out == NULL) return NULL;

	for (size_t i = 0; i < data->ndim; i += 1)
		params.outstrides[axes[i]] = CUDA_GPUARRAY_STRIDES(out)[i];

	if (!CuDnn_Context_transform(self, data->gpudata->ptr, out->gpudata->ptr, params, data->ndim, data->dtype))
	{
		Py_DECREF(out);
		out = NULL;
	}

	return out;
}


const char CuDnn_Context_pyTranspose_doc[] = PyDoc_STR(
	"transpose(self, data, axes=None, out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_pyTranspose(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"data", "axes", "out", "allocator", NULL};

	Cuda_GPUArray *data;
	PyObject *pyaxes = NULL, *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!|OOO", (char **)kwlist, Cuda_GPUArray_Type, &data, &pyaxes, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyaxes, &PyTuple_Type, "axes"))              return NULL;
	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidExtDim(data->ndim))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(data->dtype))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	size_t axes[GPUTENSOR_DIM_MAX];
	if (pyaxes == NULL)
	{
		for (size_t i = 0; i < data->ndim; i += 1)
			axes[i] = data->ndim - 1 - i;
	}
	else
	{
		if (!PyTuple_CheckExact(pyaxes))
		{
			PyErr_Format(PyExc_TypeError, "axes must be %s, not %s", PyTuple_Type.tp_name, Py_TYPE(pyaxes)->tp_name);
			return NULL;
		}

		size_t pylength = PyTuple_GET_SIZE(pyaxes);
		if (data->ndim != pylength)
		{
			PyErr_Format(PyExc_ValueError, "axes must be %d-tuple, not %d", (int)data->ndim, (int)pylength);
			return NULL;
		}

		bool axisSet[GPUTENSOR_DIM_MAX];
		for (size_t i = 0; i < data->ndim; i += 1)
			axisSet[i] = false;

		for (size_t i = 0; i < data->ndim; i += 1)
		{
			PyObject *pyaxis = PyTuple_GET_ITEM(pyaxes, i);

			size_t axis = PyLong_AsSize_t(pyaxis);
			if (axis == (size_t)-1 && PyErr_Occurred())
				return NULL;

			if (axisSet[axis] || axis >= data->ndim)
			{
				PyErr_SetString(PyExc_ValueError, "invalid axis in transpose");
				return NULL;
			}

			axes[i] = axis, axisSet[axis] = true;
		}
	}

	return (PyObject *)CuDnn_Context_transpose((CuDnn_Context *)self, axes, data, out, allocator);
}


const char CuDnn_Context_moveaxis_doc[] = PyDoc_STR(
	"moveaxis(self, data, src, dst, out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_moveaxis(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"data", "src", "dst", "out", "allocator", NULL};

	Cuda_GPUArray *data;
	Py_ssize_t pysrc, pydst;
	PyObject *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!nn|OO", (char **)kwlist, Cuda_GPUArray_Type, &data, &pysrc, &pydst, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidExtDim(data->ndim))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(data->dtype))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	size_t src = pysrc, dst = pydst;

	if (src >= data->ndim || dst >= data->ndim)
	{
		PyErr_SetString(PyExc_ValueError, "invalid moving axis");
		return NULL;
	}

	size_t axes[GPUTENSOR_DIM_MAX], offset = 0;
	if (src < dst)
	{
		for (size_t i = 0; i < src; i += 1) axes[offset++] = i;
		for (size_t i = src + 1; i < dst + 1; i += 1) axes[offset++] = i;
		axes[offset++] = src;
		for (size_t i = dst + 1; i < data->ndim; i += 1) axes[offset++] = i;
	}
	else
	{
		for (size_t i = 0; i < dst; i += 1) axes[offset++] = i;
		axes[offset++] = src;
		for (size_t i = dst; i < src; i += 1) axes[offset++] = i;
		for (size_t i = src + 1; i < data->ndim; i += 1) axes[offset++] = i;
	}

	return (PyObject *)CuDnn_Context_transpose((CuDnn_Context *)self, axes, data, out, allocator);
}


const char CuDnn_Context_swapaxes_doc[] = PyDoc_STR(
	"swapaxes(self, data, axis1, axis2, out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_swapaxes(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"data", "axis1", "axis2", "out", "allocator", NULL};

	Cuda_GPUArray *data;
	Py_ssize_t pyaxis1, pyaxis2;
	PyObject *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!nn|OO", (char **)kwlist, Cuda_GPUArray_Type, &data, &pyaxis1, &pyaxis2, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidExtDim(data->ndim))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(data->dtype))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	size_t axis1 = pyaxis1, axis2 = pyaxis2;

	if (axis1 >= data->ndim || axis2 >= data->ndim)
	{
		PyErr_SetString(PyExc_ValueError, "invalid moving axis");
		return NULL;
	}

	if (axis1 > axis2)
	{
		size_t temp = axis1;
		axis1 = axis2;
		axis2 = temp;
	}

	size_t axes[GPUTENSOR_DIM_MAX];
	for (size_t i = 0; i < data->ndim; i += 1) axes[i] = (i == axis1) ? axis2 : ((i == axis2) ? axis1 : i);

	return (PyObject *)CuDnn_Context_transpose((CuDnn_Context *)self, axes, data, out, allocator);
}


static bool CuDnn_depthConcat_outshape(PyListObject *tensors, size_t *outshape, size_t *outdim, Cuda_DataType *outtype)
{
	size_t ndim = 0, length = PyList_GET_SIZE(tensors);
	Cuda_DataType dtype = DTYPE_FLOAT32;

	if (length == 0)
	{
		PyErr_SetString(PyExc_ValueError, "tensor list is empty");
		return false;
	}

	for (size_t i = 0; i < length; i += 1)
	{
		PyObject *item = PyList_GET_ITEM(tensors, i);

		if (Py_TYPE(item) != Cuda_GPUArray_Type)
		{
			PyErr_SetString(PyExc_TypeError, "invalid tensor object in list");
			return false;
		}

		Cuda_GPUArray *tensor = (Cuda_GPUArray *)item;

		if (!CuDnn_isValidDim(tensor->ndim))
		{
			PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
			return false;
		}

		if (!CuDnn_isValidDtype(tensor->dtype))
		{
			PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
			return false;
		}

		const size_t *shape = CUDA_GPUARRAY_SHAPE(tensor);

		if (i == 0)
		{
			ndim = tensor->ndim, dtype = tensor->dtype;
			for (size_t j = 0; j < ndim; j += 1) outshape[j] = shape[j];
		}
		else
		{
			if (tensor->ndim != ndim || shape[0] != outshape[0])
			{
				PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
				return false;
			}

			if (tensor->dtype != dtype)
			{
				PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
				return false;
			}

			outshape[1] += shape[1];
			for (size_t j = 2; j < ndim; j += 1) outshape[j] = (outshape[j] > shape[j]) ? outshape[j] : shape[j];
		}
	}

	*outdim = ndim, *outtype = dtype;
	return true;
}


const char CuDnn_Context_depthConcat_doc[] = PyDoc_STR(
	"depthConcat(self, tensors, out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
PyObject *CuDnn_Context_depthConcat(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"tensors", "out", "allocator", NULL};

	PyListObject *tensors;
	PyObject *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|OO", (char **)kwlist, &PyList_Type, &tensors, &pyout, &pyalloc))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	size_t ndim, outshape[GPUTENSOR_DIM_MAX];
	Cuda_DataType dtype;

	if (!CuDnn_depthConcat_outshape(tensors, outshape, &ndim, &dtype))
		return NULL;

	out = CuDnn_enforceAllocated(out, allocator, outshape, ndim, dtype, true);
	size_t length = PyList_GET_SIZE(tensors), stride = 0;

	for (size_t i = 0; i < length; i += 1)
	{
		Cuda_GPUArray *tensor = (Cuda_GPUArray *)PyList_GET_ITEM(tensors, i);
		const size_t *shape = CUDA_GPUARRAY_SHAPE(tensor), *outstrides = CUDA_GPUARRAY_STRIDES(out);

		CuDnn_TransformParams params;

		params.ndim = ndim;
		for (size_t j = 0; j < ndim; j += 1)
		{
			params.shape[j] = shape[j];
			params.instrides[j] = CUDA_GPUARRAY_STRIDES(tensor)[j], params.outstrides[j] = outstrides[j];
		}

		size_t center = 0;

		for (size_t j = 2; j < ndim; j += 1)
			center += (outshape[j] - shape[j]) / 2 * outstrides[j];

		if (!CuDnn_Context_transform(
			(CuDnn_Context *)self, tensor->gpudata->ptr, (char *)out->gpudata->ptr + stride + center,
			params, ndim, dtype
		))
			goto error;

		stride += outstrides[1] * shape[1];
	}

	return (PyObject *)out;

error:
	Py_DECREF(out);
	return NULL;
}


static PyObject *CuDnn_depthSplit_prepareInGradList(PyListObject *tensors, PyObject *ingradList,
													const Cuda_GPUArray *grad, Cuda_MemoryPool *allocator)
{
	size_t length = PyList_GET_SIZE(tensors);

	if (ingradList != NULL && length != (size_t)PyList_GET_SIZE(ingradList))
	{
		PyErr_SetString(PyExc_ValueError, "invalid number of output gpuarrays");
		return NULL;
	}

	size_t inshape[GPUTENSOR_DIM_MAX];
	PyObject *out = PyList_New(length);

	for (size_t i = 0; i < length; i += 1)
	{
		PyObject *item = PyList_GET_ITEM(tensors, i);
		if (Py_TYPE(item) != Cuda_GPUArray_Type)
		{
			PyErr_SetString(PyExc_TypeError, "invalid item type in input gpuarray list");
			goto error;
		}

		Cuda_GPUArray *tensor = (Cuda_GPUArray *)item;
		if (tensor->ndim != grad->ndim)
		{
			PyErr_SetString(PyExc_ValueError, "invalid item gpuarray dims");
			goto error;
		}

		const size_t *shape = CUDA_GPUARRAY_SHAPE(tensor);

		if (i == 0) for (size_t j = 0; j < tensor->ndim; j += 1) inshape[j] = shape[j];
		else
		{
			if (shape[0] != inshape[0])
			{
				PyErr_SetString(PyExc_ValueError, "invalid item gpuarray dims");
				goto error;
			}

			inshape[1] += shape[1];
			for (size_t j = 2; j < tensor->ndim; j += 1) inshape[j] = (inshape[j] > shape[j]) ? inshape[j] : shape[j];
		}

		Cuda_GPUArray *ingrad = NULL;

		if (ingradList != NULL)
		{
			PyObject *outItem = PyList_GET_ITEM(ingradList, i);
			if (Py_TYPE(outItem) != Cuda_GPUArray_Type)
			{
				PyErr_SetString(PyExc_TypeError, "invalid item type in output gpuarray list");
				goto error;
			}

			ingrad = (Cuda_GPUArray *)outItem;
		}

		ingrad = CuDnn_enforceAllocated(ingrad, allocator, shape, tensor->ndim, tensor->dtype, false);
		if (ingrad == NULL) goto error;

		PyList_SET_ITEM(out, i, (PyObject *)ingrad);
	}

	for (size_t i = 0; i < grad->ndim; i += 1)
	{
		if (inshape[i] != CUDA_GPUARRAY_SHAPE(grad)[i])
		{
			PyErr_SetString(PyExc_ValueError, "invalid total input gpuarray dims");
			goto error;
		}
	}

	return out;

error:
	Py_DECREF(out);
	return NULL;
}


const char CuDnn_Context_depthSplit_doc[] = PyDoc_STR(
	"depthSplit(self, grad, tensors, out=None, allocator=None) -> List[" CUDA_GPUARRAY_FULLNAME "]"
);
PyObject *CuDnn_Context_depthSplit(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"grad", "tensors", "out", "allocator", NULL};

	Cuda_GPUArray *grad;
	PyListObject *tensors;
	PyObject *ingradList = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!|OO", (char **)kwlist, Cuda_GPUArray_Type, &grad, &PyList_Type, &tensors, &ingradList, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&ingradList, &PyList_Type, "out"))            return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (!CuDnn_isValidDim(grad->ndim))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	if (!CuDnn_isValidDtype(grad->dtype))
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	PyObject *out = CuDnn_depthSplit_prepareInGradList(tensors, ingradList, grad, allocator);
	if (out == NULL) return NULL;

	size_t length = PyList_GET_SIZE(out), stride = 0;
	for (size_t i = 0; i < length; i += 1)
	{
		Cuda_GPUArray *ingrad = (Cuda_GPUArray *)PyList_GET_ITEM(out, i);
		const size_t *shape = CUDA_GPUARRAY_SHAPE(ingrad), *strides = CUDA_GPUARRAY_STRIDES(grad);

		CuDnn_TransformParams params;

		params.ndim = grad->ndim;
		for (size_t j = 0; j < grad->ndim; j += 1)
		{
			params.shape[j] = shape[j];
			params.instrides[j] = strides[j], params.outstrides[j] = CUDA_GPUARRAY_STRIDES(ingrad)[j];
		}

		size_t center = 0;

		for (size_t j = 2; j < grad->ndim; j += 1)
			center += (CUDA_GPUARRAY_SHAPE(grad)[j] - shape[j]) / 2 * strides[j];

		if (!CuDnn_Context_transform(
			(CuDnn_Context *)self, (char *)grad->gpudata->ptr + stride + center, ingrad->gpudata->ptr,
			params, grad->ndim, grad->dtype
		))
			goto error;

		stride += strides[1] * shape[1];
	}

	return out;

error:
	Py_DECREF(out);
	return NULL;
}
