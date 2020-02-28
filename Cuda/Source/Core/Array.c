#include "Driver.h"


enum
{
	GPUARRAY_DISCONTIG_NDIM_LIMIT = 2
};


inline static int Cuda_fromDataType(Cuda_DataType dtype)
{
	switch (dtype)
	{
		case DTYPE_FLOAT32: return NPY_FLOAT32;
		case DTYPE_INT32:   return NPY_INT32;
		case DTYPE_INT8:    return NPY_INT8;
		case DTYPE_FLOAT16: return NPY_FLOAT16;
		case DTYPE_INT64:   return NPY_INT64;
		case DTYPE_FLOAT64: return NPY_FLOAT64;
		case DTYPE_UINT32:  return NPY_UINT32;
		case DTYPE_UINT8:   return NPY_UINT8;
		case DTYPE_UINT64:  return NPY_UINT64;
		case DTYPE_INT16:   return NPY_INT16;
		case DTYPE_UINT16:  return NPY_UINT16;
		default:            assert(false); return -1;
	}
}


inline static size_t Cuda_GPUArray_nbytes(const Cuda_GPUArray *self)
{
	return self->size * Cuda_dtypeSize(self->dtype);
}


inline static void Cuda_GPUArray_shapeToNumpy(const Cuda_GPUArray *self, npy_intp shape[GPUARRAY_NDIM_LIMIT])
{
	for (size_t i = 0; i < self->ndim; i += 1)
		shape[i] = (npy_intp)CUDA_GPUARRAY_SHAPE(self)[i];
}


inline static bool Cuda_GPUArray_isValidDim(size_t ndim)
{
	if (ndim > GPUARRAY_NDIM_LIMIT)
	{
		PyErr_Format(PyExc_ValueError, "shape number of axes overflow (limit is %d)", GPUARRAY_NDIM_LIMIT);
		return false;
	}

	return true;
}


inline static void Cuda_copyShape(size_t *outshape, size_t *outstrides, const size_t *inshape, const size_t *instrides,
								  size_t ndim)
{
	for (size_t i = 0; i < ndim; i += 1)
	{
		outshape[i] = inshape[i];
		outstrides[i] = instrides[i];
	}
}


static void Cuda_GPUArray_toSpecAsContiguous(const Cuda_GPUArray *self, Cuda_ArraySpec *spec)
{
	assert(self->ndim <= GPUARRAY_NDIM_LIMIT);

	if (self->contiguous)
		Cuda_copyShape(spec->shape, spec->strides, CUDA_GPUARRAY_SHAPE(self), CUDA_GPUARRAY_STRIDES(self), self->ndim);
	else
		Cuda_fillShapeAsContiguous(
			spec->shape, spec->strides, CUDA_GPUARRAY_SHAPE(self), self->ndim, Cuda_dtypeSize(self->dtype)
		);

	spec->ndim = self->ndim;
	spec->size = self->size;

	spec->dtype = self->dtype;
	spec->contiguous = true;
}


inline static bool Cuda_numpyToSpecAsContiguous(PyArrayObject *ary, Cuda_ArraySpec *spec)
{
	size_t ndim = PyArray_NDIM(ary);
	if (!Cuda_GPUArray_isValidDim(ndim))
		return false;

	Cuda_DataType dtype = Cuda_toDataType(PyArray_DTYPE(ary)->type_num);
	if (dtype == DTYPE_INVALID)
		return false;

	npy_intp *npshape = PyArray_DIMS(ary);
	size_t lastdim = 1, laststride = Cuda_dtypeSize(dtype);

	for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; i -= 1)
	{
		spec->shape[i] = npshape[i];
		spec->strides[i] = lastdim * laststride;

		lastdim = npshape[i], laststride = spec->strides[i];
	}

	spec->ndim = ndim;
	spec->size = PyArray_SIZE(ary);

	spec->dtype = dtype;
	spec->contiguous = true;

	return true;
}


static bool Cuda_pyToSpec(PyTupleObject *pyshape, Cuda_DataType dtype, bool contiguous, size_t currsize,
						  Cuda_ArraySpec *spec)
{
	size_t ndim = PyTuple_GET_SIZE(pyshape);
	if (!Cuda_GPUArray_isValidDim(ndim))
		return false;

	spec->dtype = dtype;
	spec->contiguous = contiguous;

	size_t size = 1;
	size_t lastdim = 1, laststride = Cuda_dtypeSize(dtype);

	ptrdiff_t undefidx = -1;

	for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; i -= 1)
	{
		PyObject *item = PyTuple_GET_ITEM(pyshape, i);
		Py_ssize_t dim = PyLong_AsSsize_t(item);

		if (dim <= 0)
		{
			if (PyErr_Occurred())
				return false;

			else if (dim != -1 || currsize == 0)
			{
				PyErr_Format(PyExc_ValueError, "shape item #%d is invalid", (int)(i + 1));
				return false;
			}

			undefidx = i;
			break;
		}

		spec->shape[i] = dim;
		spec->strides[i] = lastdim * laststride;
		size *= dim;

		lastdim = dim, laststride = spec->strides[i];
	}

	if (undefidx >= 0)
	{
		for (ptrdiff_t i = 0; i < undefidx; i += 1)
		{
			PyObject *item = PyTuple_GET_ITEM(pyshape, i);
			Py_ssize_t dim = PyLong_AsSsize_t(item);

			if (dim <= 0)
			{
				if (PyErr_Occurred())
					return false;

				if (dim != -1)
					PyErr_Format(PyExc_ValueError, "shape item #%d is invalid", (int)(i + 1));
				else
					PyErr_SetString(PyExc_ValueError, "multiple undefined axes");

				return false;
			}

			spec->shape[i] = dim;
			size *= dim;
		}

		if (currsize % size != 0)
		{
			PyErr_SetString(PyExc_ValueError, "shape size is not divisible by undefined axis");
			return false;
		}

		spec->shape[undefidx] = currsize / size;
		spec->strides[undefidx] = lastdim * laststride;
		size = currsize;

		for (ptrdiff_t i = undefidx - 1; i >= 0; i -= 1)
			spec->strides[i] = spec->strides[i + 1] * spec->shape[i + 1];
	}

	spec->ndim = ndim;
	spec->size = size;

	return true;
}


static Cuda_Buffer *Cuda_GPUArray_acquireData(Cuda_MemoryPool *allocator, Cuda_Buffer *gpudata, size_t size)
{
	if (gpudata != NULL)
	{
		if (gpudata->size < size)
		{
			PyErr_Format(
				PyExc_ValueError, "required gpudata of at least %" PRIuMAX " bytes, got %" PRIuMAX, size, gpudata->size
			);
			return NULL;
		}

		Py_INCREF(gpudata);
	}
	else
	{
		if (allocator != NULL)
		{
#if defined(TRACE_CUDA_DRIVER)
			fprintf(
				stderr,
				"[" CUDA_GPUARRAY_OBJNAME "] Acquiring gpudata of size %" PRIuMAX " from MEMORY POOL 0x%" PRIXMAX
				" ...\n", size, (size_t)allocator
			);
#endif

			gpudata = Cuda_MemoryPool_allocate(allocator, size);
		}
		else
		{
#if defined(TRACE_CUDA_DRIVER)
			fprintf(
				stderr, "[" CUDA_GPUARRAY_OBJNAME "] Acquiring gpudata of size %" PRIuMAX " from DEVICE ...\n", size
			);
#endif

			gpudata = Cuda_Driver_allocate(size);
		}
	}

	return gpudata;
}


static void Cuda_GPUArray_init(Cuda_GPUArray *self, Cuda_Buffer *gpudata, const Cuda_ArraySpec *spec)
{
	self->ndim = spec->ndim;
	self->size = spec->size;

	self->gpudata = gpudata;

	self->dtype = spec->dtype;
	self->contiguous = spec->contiguous;

	Cuda_copyShape(CUDA_GPUARRAY_SHAPE(self), CUDA_GPUARRAY_STRIDES(self), spec->shape, spec->strides, spec->ndim);

#if defined(TRACE_CUDA_DRIVER)
	fprintf(
		stderr,
		"[" CUDA_GPUARRAY_OBJNAME "] (0x%" PRIXMAX ") Allocated gpuarray of size %" PRIuMAX " with gpudata 0x%" PRIXMAX
		"\n", (size_t)self, self->size, (size_t)self->gpudata
	);
#endif
}


static Cuda_GPUArray *Cuda_GPUArray_new(Cuda_Buffer *gpudata, const Cuda_ArraySpec *spec)
{
	Cuda_GPUArray *self = PyObject_NEW_VAR(Cuda_GPUArray, Cuda_GPUArray_Type, spec->ndim * 2);
	if (self == NULL)
		return NULL;

	Cuda_GPUArray_init(self, gpudata, spec);
	return self;
}


Cuda_GPUArray *Cuda_GPUArray_newWithAllocator(Cuda_MemoryPool *allocator, Cuda_Buffer *gpudata,
											  const Cuda_ArraySpec *spec)
{
	gpudata = Cuda_GPUArray_acquireData(allocator, gpudata, spec->size * Cuda_dtypeSize(spec->dtype));
	if (gpudata == NULL)
		goto error_1;

	Cuda_GPUArray *self; self = Cuda_GPUArray_new(gpudata, spec);
	if (self == NULL)
		goto error_2;

	return self;

error_2:
	Py_DECREF(gpudata);

error_1:
	return NULL;
}


static PyObject *Cuda_GPUArray_pyNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)type;
	const char *kwlist[] = {"shape", "dtype", "allocator", "gpudata", NULL};

	PyTupleObject *pyshape;
	PyArray_Descr *pytype;

	PyObject *pyalloc = NULL;
	Cuda_Buffer *gpudata = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O&|OO!", (char **)kwlist, &PyTuple_Type, &pyshape, PyArray_DescrConverter2, &pytype,
		&pyalloc, Cuda_Buffer_Type, &gpudata
	))
		return NULL;

	if (pytype == NULL)
		goto error_1;

	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator"))
		goto error_2;

	Cuda_MemoryPool *allocator; allocator = (Cuda_MemoryPool *)pyalloc;

	Cuda_DataType dtype; dtype = Cuda_toDataType(pytype->type_num);
	if (dtype == DTYPE_INVALID)
		goto error_2;

	Cuda_ArraySpec spec;
	if (!Cuda_pyToSpec(pyshape, dtype, true, 0, &spec))
		goto error_2;

	Cuda_GPUArray *self; self = Cuda_GPUArray_newWithAllocator(allocator, gpudata, &spec);
	if (self == NULL)
		goto error_2;

	Py_DECREF(pytype);
	return (PyObject *)self;

error_2:
	Py_DECREF(pytype);

error_1:
	return NULL;
}


static void Cuda_GPUArray_dealloc(PyObject *self)
{
	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;

#if defined(TRACE_CUDA_DRIVER)
	fprintf(
		stderr, "[" CUDA_GPUARRAY_OBJNAME "] (0x%" PRIXMAX ") Deallocated gpuarray with gpudata 0x%" PRIXMAX "\n",
		(size_t)pyary, (size_t)pyary->gpudata
	);
#endif

	Py_DECREF(pyary->gpudata);
	PyObject_FREE(self);
}


typedef enum Cuda_ReshapeState
{
	Cuda_ReshapeState_equal,
	Cuda_ReshapeState_leftMoved,
	Cuda_ReshapeState_rightMoved
}
Cuda_ReshapeState;


static bool Cuda_GPUArray_reshapeDiscontig(const Cuda_GPUArray *self, Cuda_ArraySpec *spec)
{
	size_t lhsdim = 1, rhsdim = 1;
	ptrdiff_t lhs = self->ndim - 1, rhs = spec->ndim - 1;

	Cuda_ReshapeState state = Cuda_ReshapeState_equal;
	const size_t *shape = CUDA_GPUARRAY_SHAPE(self), *strides = CUDA_GPUARRAY_STRIDES(self);

	while (rhs >= 0)
	{
		if (state == Cuda_ReshapeState_equal)
		{
			spec->strides[rhs] = strides[lhs];
			lhsdim = shape[lhs], rhsdim = spec->shape[rhs];
		}
		else if (state == Cuda_ReshapeState_rightMoved)
		{
			spec->strides[rhs] = spec->strides[rhs + 1] * spec->shape[rhs + 1];
			rhsdim *= spec->shape[rhs];
		}
		else
		{
			if (strides[lhs] != strides[lhs + 1] * shape[lhs + 1])
			{
				PyErr_Format(PyExc_ValueError, "gpuarray axis %d is not contiguous", (int)lhs);
				return false;
			}
			lhsdim *= shape[lhs];
		}

		if (lhsdim == rhsdim)
		{
			state = Cuda_ReshapeState_equal;
			lhs -= 1;
			rhs -= 1;
		}
		else if (lhsdim > rhsdim)
		{
			state = Cuda_ReshapeState_rightMoved;
			rhs -= 1;
		}
		else
		{
			state = Cuda_ReshapeState_leftMoved;
			lhs -= 1;
		}
	}

	return true;
}


PyDoc_STRVAR(Cuda_GPUArray_reshape_doc, "reshape(self, newshape, *args) -> " CUDA_GPUARRAY_FULLNAME);
static PyObject *Cuda_GPUArray_reshape(PyObject *self, PyObject *args)
{
	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;

	if (PyTuple_GET_SIZE(args) == 0)
	{
		PyErr_SetString(PyExc_ValueError, "function takes at least 1 argument");
		return NULL;
	}

	PyObject *arg = PyTuple_GET_ITEM(args, 0);
	PyTupleObject *pyshape = (PyTupleObject *)(PyTuple_CheckExact(arg) ? arg : args);

	Cuda_ArraySpec spec;
	if (!Cuda_pyToSpec(pyshape, pyary->dtype, pyary->contiguous, pyary->size, &spec))
		return NULL;

	if (spec.size != pyary->size)
	{
		PyErr_Format(PyExc_ValueError, "new shape must have same size %" PRIuMAX, pyary->size);
		return NULL;
	}

	if (!pyary->contiguous && !Cuda_GPUArray_reshapeDiscontig(pyary, &spec))
		return NULL;

	Cuda_GPUArray *reshaped = Cuda_GPUArray_new(pyary->gpudata, &spec);
	if (reshaped == NULL)
		return NULL;

	Py_INCREF(pyary->gpudata);
	return (PyObject *)reshaped;
}


static void Cuda_GPUArray_reinterpretShape(const Cuda_GPUArray *self, Cuda_DataType dtype, Cuda_ArraySpec *spec)
{
	size_t itemsize = Cuda_dtypeSize(dtype);

	spec->ndim = self->ndim;
	spec->size = Cuda_GPUArray_nbytes(self) / itemsize;

	spec->dtype = dtype;
	spec->contiguous = self->contiguous;

	Cuda_copyShape(spec->shape, spec->strides, CUDA_GPUARRAY_SHAPE(self), CUDA_GPUARRAY_STRIDES(self), self->ndim - 1);

	spec->shape[self->ndim - 1] = CUDA_GPUARRAY_SHAPE(self)[self->ndim - 1] * Cuda_dtypeSize(self->dtype) / itemsize;
	spec->strides[self->ndim - 1] = itemsize;
}


PyDoc_STRVAR(Cuda_GPUArray_view_doc, "view(self, dtype) -> " CUDA_GPUARRAY_FULLNAME);
static PyObject *Cuda_GPUArray_view(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"dtype", NULL};

	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;
	Cuda_GPUArray *view = NULL;

	if (pyary->ndim == 0)
	{
		PyErr_SetString(PyExc_ValueError, "cannot view 0d gpuarray");
		goto error_1;
	}

	PyArray_Descr *pytype;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", (char **)kwlist, PyArray_DescrConverter2, &pytype))
		goto error_1;

	if (pytype == NULL)
		goto error_1;

	Cuda_DataType dtype; dtype = Cuda_toDataType(pytype->type_num);
	if (dtype == DTYPE_INVALID)
		goto error_2;

	size_t lastdim, itemsize;
	lastdim = CUDA_GPUARRAY_SHAPE(pyary)[pyary->ndim - 1] * Cuda_dtypeSize(pyary->dtype);
	itemsize = Cuda_dtypeSize(dtype);

	if (lastdim % itemsize != 0)
	{
		PyErr_Format(PyExc_ValueError, "gpuarray last axis is not divisible by item size %d", itemsize);
		goto error_2;
	}

	Cuda_ArraySpec spec;
	Cuda_GPUArray_reinterpretShape(pyary, dtype, &spec);

	view = Cuda_GPUArray_new(pyary->gpudata, &spec);
	if (view == NULL)
		goto error_2;

	Py_INCREF(pyary->gpudata);

error_2:
	Py_DECREF(pytype);

error_1:
	return (PyObject *)view;
}


PyDoc_STRVAR(Cuda_GPUArray_ravel_doc, "ravel(self) -> " CUDA_GPUARRAY_FULLNAME);
static PyObject *Cuda_GPUArray_ravel(PyObject *self, PyObject *args)
{
	(void)args;

	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;
	if (!pyary->contiguous)
	{
		PyErr_SetString(PyExc_ValueError, "gpuarray is not contiguous");
		return NULL;
	}

	Cuda_ArraySpec spec;

	spec.shape[0] = pyary->size;
	spec.strides[0] = Cuda_dtypeSize(pyary->dtype);

	spec.ndim = 1;
	spec.size = pyary->size;

	spec.dtype = pyary->dtype;
	spec.contiguous = true;

	Cuda_GPUArray *raveled = Cuda_GPUArray_new(pyary->gpudata, &spec);
	if (raveled == NULL)
		return NULL;

	Py_INCREF(pyary->gpudata);
	return (PyObject *)raveled;
}


typedef struct Cuda_MergedArraySpec
{
	size_t shape[GPUARRAY_DISCONTIG_NDIM_LIMIT + 1], strides[GPUARRAY_DISCONTIG_NDIM_LIMIT + 1];
	size_t ndim;

	void *ptr;
	bool isDevice;
}
Cuda_MergedArraySpec;


static bool Cuda_mergeAxesForNumpy(PyArrayObject *ary, Cuda_MergedArraySpec *spec)
{
	size_t ndim = 0, dim = 1;

	size_t npNDim = PyArray_NDIM(ary);
	npy_intp *shape = PyArray_SHAPE(ary), *strides = PyArray_STRIDES(ary);

	for (size_t i = 0; i < npNDim - 1; i += 1)
	{
		dim *= shape[i];

		if (strides[i] != shape[i + 1] * strides[i + 1])
		{
			spec->shape[ndim] = dim;
			spec->strides[ndim] = strides[i];

			ndim += 1;
			if (ndim > GPUARRAY_DISCONTIG_NDIM_LIMIT)
				goto error;

			dim = 1;
		}
	}

	if (npNDim > 0)
	{
		spec->shape[ndim] = dim * shape[npNDim - 1];
		spec->strides[ndim] = strides[npNDim - 1];
	}
	else
	{
		spec->shape[ndim] = dim;
		spec->strides[ndim] = PyArray_DTYPE(ary)->elsize;
	}

	spec->ndim = ndim + 1;

	spec->isDevice = false;
	spec->ptr = PyArray_DATA(ary);

	return true;

error:
	PyErr_Format(
		PyExc_ValueError, "%s has %" PRIuMAX " discontiguous axes (limit is %d)",
		Py_TYPE(ary)->tp_name, ndim, GPUARRAY_DISCONTIG_NDIM_LIMIT
	);
	return false;
}


static bool Cuda_GPUArray_mergeAxes(const Cuda_GPUArray *self, Cuda_MergedArraySpec *spec)
{
	size_t ndim = 0, dim = 1;
	const size_t *shape = CUDA_GPUARRAY_SHAPE(self), *strides = CUDA_GPUARRAY_STRIDES(self);

	for (size_t i = 0; i < self->ndim - 1; i += 1)
	{
		dim *= shape[i];

		if (strides[i] != shape[i + 1] * strides[i + 1])
		{
			spec->shape[ndim] = dim;
			spec->strides[ndim] = strides[i];

			ndim += 1;
			if (ndim > GPUARRAY_DISCONTIG_NDIM_LIMIT)
				goto error;

			dim = 1;
		}
	}

	if (self->ndim > 0)
	{
		spec->shape[ndim] = dim * shape[self->ndim - 1];
		spec->strides[ndim] = strides[self->ndim - 1];
	}
	else
	{
		spec->shape[ndim] = dim;
		spec->strides[ndim] = Cuda_dtypeSize(self->dtype);
	}

	spec->ndim = ndim + 1;

	spec->isDevice = true;
	spec->ptr = self->gpudata->ptr;

	return true;

error:
	PyErr_Format(
		PyExc_ValueError, "gpuarray has %" PRIuMAX " discontiguous axes (limit is %d)",
		ndim, GPUARRAY_DISCONTIG_NDIM_LIMIT
	);
	return false;
}


static bool Cuda_GPUArray_pyMergeAxes(PyObject *ary, Cuda_MergedArraySpec *spec)
{
	assert(Py_TYPE(ary) == Cuda_GPUArray_Type || PyArray_CheckExact(ary));

	if (Py_TYPE(ary) == Cuda_GPUArray_Type)
		return Cuda_GPUArray_mergeAxes((Cuda_GPUArray *)ary, spec);
	else
		return Cuda_mergeAxesForNumpy((PyArrayObject *)ary, spec);
}


static void Cuda_GPUArray_promoteMergedArraySpec(Cuda_MergedArraySpec *spec, Cuda_MergedArraySpec *base)
{
	assert(spec->ndim < base->ndim);

	if (spec->ndim == 1)
		Cuda_fillShapeAsContiguous(spec->shape, spec->strides, base->shape, base->ndim, base->strides[base->ndim - 1]);

	else
	{
		assert(spec->ndim == GPUARRAY_DISCONTIG_NDIM_LIMIT && base->ndim == GPUARRAY_DISCONTIG_NDIM_LIMIT + 1);

		size_t stride = spec->strides[0];
		ptrdiff_t stridedim = (spec->shape[0] == base->shape[0]) ? 0 : 1;

		size_t lastdim = 1, laststride = base->strides[GPUARRAY_DISCONTIG_NDIM_LIMIT];

		for (ptrdiff_t i = GPUARRAY_DISCONTIG_NDIM_LIMIT; i >= 0; i -= 1)
		{
			spec->shape[i] = base->shape[i];
			spec->strides[i] = (i == stridedim) ? stride : lastdim * laststride;

			lastdim = spec->shape[i], laststride = spec->strides[i];
		}
	}

	spec->ndim = base->ndim;
}


static bool Cuda_GPUArray_memcpyDiscontig(PyObject *pysrc, PyObject *pydst)
{
	Cuda_MergedArraySpec srcSpec;
	if (!Cuda_GPUArray_pyMergeAxes(pysrc, &srcSpec))
		return false;

	Cuda_MergedArraySpec dstSpec;
	if (!Cuda_GPUArray_pyMergeAxes(pydst, &dstSpec))
		return false;

	assert(srcSpec.ndim > 1 || dstSpec.ndim > 1);

	if (srcSpec.ndim > dstSpec.ndim)
		Cuda_GPUArray_promoteMergedArraySpec(&dstSpec, &srcSpec);
	else if (srcSpec.ndim < dstSpec.ndim)
		Cuda_GPUArray_promoteMergedArraySpec(&srcSpec, &dstSpec);

	enum cudaMemcpyKind kind;

	if (dstSpec.isDevice)
		kind = srcSpec.isDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
	else
		kind = srcSpec.isDevice ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;

	if (srcSpec.ndim == 2)
		CUDA_CHECK(cudaMemcpy2D(
			dstSpec.ptr, dstSpec.strides[0], srcSpec.ptr, srcSpec.strides[0],
			srcSpec.shape[1] * srcSpec.strides[srcSpec.ndim - 1], srcSpec.shape[0], kind
		), return false);

	else
	{
		struct cudaMemcpy3DParms memcpy;

		memcpy.dstArray = NULL, memcpy.srcArray = NULL;
		memcpy.dstPos = make_cudaPos(0, 0, 0), memcpy.srcPos = make_cudaPos(0, 0, 0);

		assert(dstSpec.strides[1] % dstSpec.strides[2] == 0);
		assert(dstSpec.strides[0] % dstSpec.strides[1] == 0);

		memcpy.dstPtr = make_cudaPitchedPtr(
			dstSpec.ptr, dstSpec.strides[1], dstSpec.strides[1] / dstSpec.strides[2],
			dstSpec.strides[0] / dstSpec.strides[1]
		);

		assert(srcSpec.strides[1] % srcSpec.strides[2] == 0);
		assert(srcSpec.strides[0] % srcSpec.strides[1] == 0);

		memcpy.srcPtr = make_cudaPitchedPtr(
			srcSpec.ptr, srcSpec.strides[1], srcSpec.strides[1] / srcSpec.strides[2],
			srcSpec.strides[0] / srcSpec.strides[1]
		);

		memcpy.extent = make_cudaExtent(
			srcSpec.shape[2] * srcSpec.strides[srcSpec.ndim - 1], srcSpec.shape[1], srcSpec.shape[0]
		);

		memcpy.kind = kind;
		CUDA_CHECK(cudaMemcpy3D(&memcpy), return false);
	}

	return true;
}


PyDoc_STRVAR(Cuda_GPUArray_get_doc, "get(self) -> numpy.ndarray");
static PyObject *Cuda_GPUArray_get(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;

	npy_intp npshape[GPUARRAY_NDIM_LIMIT];
	Cuda_GPUArray_shapeToNumpy(pyary, npshape);

	int typenum = Cuda_fromDataType(pyary->dtype);

	PyArrayObject *ary = (PyArrayObject *)PyArray_EMPTY((int)pyary->ndim, npshape, typenum, 0);
	if (ary == NULL)
		goto error_1;

	if (pyary->contiguous)
		CU_CHECK(cuMemcpyDtoH(
			PyArray_DATA(ary), (CUdeviceptr)pyary->gpudata->ptr, Cuda_GPUArray_nbytes(pyary)
		), goto error_2);

	else
	{
		if (!Cuda_GPUArray_memcpyDiscontig(self, (PyObject *)ary))
			goto error_2;
	}

	return (PyObject *)ary;

error_2:
	Py_DECREF(ary);

error_1:
	return NULL;
}


inline static bool Cuda_GPUArray_compareShapeWithNumpy(const Cuda_GPUArray *self, PyArrayObject *other)
{
	npy_intp shape[GPUARRAY_NDIM_LIMIT];
	Cuda_GPUArray_shapeToNumpy(self, shape);

	int ndim = (int)self->ndim;

	if (ndim != PyArray_NDIM(other) || !PyArray_CompareLists(shape, PyArray_SHAPE(other), ndim))
		return false;

	return true;
}


static bool Cuda_GPUArray_setWithNumpy(Cuda_GPUArray *self, PyArrayObject *other)
{
	if (Cuda_fromDataType(self->dtype) != PyArray_DTYPE(other)->type_num)
	{
		PyErr_SetString(PyExc_ValueError, "gpuarray and input array types are not equal");
		return false;
	}

	if (!Cuda_GPUArray_compareShapeWithNumpy(self, other))
	{
		PyErr_SetString(PyExc_ValueError, "gpuarray and input array shapes are not equal");
		return false;
	}

	if (self->contiguous && PyArray_IS_C_CONTIGUOUS(other))
	{
		CU_CHECK(
			cuMemcpyHtoD((CUdeviceptr)self->gpudata->ptr, PyArray_DATA(other), Cuda_GPUArray_nbytes(self)), return false
		);
	}
	else
	{
		if (!Cuda_GPUArray_memcpyDiscontig((PyObject *)other, (PyObject *)self))
			return false;
	}

	return true;
}


inline static bool Cuda_GPUArray_compareShapes(const Cuda_GPUArray *self, const Cuda_GPUArray *other)
{
	if (self->ndim != other->ndim)
		return false;

	for (size_t i = 0; i < self->ndim; i += 1)
	{
		if (CUDA_GPUARRAY_SHAPE(self)[i] != CUDA_GPUARRAY_SHAPE(other)[i])
			return false;
	}

	return true;
}


static bool Cuda_GPUArray_set(Cuda_GPUArray *self, const Cuda_GPUArray *other)
{
	if (self->dtype != other->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "gpuarray data types are not equal");
		return false;
	}

	if (!Cuda_GPUArray_compareShapes(self, other))
	{
		PyErr_SetString(PyExc_ValueError, "gpuarray shapes are not equal");
		return false;
	}

	if (self->contiguous && other->contiguous)
	{
		CU_CHECK(
			cuMemcpyDtoD((CUdeviceptr)self->gpudata->ptr, (CUdeviceptr)other->gpudata->ptr, Cuda_GPUArray_nbytes(other)
		), return false);
	}
	else
	{
		if (!Cuda_GPUArray_memcpyDiscontig((PyObject *)other, (PyObject *)self))
			return false;
	}

	return true;
}


PyDoc_STRVAR(Cuda_GPUArray_pySet_doc, "set(self, ary)");
static PyObject *Cuda_GPUArray_pySet(PyObject *self, PyObject *args, PyObject *kwds)
{
	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;

	const char *kwlist[] = {"ary", NULL};
	PyObject *pyother;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char **)kwlist, &pyother))
		return NULL;

	if (Py_TYPE(pyother) == &PyArray_Type)
	{
		if (!Cuda_GPUArray_setWithNumpy(pyary, (PyArrayObject *)pyother))
			return NULL;
	}
	else if (Py_TYPE(pyother) == Cuda_GPUArray_Type)
	{
		if (!Cuda_GPUArray_set(pyary, (Cuda_GPUArray *)pyother))
			return NULL;
	}
	else
	{
		PyErr_Format(
			PyExc_TypeError, "input gpuarray must be %s or %s, not %s",
			(&PyArray_Type)->tp_name, Cuda_GPUArray_Type->tp_name, Py_TYPE(pyother)->tp_name
		);
		return NULL;
	}

	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_GPUArray_copy_doc, "copy(self, allocator=None) -> " CUDA_GPUARRAY_FULLNAME);
static PyObject *Cuda_GPUArray_copy(PyObject *self, PyObject *args, PyObject *kwds)
{
	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;

	const char *kwlist[] = {"allocator", NULL};
	PyObject *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", (char **)kwlist, &pyalloc))
		return NULL;

	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator"))
		return NULL;

	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	Cuda_ArraySpec spec;
	Cuda_GPUArray_toSpecAsContiguous(pyary, &spec);

	Cuda_GPUArray *dst = Cuda_GPUArray_newWithAllocator(allocator, NULL, &spec);
	if (dst == NULL)
		return NULL;

	if (pyary->contiguous)
		CU_CHECK(cuMemcpyDtoD(
			(CUdeviceptr)dst->gpudata->ptr, (CUdeviceptr)pyary->gpudata->ptr, Cuda_GPUArray_nbytes(pyary)
		), goto error);

	else
	{
		if (!Cuda_GPUArray_memcpyDiscontig(self, (PyObject *)dst))
			goto error;
	}

	return (PyObject *)dst;

error:
	Py_DECREF(dst);
	return NULL;
}


static PyObject *Cuda_GPUArray_toString(PyObject *self)
{
	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;
	PyObject *str = NULL;

	if (pyary->gpudata->ptr == NULL)
	{
		PyErr_SetString(Cuda_Error, "buffer is freed");
		goto error_1;
	}

	int device;
	CUDA_CHECK(cudaGetDevice(&device), goto error_1);
	CUDA_CHECK(cudaSetDevice(pyary->gpudata->device), goto error_1);

	PyObject *ary; ary = Cuda_GPUArray_get(self, NULL);
	if (ary == NULL)
		goto error_2;

	str = PyObject_Str(ary);
	Py_DECREF(ary);

error_2:
	if (pyary->gpudata->device != device)
		CUDA_ASSERT(cudaSetDevice(device));

error_1:
	return str;
}


PyDoc_STRVAR(Cuda_GPUArray_empty_doc, "empty(shape, dtype, allocator=None, gpudata=None) -> " CUDA_GPUARRAY_FULLNAME);
static PyObject *Cuda_GPUArray_empty(PyObject *type, PyObject *args, PyObject *kwds)
{
	(void)type;
	return Cuda_GPUArray_pyNew(Cuda_GPUArray_Type, args, kwds);
}


PyDoc_STRVAR(Cuda_GPUArray_zeros_doc, "zeros(shape, dtype, allocator=None, gpudata=None) -> " CUDA_GPUARRAY_FULLNAME);
static PyObject *Cuda_GPUArray_zeros(PyObject *type, PyObject *args, PyObject *kwds)
{
	Cuda_GPUArray *pyary = (Cuda_GPUArray *)Cuda_GPUArray_empty(type, args, kwds);
	if (pyary == NULL)
		return NULL;

	CU_CHECK(cuMemsetD8((CUdeviceptr)pyary->gpudata->ptr, 0, Cuda_GPUArray_nbytes(pyary)), goto error);
	return (PyObject *)pyary;

error:
	Py_DECREF(pyary);
	return NULL;
}


static Cuda_GPUArray *Cuda_GPUArray_emptyLike(PyObject *pyary, Cuda_MemoryPool *allocator)
{
	Cuda_ArraySpec spec;

	if (PyArray_CheckExact(pyary))
	{
		PyArrayObject *ary = (PyArrayObject *)pyary;

		if (!Cuda_numpyToSpecAsContiguous(ary, &spec))
			return NULL;
	}
	else if (Py_TYPE(pyary) == Cuda_GPUArray_Type)
	{
		Cuda_GPUArray *ary = (Cuda_GPUArray *)pyary;
		Cuda_GPUArray_toSpecAsContiguous(ary, &spec);
	}
	else
	{
		PyErr_Format(
			PyExc_TypeError, "input array must be %s or %s, not %s",
			Cuda_GPUArray_Type->tp_name, (&PyArray_Type)->tp_name, Py_TYPE(pyary)->tp_name
		);
		return NULL;
	}

	Cuda_GPUArray *newary = Cuda_GPUArray_newWithAllocator(allocator, NULL, &spec);
	if (newary == NULL)
		return NULL;

	return newary;
}


PyDoc_STRVAR(Cuda_GPUArray_pyEmptyLike_doc, "emptyLike(ary, allocator=None) -> " CUDA_GPUARRAY_FULLNAME);
static PyObject *Cuda_GPUArray_pyEmptyLike(PyObject *type, PyObject *args, PyObject *kwds)
{
	(void)type;
	const char *kwlist[] = {"", "allocator", NULL};

	PyObject *ary, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", (char **)kwlist, &ary, &pyalloc))
		return NULL;

	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator"))
		return NULL;

	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;
	return (PyObject *)Cuda_GPUArray_emptyLike(ary, allocator);
}


PyDoc_STRVAR(Cuda_GPUArray_zerosLike_doc, "zerosLike(ary, allocator=None) -> " CUDA_GPUARRAY_FULLNAME);
static PyObject *Cuda_GPUArray_zerosLike(PyObject *type, PyObject *args, PyObject *kwds)
{
	(void)type;
	const char *kwlist[] = {"", "allocator", NULL};

	PyObject *ary, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$O", (char **)kwlist, &ary, &pyalloc))
		return NULL;

	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator"))
		return NULL;

	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	Cuda_GPUArray *pyary = Cuda_GPUArray_emptyLike((PyObject *)ary, allocator);
	if (pyary == NULL)
		return NULL;

	CU_CHECK(cuMemsetD8((CUdeviceptr)pyary->gpudata->ptr, 0, Cuda_GPUArray_nbytes(pyary)), goto error);
	return (PyObject *)pyary;

error:
	Py_DECREF(pyary);
	return NULL;
}


PyDoc_STRVAR(Cuda_GPUArray_toGpu_doc, "toGpu(ary, allocator=None) -> " CUDA_GPUARRAY_FULLNAME);
static PyObject *Cuda_GPUArray_toGpu(PyObject *type, PyObject *args, PyObject *kwds)
{
	(void)type;
	const char *kwlist[] = {"", "allocator", NULL};

	PyArrayObject *ary;
	PyObject *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|$O", (char **)kwlist, &PyArray_Type, &ary, &pyalloc))
		return NULL;

	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator"))
		return NULL;

	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	Cuda_GPUArray *pyary = Cuda_GPUArray_emptyLike((PyObject *)ary, allocator);
	if (pyary == NULL)
		return NULL;

	if (PyArray_IS_C_CONTIGUOUS(ary))
		CU_CHECK(cuMemcpyHtoD(
			(CUdeviceptr)pyary->gpudata->ptr, PyArray_DATA(ary), Cuda_GPUArray_nbytes(pyary)
		), goto error);

	else
	{
		if (!Cuda_GPUArray_memcpyDiscontig((PyObject *)ary, (PyObject *)pyary))
			goto error;
	}

	return (PyObject *)pyary;

error:
	Py_DECREF(pyary);
	return NULL;
}


static bool Cuda_GPUArray_unpackDimSlice(const Cuda_GPUArray *self, size_t idx, PyObject *slice,
										 Cuda_ArraySpec *spec, size_t *offset, size_t *slicesize)
{
	const size_t *shape = CUDA_GPUARRAY_SHAPE(self), *strides = CUDA_GPUARRAY_STRIDES(self);

	if (PySlice_Check(slice))
	{
		Py_ssize_t start, stop, step, slicelength;

		if (PySlice_GetIndicesEx(slice, shape[idx], &start, &stop, &step, &slicelength) < 0)
			return false;

		if (step != 1)
		{
			PyErr_Format(
				PyExc_ValueError, "slice #%d step %" PRIuMAX " is not contiguous", (int)(idx + 1), (size_t)step
			);
			return false;
		}

		spec->shape[spec->ndim] = slicelength;
		spec->strides[spec->ndim] = strides[idx];

		spec->ndim += 1;
		spec->size *= slicelength;

		*offset += start * strides[idx];
	}
	else if (PyLong_CheckExact(slice))
	{
		Py_ssize_t pydim = PyLong_AsSsize_t(slice);
		if (pydim == -1 && PyErr_Occurred())
			return false;

		pydim = (pydim < 0) ? (Py_ssize_t)shape[idx] + pydim : pydim;
		size_t dim = pydim;

		if (dim >= shape[idx])
		{
			PyErr_Format(
				PyExc_IndexError, "index is out of bounds for axis %d with size %" PRIuMAX, (int)idx, shape[idx]
			);
			return false;
		}

		*offset += dim * strides[idx];
	}
	else if (Py_TYPE(slice) == Py_TYPE(Py_None))
	{
		spec->shape[spec->ndim] = 1;
		spec->strides[spec->ndim] = strides[idx > 0 ? idx - 1 : 0];

		spec->ndim += 1;
		*slicesize -= 1;
	}
	else
	{
		PyErr_Format(
			PyExc_TypeError, "subscript item #%d must be %s or %s, not %s", (int)(idx + 1),
			(&PySlice_Type)->tp_name, (&PyLong_Type)->tp_name, Py_TYPE(slice)->tp_name
		);
		return false;
	}

	return true;
}


static void Cuda_updateSliceContigFlag(Cuda_ArraySpec *spec)
{
	spec->contiguous = true;

	for (size_t i = 0; i < spec->ndim - 1; i += 1)
	{
		if (spec->strides[i] != spec->strides[i + 1] * spec->shape[i + 1])
		{
			spec->contiguous = false;
			break;
		}
	}
}


static PyObject *Cuda_GPUArray_getSlice(PyObject *self, PyObject *slice)
{
	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;
	const size_t *shape = CUDA_GPUARRAY_SHAPE(pyary), *strides = CUDA_GPUARRAY_STRIDES(pyary);

	if (pyary->ndim == 0)
	{
		PyErr_SetString(PyExc_ValueError, "cannot slice 0d gpuarray");
		return NULL;
	}

	Cuda_ArraySpec spec;

	spec.ndim = 0, spec.size = 1;
	spec.dtype = pyary->dtype;

	size_t slicesize = 1, offset = 0;

	if (PyTuple_CheckExact(slice))
	{
		slicesize = PyTuple_GET_SIZE(slice);

		if (slicesize > pyary->ndim)
		{
			PyErr_SetString(PyExc_ValueError, "too many indices for gpuarray");
			return NULL;
		}

		for (size_t i = 0; i < slicesize; i += 1)
		{
			if (!Cuda_GPUArray_unpackDimSlice(pyary, i, PyTuple_GET_ITEM(slice, i), &spec, &offset, &slicesize))
				return NULL;
		}
	}
	else if (!Cuda_GPUArray_unpackDimSlice(pyary, 0, slice, &spec, &offset, &slicesize))
		return NULL;

	for (size_t i = slicesize; i < pyary->ndim; i += 1)
	{
		spec.shape[spec.ndim + i - slicesize] = shape[i];
		spec.strides[spec.ndim + i - slicesize] = strides[i];

		spec.size *= shape[i];
	}

	spec.ndim += pyary->ndim - slicesize;
	Cuda_updateSliceContigFlag(&spec);

	Cuda_Buffer *gpudata = pyary->gpudata;

	if (offset > 0)
	{
		gpudata = Cuda_Buffer_getSlice(gpudata, offset, gpudata->size - offset);
		if (gpudata == NULL)
			return NULL;
	}
	else
		Py_INCREF(gpudata);

	Cuda_GPUArray *sliced = Cuda_GPUArray_new(gpudata, &spec);
	if (sliced == NULL)
		goto error;

	return (PyObject *)sliced;

error:
	Py_DECREF(gpudata);
	return NULL;
}


inline static PyObject *Cuda_GPUArray_getSequence(size_t *sequence, size_t length)
{
	PyObject *pyseq = PyTuple_New(length);
	if (pyseq == NULL)
		goto error_1;

	for (size_t i = 0; i < length; i += 1)
	{
		PyObject *item = PyLong_FromSize_t(sequence[i]);
		if (item == NULL)
			goto error_2;

		PyTuple_SET_ITEM(pyseq, i, item);
	}

	return pyseq;

error_2:
	Py_DECREF(pyseq);

error_1:
	return NULL;
}


inline static PyObject *Cuda_GPUArray_getElement(PyObject *args, size_t *sequence, size_t ndim)
{
	Py_ssize_t pyidx;

	if (!PyArg_ParseTuple(args, "n", &pyidx))
		return NULL;

	pyidx = (pyidx < 0) ? (Py_ssize_t)ndim + pyidx : pyidx;
	size_t idx = pyidx;

	if (idx >= ndim)
	{
		PyErr_SetString(PyExc_IndexError, "index is out of bounds");
		return NULL;
	}

	return PyLong_FromSize_t(sequence[idx]);
}


static PyObject *Cuda_GPUArray_getShape(PyObject *self, void *closure)
{
	(void)closure;

	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;
	return Cuda_GPUArray_getSequence(CUDA_GPUARRAY_SHAPE(pyary), pyary->ndim);
}


PyDoc_STRVAR(Cuda_GPUArray_dimAt_doc, "dimAt(index) -> int");
static PyObject *Cuda_GPUArray_dimAt(PyObject *self, PyObject *args)
{
	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;
	return Cuda_GPUArray_getElement(args, CUDA_GPUARRAY_SHAPE(pyary), pyary->ndim);
}


static PyObject *Cuda_GPUArray_getStrides(PyObject *self, void *closure)
{
	(void)closure;

	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;
	return Cuda_GPUArray_getSequence(CUDA_GPUARRAY_STRIDES(pyary), pyary->ndim);
}


PyDoc_STRVAR(Cuda_GPUArray_strideAt_doc, "strideAt(index) -> int");
static PyObject *Cuda_GPUArray_strideAt(PyObject *self, PyObject *args)
{
	Cuda_GPUArray *pyary = (Cuda_GPUArray *)self;
	return Cuda_GPUArray_getElement(args, CUDA_GPUARRAY_STRIDES(pyary), pyary->ndim);
}


static PyObject *Cuda_GPUArray_getPtr(PyObject *self, void *closure)
{
	(void)closure;
	return PyLong_FromSize_t((size_t)((Cuda_GPUArray *)self)->gpudata->ptr);
}


static PyObject *Cuda_GPUArray_getNbytes(PyObject *self, void *closure)
{
	(void)closure;
	return PyLong_FromSize_t(Cuda_GPUArray_nbytes((Cuda_GPUArray *)self));
}


static PyObject *Cuda_GPUArray_getDtype(PyObject *self, void *closure)
{
	(void)closure;
	return (PyObject *)PyArray_DescrFromType(Cuda_fromDataType(((Cuda_GPUArray *)self)->dtype));
}


static PyObject *Cuda_GPUArray_getIsContiguous(PyObject *self, void *closure)
{
	(void)closure;

	if (((Cuda_GPUArray *)self)->contiguous)
		Py_RETURN_TRUE;
	else
		Py_RETURN_FALSE;
}


static PyObject *Cuda_GPUArray_getDevice(PyObject *self, void *closure)
{
	(void)closure;
	return PyLong_FromLong(((Cuda_GPUArray *)self)->gpudata->device);
}


static PyGetSetDef Cuda_GPUArray_getset[] = {
	{(char *)"shape", Cuda_GPUArray_getShape, NULL, NULL, NULL},
	{(char *)"strides", Cuda_GPUArray_getStrides, NULL, NULL, NULL},
	{(char *)"ptr", Cuda_GPUArray_getPtr, NULL, NULL, NULL},
	{(char *)"nbytes", Cuda_GPUArray_getNbytes, NULL, NULL, NULL},
	{(char *)"dtype", Cuda_GPUArray_getDtype, NULL, NULL, NULL},
	{(char *)"contiguous", Cuda_GPUArray_getIsContiguous, NULL, NULL, NULL},
	{(char *)"device", Cuda_GPUArray_getDevice, NULL, NULL, NULL},
	{NULL, NULL, NULL, NULL, NULL}
};

static PyMemberDef Cuda_GPUArray_members[] = {
	{(char *)"ndim", T_PYSSIZET, offsetof(Cuda_GPUArray, ndim), READONLY, NULL},
	{(char *)"size", T_PYSSIZET, offsetof(Cuda_GPUArray, size), READONLY, NULL},
	{(char *)"gpudata", T_OBJECT_EX, offsetof(Cuda_GPUArray, gpudata), READONLY, NULL},
	{NULL, 0, 0, 0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic push
	#if __GNUC__ >= 8
		#pragma GCC diagnostic ignored "-Wcast-function-type"
	#endif
#endif

static PyMethodDef Cuda_GPUArray_methods[] = {
	{"reshape", Cuda_GPUArray_reshape, METH_VARARGS, Cuda_GPUArray_reshape_doc},
	{"view", (PyCFunction)Cuda_GPUArray_view, METH_VARARGS | METH_KEYWORDS, Cuda_GPUArray_view_doc},
	{"ravel", Cuda_GPUArray_ravel, METH_NOARGS, Cuda_GPUArray_ravel_doc},

	{"get", Cuda_GPUArray_get, METH_NOARGS, Cuda_GPUArray_get_doc},
	{"set", (PyCFunction)Cuda_GPUArray_pySet, METH_VARARGS | METH_KEYWORDS, Cuda_GPUArray_pySet_doc},
	{"copy", (PyCFunction)Cuda_GPUArray_copy, METH_VARARGS | METH_KEYWORDS, Cuda_GPUArray_copy_doc},

	{"dimAt", Cuda_GPUArray_dimAt, METH_VARARGS, Cuda_GPUArray_dimAt_doc},
	{"strideAt", Cuda_GPUArray_strideAt, METH_VARARGS, Cuda_GPUArray_strideAt_doc},

	{"empty", (PyCFunction)Cuda_GPUArray_empty, METH_STATIC | METH_VARARGS | METH_KEYWORDS, Cuda_GPUArray_empty_doc},
	{"zeros", (PyCFunction)Cuda_GPUArray_zeros, METH_STATIC | METH_VARARGS | METH_KEYWORDS, Cuda_GPUArray_zeros_doc},
	{
		"emptyLike", (PyCFunction)Cuda_GPUArray_pyEmptyLike, METH_STATIC | METH_VARARGS | METH_KEYWORDS,
		Cuda_GPUArray_pyEmptyLike_doc
	},
	{
		"zerosLike", (PyCFunction)Cuda_GPUArray_zerosLike, METH_STATIC | METH_VARARGS | METH_KEYWORDS,
		Cuda_GPUArray_zerosLike_doc
	},

	{"toGpu", (PyCFunction)Cuda_GPUArray_toGpu, METH_STATIC | METH_VARARGS | METH_KEYWORDS, Cuda_GPUArray_toGpu_doc},
	{NULL, NULL, 0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif

static PyType_Slot Cuda_GPUArray_slots[] = {
	{Py_tp_str, (void *)Cuda_GPUArray_toString},
	{Py_tp_repr, (void *)Cuda_GPUArray_toString},
	{Py_tp_new, (void *)Cuda_GPUArray_pyNew},
	{Py_tp_dealloc, (void *)Cuda_GPUArray_dealloc},
	{Py_mp_subscript, (void *)Cuda_GPUArray_getSlice},
	{Py_tp_getset, Cuda_GPUArray_getset},
	{Py_tp_members, Cuda_GPUArray_members},
	{Py_tp_methods, Cuda_GPUArray_methods},
	{0, NULL}
};

static PyType_Spec Cuda_GPUArray_TypeSpec = {
	CUDA_GPUARRAY_FULLNAME,
	sizeof(Cuda_GPUArray),
	sizeof(size_t),
	Py_TPFLAGS_DEFAULT,
	Cuda_GPUArray_slots
};


PyTypeObject *Cuda_GPUArray_Type = NULL;


bool Cuda_GPUArray_moduleInit(PyObject *m)
{
	if (!createPyClass(m, CUDA_GPUARRAY_OBJNAME, &Cuda_GPUArray_TypeSpec, &Cuda_GPUArray_Type))
		return false;

	return true;
}


void Cuda_GPUArray_moduleDealloc(void)
{
	REMOVE_PY_OBJECT(&Cuda_GPUArray_Type);
}
