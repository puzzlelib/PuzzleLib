#include "Libs.h"


PyObject *CuBlas_Error = NULL;


inline static const char *cublasGetErrorString(cublasStatus_t code)
{
	switch (code)
	{
		case CUBLAS_STATUS_SUCCESS:          return TO_STRING(CUBLAS_STATUS_SUCCESS);
		case CUBLAS_STATUS_NOT_INITIALIZED:  return TO_STRING(CUBLAS_STATUS_NOT_INITIALIZED);
		case CUBLAS_STATUS_ALLOC_FAILED:     return TO_STRING(CUBLAS_STATUS_ALLOC_FAILED);
		case CUBLAS_STATUS_INVALID_VALUE:    return TO_STRING(CUBLAS_STATUS_INVALID_VALUE);
		case CUBLAS_STATUS_ARCH_MISMATCH:    return TO_STRING(CUBLAS_STATUS_ARCH_MISMATCH);
		case CUBLAS_STATUS_MAPPING_ERROR:    return TO_STRING(CUBLAS_STATUS_MAPPING_ERROR);
		case CUBLAS_STATUS_EXECUTION_FAILED: return TO_STRING(CUBLAS_STATUS_EXECUTION_FAILED);
		case CUBLAS_STATUS_INTERNAL_ERROR:   return TO_STRING(CUBLAS_STATUS_INTERNAL_ERROR);
		case CUBLAS_STATUS_NOT_SUPPORTED:    return TO_STRING(CUBLAS_STATUS_NOT_SUPPORTED);
		case CUBLAS_STATUS_LICENSE_ERROR:    return TO_STRING(CUBLAS_STATUS_LICENSE_ERROR);
		default:                             assert(false); return "UNKNOWN_ERROR";
	}
}


inline static bool cublasCheckStatus(cublasStatus_t code, const char *file, int line)
{
	if (code == CUBLAS_STATUS_SUCCESS)
		return true;

	const char *error = cublasGetErrorString(code);
	PyErr_Format(CuBlas_Error, "%s (%s:%d)\n", error, file, line);

	return false;
}


#define CUBLAS_CHECK(status, atexit) do { if (!cublasCheckStatus((status), __FILE__, __LINE__)) { atexit; } } while (0)
#define CUBLAS_ENFORCE(status) CUBLAS_CHECK(status, return NULL)
#define CUBLAS_ASSERT(status) \
do { cublasStatus_t code = (status); (void)code; assert(code == CUBLAS_STATUS_SUCCESS); } while (0)


typedef struct CuBlas_Context
{
	PyObject_HEAD

	cublasHandle_t handle;
	cublasGemmAlgo_t algo;
}
CuBlas_Context;


static PyObject *CuBlas_Context_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)args, (void)kwds;

	CuBlas_Context *self = (CuBlas_Context *)type->tp_alloc(type, 0);
	if (self == NULL)
		goto error_1;

	CUBLAS_CHECK(cublasCreate(&self->handle), goto error_2);
	self->algo = CUBLAS_GEMM_DEFAULT;

#if defined(TRACE_CUDA_CUBLAS)
	fprintf(stderr, "[" CUBLAS_CONTEXT_OBJNAME "] (0x%" PRIXMAX ") Allocated context\n", (size_t)self);
#endif

	return (PyObject *)self;

error_2:
	self->handle = NULL;
	Py_DECREF(self);

error_1:
	return NULL;
}


static void CuBlas_Context_dealloc(PyObject *self)
{
	CuBlas_Context *pyctx = (CuBlas_Context *)self;

	if (pyctx->handle != NULL)
	{
		CUBLAS_ASSERT(cublasDestroy(pyctx->handle));

#if defined(TRACE_CUDA_CUBLAS)
		fprintf(stderr, "[" CUBLAS_CONTEXT_OBJNAME "] (0x%" PRIXMAX ") Deallocated context\n", (size_t)self);
#endif
	}

	Py_TYPE(self)->tp_free(self);
}


PyDoc_STRVAR(CuBlas_Context_getVersion_doc, "getVersion(self) -> int");
static PyObject *CuBlas_Context_getVersion(PyObject *self, PyObject *args)
{
	(void)args;
	CuBlas_Context *pyctx = (CuBlas_Context *)self;

	int version;
	CUBLAS_ENFORCE(cublasGetVersion(pyctx->handle, &version));

	return Py_BuildValue("i", version);
}


PyDoc_STRVAR(CuBlas_Context_enableTensorOps_doc, "enableTensorOps(self, enable)");
static PyObject *CuBlas_Context_enableTensorOps(PyObject *self, PyObject *args)
{
	CuBlas_Context *pyctx = (CuBlas_Context *)self;
	int enable;

	if (!PyArg_ParseTuple(args, "p", &enable))
		return NULL;

	pyctx->algo = enable ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
	Py_RETURN_NONE;
}


typedef enum CuBlas_GroupFormat
{
	GROUPFORMAT_GBP = 0,
	GROUPFORMAT_BGP = 1
}
CuBlas_GroupFormat;


inline static bool CuBlas_unpackGroups(CuBlas_GroupFormat fmt, const size_t *shape, size_t *groups, size_t normShape[2])
{
	size_t groupAxis, batchAxis, paramAxis;

	switch (fmt)
	{
		case GROUPFORMAT_GBP: { groupAxis = 0, batchAxis = 1, paramAxis = 2; break; }
		case GROUPFORMAT_BGP: { groupAxis = 1, batchAxis = 0, paramAxis = 2; break; }
		default: { PyErr_SetString(PyExc_ValueError, "invalid group format"); return false; }
	}

	*groups = shape[groupAxis], normShape[0] = shape[batchAxis], normShape[1] = shape[paramAxis];
	return true;
}


inline static bool CuBlas_packGroups(CuBlas_GroupFormat fmt, size_t groups, const size_t normShape[2], size_t shape[3])
{
	size_t groupAxis, batchAxis, paramAxis;

	switch (fmt)
	{
		case GROUPFORMAT_GBP: { groupAxis = 0, batchAxis = 1, paramAxis = 2; break; }
		case GROUPFORMAT_BGP: { groupAxis = 1, batchAxis = 0, paramAxis = 2; break; }
		default: { PyErr_SetString(PyExc_ValueError, "invalid group format"); return false; }
	}

	shape[groupAxis] = groups, shape[batchAxis] = normShape[0], shape[paramAxis] = normShape[1];
	return true;
}


inline static bool CuBlas_unpackDtype(const Cuda_GPUArray *ary, cudaDataType *dtype)
{
	if (ary->dtype != DTYPE_FLOAT32 && ary->dtype != DTYPE_FLOAT16)
	{
		PyErr_SetString(PyExc_ValueError, "unsupported gpuarray dtype");
		return false;
	}

	*dtype = (ary->dtype == DTYPE_FLOAT32) ? CUDA_R_32F : CUDA_R_16F;
	return true;
}


typedef struct CuBlas_FortranFormat
{
	cublasOperation_t transa, transb;
	size_t m, n, k, lda, ldb, ldc;
}
CuBlas_FortranFormat;


inline static bool CuBlas_unpackFortranFormat(const size_t *Ashape, bool transpA, const size_t *Bshape, bool transpB,
											  CuBlas_FortranFormat *fmt)
{
	fmt->transa = CUBLAS_OP_N, fmt->transb = CUBLAS_OP_N;
	if (transpA && transpB) { PyErr_SetString(PyExc_ValueError, "invalid transpose mode"); return false; }

	if (transpA)
	{
		if (Ashape[0] != Bshape[0]) { PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims"); return false; }

		fmt->transb = CUBLAS_OP_T;
		fmt->m = Bshape[1], fmt->n = Ashape[1], fmt->k = Bshape[0];
		fmt->lda = Bshape[1], fmt->ldb = Ashape[1], fmt->ldc = Bshape[1];
	}
	else if (transpB)
	{
		if (Ashape[1] != Bshape[1]) { PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims"); return false; }

		fmt->transa = CUBLAS_OP_T;
		fmt->m = Bshape[0], fmt->n = Ashape[0], fmt->k = Bshape[1];
		fmt->lda = Bshape[1], fmt->ldb = Bshape[1], fmt->ldc = Bshape[0];
	}
	else
	{
		if (Ashape[1] != Bshape[0]) { PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims"); return false; }

		fmt->m = Bshape[1], fmt->n = Ashape[0], fmt->k = Bshape[0];
		fmt->lda = Bshape[1], fmt->ldb = Bshape[0], fmt->ldc = Bshape[1];
	}

	return true;
}


PyDoc_STRVAR(
	CuBlas_Context_gemmBatched_doc,
	"gemmBatched(self, A, B, formatA, formatB, formatOut, "
	"transpA=false, transpB=false, alpha=1.0, beta=0.0, out=None, allocator=None) -> " CUDA_GPUARRAY_FULLNAME
);
static PyObject *CuBlas_Context_gemmBatched(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {
		"A", "B", "formatA", "formatB", "formatOut", "transpA", "transpB", "alpha", "beta", "out", "allocator", NULL
	};
	CuBlas_Context *pyctx = (CuBlas_Context *)self;

	Cuda_GPUArray *A, *B;
	int formatA, formatB, formatOut, transpA = 0, transpB = 0;

	float alpha = 1.0f, beta = 0.0f;
	PyObject *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!iii|ppffOO", (char **)kwlist,
		Cuda_GPUArray_Type, &A, Cuda_GPUArray_Type, &B, &formatA, &formatB, &formatOut,
		&transpA, &transpB, &alpha, &beta, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (A->ndim != 3 || B->ndim != 3 || !A->contiguous || !B->contiguous || A->dtype != B->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	cudaDataType dtype;
	if (!CuBlas_unpackDtype(A, &dtype))
		return NULL;

	size_t groupsA, Ashape[2], groupsB, Bshape[2];

	if (!CuBlas_unpackGroups((CuBlas_GroupFormat)formatA, CUDA_GPUARRAY_SHAPE(A), &groupsA, Ashape))
		return NULL;

	if (!CuBlas_unpackGroups((CuBlas_GroupFormat)formatB, CUDA_GPUARRAY_SHAPE(B), &groupsB, Bshape))
		return NULL;

	size_t count = (groupsA > groupsB) ? groupsA : groupsB;

	if (groupsA != groupsB && groupsA != 1 && groupsB != 1)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray dims");
		return NULL;
	}

	CuBlas_FortranFormat fmt;
	if (!CuBlas_unpackFortranFormat(Ashape, transpA, Bshape, transpB, &fmt))
		return NULL;

	size_t normOutshape[2];
	normOutshape[0] = fmt.n, normOutshape[1] = fmt.m;

	size_t outshape[3];
	if (!CuBlas_packGroups((CuBlas_GroupFormat)formatOut, count, normOutshape, outshape))
		return NULL;

	if (out == NULL)
	{
		Cuda_ArraySpec spec;

		spec.shape[0] = outshape[0], spec.shape[1] = outshape[1], spec.shape[2] = outshape[2];

		spec.strides[2] = Cuda_dtypeSize(A->dtype);
		spec.strides[1] = outshape[2] * spec.strides[2];
		spec.strides[0] = outshape[1] * spec.strides[1];

		spec.ndim = 3, spec.size = outshape[0] * outshape[1] * outshape[2];
		spec.dtype = A->dtype, spec.contiguous = true;

		out = Cuda_GPUArray_newWithAllocator(allocator, NULL, &spec);
		if (out == NULL)
			return NULL;
	}
	else
	{
		const size_t *shapeOut = CUDA_GPUARRAY_SHAPE(out);
		if (out->ndim != 3 || !out->contiguous || out->dtype != A->dtype ||
			outshape[0] != shapeOut[0] || outshape[1] != shapeOut[1] || outshape[2] != shapeOut[2])
		{
			PyErr_SetString(PyExc_ValueError, "invalid output gpuarray data layout");
			return NULL;
		}

		Py_INCREF(out);
	}

	fmt.lda *= (formatB == GROUPFORMAT_GBP) ? 1 : groupsB;
	fmt.ldb *= (formatA == GROUPFORMAT_GBP) ? 1 : groupsA;
	fmt.ldc *= (formatOut == GROUPFORMAT_GBP) ? 1 : count;

	size_t strideA = (groupsB > 1) ? ((formatB == GROUPFORMAT_GBP) ? Bshape[0] * Bshape[1] : Bshape[1]) : 0;
	size_t strideB = (groupsA > 1) ? ((formatA == GROUPFORMAT_GBP) ? Ashape[0] * Ashape[1] : Ashape[1]) : 0;
	size_t strideOut = (formatOut == GROUPFORMAT_GBP) ? normOutshape[0] * normOutshape[1] : normOutshape[1];

	CUBLAS_CHECK(cublasGemmStridedBatchedEx(
		pyctx->handle, fmt.transa, fmt.transb, (int)fmt.m, (int)fmt.n, (int)fmt.k, &alpha,
		B->gpudata->ptr, dtype, (int)fmt.lda, strideA, A->gpudata->ptr, dtype, (int)fmt.ldb, strideB, &beta,
		out->gpudata->ptr, dtype, (int)fmt.ldc, strideOut, (int)count, CUDA_R_32F, pyctx->algo
	), goto error);

	return (PyObject *)out;

error:
	Py_DECREF(out);
	return NULL;
}


PyDoc_STRVAR(
	CuBlas_Context_gemm_doc,
	"gemm(self, A, B, out=None, transpA=false, transpB=false, alpha=1.0, beta=0.0, allocator=None) -> "
	CUDA_GPUARRAY_FULLNAME
);
static PyObject *CuBlas_Context_gemm(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"A", "B", "out", "transpA", "transpB", "alpha", "beta", "allocator", NULL};
	CuBlas_Context *pyctx = (CuBlas_Context *)self;

	Cuda_GPUArray *A, *B;

	int transpA = 0, transpB = 0;
	float alpha = 1.0f, beta = 0.0f;
	PyObject *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!|OppffO", (char **)kwlist,
		Cuda_GPUArray_Type, &A, Cuda_GPUArray_Type, &B, &pyout, &transpA, &transpB, &alpha, &beta, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	if (A->ndim != 2 || B->ndim != 2 || !A->contiguous || !B->contiguous || A->dtype != B->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	cudaDataType dtype;
	if (!CuBlas_unpackDtype(A, &dtype))
		return NULL;

	const size_t *Ashape = CUDA_GPUARRAY_SHAPE(A), *Bshape = CUDA_GPUARRAY_SHAPE(B);

	CuBlas_FortranFormat fmt;
	if (!CuBlas_unpackFortranFormat(Ashape, transpA, Bshape, transpB, &fmt))
		return NULL;

	if (out == NULL)
	{
		Cuda_ArraySpec spec;

		spec.shape[0] = fmt.n, spec.shape[1] = fmt.m;
		spec.strides[0] = fmt.m * Cuda_dtypeSize(A->dtype), spec.strides[1] = Cuda_dtypeSize(A->dtype);
		spec.ndim = 2, spec.size = fmt.n * fmt.m, spec.dtype = A->dtype, spec.contiguous = true;

		out = Cuda_GPUArray_newWithAllocator(allocator, NULL, &spec);
		if (out == NULL)
			return NULL;
	}
	else
	{
		const size_t *outshape = CUDA_GPUARRAY_SHAPE(out);
		if (out->ndim != 2 || !out->contiguous || out->dtype != A->dtype ||
			outshape[0] != fmt.n || outshape[1] != fmt.m)
		{
			PyErr_SetString(PyExc_ValueError, "invalid output gpuarray data layout");
			return NULL;
		}

		Py_INCREF(out);
	}

	CUBLAS_CHECK(cublasGemmEx(
		pyctx->handle, fmt.transa, fmt.transb, (int)fmt.m, (int)fmt.n, (int)fmt.k, &alpha,
		B->gpudata->ptr, dtype, (int)fmt.lda, A->gpudata->ptr, dtype, (int)fmt.ldb,
		&beta, out->gpudata->ptr, dtype, (int)fmt.ldc, CUDA_R_32F, pyctx->algo
	), goto error);

	return (PyObject *)out;

error:
	Py_DECREF(out);
	return NULL;
}


PyDoc_STRVAR(CuBlas_Context_l1norm_doc, "l1norm(self, x) -> float");
static PyObject *CuBlas_Context_l1norm(PyObject *self, PyObject *args)
{
	CuBlas_Context *pyctx = (CuBlas_Context *)self;
	Cuda_GPUArray *x;

	if (!PyArg_ParseTuple(args, "O!", Cuda_GPUArray_Type, &x))
		return NULL;

	if (x->ndim != 1 || !x->contiguous || x->dtype != DTYPE_FLOAT32)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	float result;
	CUBLAS_ENFORCE(cublasSasum(
		pyctx->handle, (int)CUDA_GPUARRAY_SHAPE(x)[0], (const float *)x->gpudata->ptr, 1, &result
	));

	return Py_BuildValue("f", result);
}


PyDoc_STRVAR(CuBlas_Context_l2norm_doc, "l2norm(self, x) -> float");
static PyObject *CuBlas_Context_l2norm(PyObject *self, PyObject *args)
{
	CuBlas_Context *pyctx = (CuBlas_Context *)self;
	Cuda_GPUArray *x;

	if (!PyArg_ParseTuple(args, "O!", Cuda_GPUArray_Type, &x))
		return NULL;

	if (x->ndim != 1 || !x->contiguous || x->dtype != DTYPE_FLOAT32)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	float result;
	CUBLAS_ENFORCE(cublasSnrm2(
		pyctx->handle, (int)CUDA_GPUARRAY_SHAPE(x)[0], (const float *)x->gpudata->ptr, 1, &result
	));

	return Py_BuildValue("f", result);
}


PyDoc_STRVAR(CuBlas_Context_dot_doc, "dot(self, x, y) -> float");
static PyObject *CuBlas_Context_dot(PyObject *self, PyObject *args)
{
	CuBlas_Context *pyctx = (CuBlas_Context *)self;
	Cuda_GPUArray *x, *y;

	if (!PyArg_ParseTuple(args, "O!O!", Cuda_GPUArray_Type, &x, Cuda_GPUArray_Type, &y))
		return NULL;

	if (x->ndim != 1 || !x->contiguous || x->dtype != DTYPE_FLOAT32)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	float result;

	CUBLAS_ENFORCE(cublasSdot(
		pyctx->handle, (int)CUDA_GPUARRAY_SHAPE(x)[0],
		(const float *)x->gpudata->ptr, 1, (const float *)y->gpudata->ptr, 1, &result
	));

	return Py_BuildValue("f", result);
}


#if defined(__GNUC__)
	#pragma GCC diagnostic push
	#if __GNUC__ >= 8
		#pragma GCC diagnostic ignored "-Wcast-function-type"
	#endif
#endif

static PyMethodDef CuBlas_Context_methods[] = {
	{"getVersion", CuBlas_Context_getVersion, METH_NOARGS, CuBlas_Context_getVersion_doc},
	{"enableTensorOps", CuBlas_Context_enableTensorOps, METH_VARARGS, CuBlas_Context_enableTensorOps_doc},

	{
		"gemmBatched", (PyCFunction)CuBlas_Context_gemmBatched, METH_VARARGS | METH_KEYWORDS,
		CuBlas_Context_gemmBatched_doc
	},
	{"gemm", (PyCFunction)CuBlas_Context_gemm, METH_VARARGS | METH_KEYWORDS, CuBlas_Context_gemm_doc},

	{"l1norm", CuBlas_Context_l1norm, METH_VARARGS, CuBlas_Context_l1norm_doc},
	{"l2norm", CuBlas_Context_l2norm, METH_VARARGS, CuBlas_Context_l2norm_doc},
	{"dot", CuBlas_Context_dot, METH_VARARGS, CuBlas_Context_dot_doc},

	{NULL, NULL, 0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif

static PyType_Slot CuBlas_Context_slots[] = {
	{Py_tp_new, (void *)CuBlas_Context_new},
	{Py_tp_dealloc, (void *)CuBlas_Context_dealloc},
	{Py_tp_methods, CuBlas_Context_methods},
	{0, NULL}
};

static PyType_Spec CuBlas_Context_TypeSpec = {
	CUBLAS_BACKEND_NAME "." CUBLAS_CONTEXT_OBJNAME,
	sizeof(CuBlas_Context),
	0,
	Py_TPFLAGS_DEFAULT,
	CuBlas_Context_slots
};


PyTypeObject *CuBlas_Context_Type = NULL;


static PyModuleDef CuBlas_module = {
	PyModuleDef_HEAD_INIT,
	CUBLAS_BACKEND_NAME,
	NULL, 0,
	NULL,
	NULL, NULL, NULL, NULL
};


bool CuBlas_moduleInit(PyObject *module)
{
	PyObject *m = PyModule_Create(&CuBlas_module);
	if (m == NULL)
		goto error_1;

	if (!createPyClass(m, CUBLAS_CONTEXT_OBJNAME, &CuBlas_Context_TypeSpec, &CuBlas_Context_Type))    goto error_2;
	if (!createPyExc(m, CUBLAS_ERROR_NAME, CUBLAS_BACKEND_NAME "." CUBLAS_ERROR_NAME, &CuBlas_Error)) goto error_3;
	if (PyModule_AddObject(module, CUBLAS_BACKEND_NAME, m) < 0)                                       goto error_4;

	PyModule_AddIntMacro(m, GROUPFORMAT_GBP);
	PyModule_AddIntMacro(m, GROUPFORMAT_BGP);

	return true;

error_4:
	REMOVE_PY_OBJECT(&CuBlas_Error);
error_3:
	REMOVE_PY_OBJECT(&CuBlas_Context_Type);
error_2:
	Py_DECREF(m);
error_1:
	return false;
}


void CuBlas_moduleDealloc(void)
{
	REMOVE_PY_OBJECT(&CuBlas_Error);
	REMOVE_PY_OBJECT(&CuBlas_Context_Type);
}
