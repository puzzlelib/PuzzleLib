#pragma once

#include "../Core/Driver.h"


// CuRand
#define CURAND_ERROR_NAME CURAND_BACKEND_NAME "Error"
#define CURAND_RNG_OBJNAME "RandomNumberGenerator"

bool CuRand_moduleInit(PyObject *m);
void CuRand_moduleDealloc(void);


// CuBlas
#define CUBLAS_ERROR_NAME CUBLAS_BACKEND_NAME "Error"
#define CUBLAS_CONTEXT_OBJNAME "BlasContext"

bool CuBlas_moduleInit(PyObject *m);
void CuBlas_moduleDealloc(void);


// CuDnn
#if defined(CUDA_BACKEND_IS_CUDA)

#define CUDNN_ERROR_NAME CUDNN_BACKEND_NAME "Error"
#define CUDNN_CONTEXT_OBJNAME "DnnContext"

extern PyObject *CuDnn_Error;

inline static bool cudnnCheckStatus(cudnnStatus_t code, const char *file, int line)
{
	if (code == CUDNN_STATUS_SUCCESS)
		return true;

	const char *error = cudnnGetErrorString(code);
	PyErr_Format(CuDnn_Error, "%s (%s:%d)\n", error, file, line);

	return false;
}

#define CUDNN_CHECK(status, atexit) do { if (!cudnnCheckStatus((status), __FILE__, __LINE__)) { atexit; } } while (0)
#define CUDNN_ENFORCE(status) CUDNN_CHECK(status, return NULL)
#define CUDNN_ASSERT(status) \
do { cudnnStatus_t code = (status); (void)code; assert(code == CUDNN_STATUS_SUCCESS); } while (0)

enum
{
	GPUTENSOR_DIM_MIN = 4,
	GPUTENSOR_DIM_MAX = 5
};

inline static bool CuDnn_isValidDim(size_t dim) { return dim <= GPUTENSOR_DIM_MAX && dim >= GPUTENSOR_DIM_MIN; }
inline static bool CuDnn_isValidExtDim(size_t dim) { return dim <= GPUTENSOR_DIM_MAX && dim >= 1; }
inline static bool CuDnn_isValidDescribedDim(size_t dim) { return dim <= GPUTENSOR_DIM_MAX && dim >= 3; }
inline static bool CuDnn_isValidDtype(Cuda_DataType dtype) { return dtype == DTYPE_FLOAT32 || dtype == DTYPE_FLOAT16; }

inline static bool CuDnn_isValid1DTensor(const Cuda_GPUArray *tensor, size_t size, Cuda_DataType dtype, const char *key)
{
	if (tensor->ndim != 1 || !tensor->contiguous || CUDA_GPUARRAY_SHAPE(tensor)[0] != size || tensor->dtype != dtype)
	{
		PyErr_Format(PyExc_ValueError, "invalid %s gpuarray data layout", key);
		return false;
	}

	return true;
}

inline static bool CuDnn_unpackIntTuple(PyObject *pytuple, size_t *seq, size_t length, const char *key)
{
	size_t pylength = (size_t)PyTuple_GET_SIZE(pytuple);

	if (length != pylength)
	{
		PyErr_Format(PyExc_ValueError, "%s must be %d-tuple, not %d", key, (int)length, (int)pylength);
		return false;
	}

	for (size_t i = 0; i < length; i += 1)
	{
		size_t item = PyLong_AsSize_t(PyTuple_GET_ITEM(pytuple, i));
		if (item == (size_t)-1 && PyErr_Occurred())
			return false;

		seq[i] = item;
	}

	return true;
}

inline static bool CuDnn_unpackIntSequence(PyObject *pyseq, size_t *seq, size_t length, size_t defval, const char *key)
{
	if (pyseq == NULL)
	{
		for (size_t i = 0; i < length; i += 1)
			seq[i] = defval;
	}
	else if (PyLong_CheckExact(pyseq))
	{
		size_t item = PyLong_AsSize_t(pyseq);
		if (item == (size_t)-1 && PyErr_Occurred())
			return false;

		for (size_t i = 0; i < length; i += 1)
			seq[i] = item;
	}
	else if (PyTuple_CheckExact(pyseq))
		return CuDnn_unpackIntTuple(pyseq, seq, length, key);
	else
	{
		PyErr_Format(
			PyExc_TypeError, "%s must be %s or %s, not %s",
			key, PyLong_Type.tp_name, PyTuple_Type.tp_name, Py_TYPE(pyseq)->tp_name
		);
		return false;
	}

	return true;
}

inline static cudnnDataType_t CuDnn_dtypeToDnn(Cuda_DataType dtype)
{
	switch (dtype)
	{
		case DTYPE_FLOAT32: return CUDNN_DATA_FLOAT;
		case DTYPE_FLOAT16: return CUDNN_DATA_HALF;
		default:            assert(false); return (cudnnDataType_t)-1;
	}
}

inline static Cuda_Buffer *Cuda_Buffer_newWithAllocator(size_t size, int device, Cuda_MemoryPool *allocator)
{
	if (allocator != NULL)
	{
		assert(device == allocator->device);
		return Cuda_MemoryPool_allocate(allocator, size);
	}
	else
		return Cuda_Driver_allocateWithKnownDevice(size, device);
}

Cuda_GPUArray *CuDnn_enforceAllocated(Cuda_GPUArray *out, Cuda_MemoryPool *allocator, const size_t *shape, size_t ndim,
									  Cuda_DataType dtype, bool zeroOut);

bool CuDnn_describeTensorFromShape(cudnnTensorDescriptor_t *desc, const size_t *shape, const size_t *strides,
								   size_t ndim, Cuda_DataType dtype);

bool CuDnn_describeTensor(cudnnTensorDescriptor_t *desc, const Cuda_GPUArray *tensor);
bool CuDnn_describe1DTensor(cudnnTensorDescriptor_t *desc, const Cuda_GPUArray *tn, size_t ndim);

bool CuDnn_describeFilterFromShape(cudnnFilterDescriptor_t *desc, const size_t *sh, size_t ndim, Cuda_DataType dtype);

typedef struct CuDnn_Context
{
	PyObject_HEAD

	cudnnHandle_t handle;
	cudnnMathType_t mathType;
}
CuDnn_Context;

extern PyTypeObject *CuDnn_Context_Type;
bool CuDnn_moduleInit(PyObject *m);
void CuDnn_moduleDealloc(void);


// CuDnn Pooling
extern const char CuDnn_Context_pyPoolNd_doc[], CuDnn_Context_pyPoolNdBackward_doc[];

PyObject *CuDnn_Context_pyPoolNd(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_pyPoolNdBackward(PyObject *self, PyObject *args, PyObject *kwds);

void CuDnnPool_moduleInit(PyObject *m);


// CuDnn Normalization
extern const char CuDnn_Context_pyBatchNormNd_doc[], CuDnn_Context_pyBatchNormNdBackward_doc[];
extern const char CuDnn_Context_pyMapLRN_doc[], CuDnn_Context_pyMapLRNBackward_doc[];
extern const char CuDnn_Context_pyCrossMapLRN_doc[], CuDnn_Context_pyCrossMapLRNBackward_doc[];

PyObject *CuDnn_Context_pyBatchNormNd(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_pyBatchNormNdBackward(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_pyMapLRN(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_pyMapLRNBackward(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_pyCrossMapLRN(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_pyCrossMapLRNBackward(PyObject *self, PyObject *args, PyObject *kwds);

void CuDnnNorm_moduleInit(PyObject *m);


// CuDnn Memory Ops
extern const char CuDnn_Context_pyTranspose_doc[], CuDnn_Context_moveaxis_doc[], CuDnn_Context_swapaxes_doc[];
extern const char CuDnn_Context_depthConcat_doc[], CuDnn_Context_depthSplit_doc[];

PyObject *CuDnn_Context_pyTranspose(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_moveaxis(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_swapaxes(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_depthConcat(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_depthSplit(PyObject *self, PyObject *args, PyObject *kwds);


// CuDnn SpatialTf
#if defined(CUDA_BACKEND_IS_CUDA)

extern const char CuDnn_Context_pySpatialTf_doc[];
extern const char CuDnn_Context_pySpatialTfBackward_doc[];

PyObject *CuDnn_Context_pySpatialTf(PyObject *self, PyObject *args, PyObject *kwds);
PyObject *CuDnn_Context_pySpatialTfBackward(PyObject *self, PyObject *args, PyObject *kwds);

#endif


// CuDnn Rnn
bool CuDnnRnn_moduleInit(PyObject *m);
void CuDnnRnn_moduleDealloc(void);

#endif
