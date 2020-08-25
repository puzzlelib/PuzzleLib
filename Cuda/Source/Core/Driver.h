#pragma once

#include "Common.h"
#include "AllocVector.gen.h"


// Device
#define CUDA_DEVICE_OBJNAME "Device"

typedef struct Cuda_Device
{
	PyObject_HEAD
	int index;
}
Cuda_Device;

extern PyTypeObject *Cuda_Device_Type;
bool Cuda_Device_moduleInit(PyObject *m);
void Cuda_Device_moduleDealloc(void);


// Stream
#define CUDA_STREAM_OBJNAME "Stream"
#define CUDA_EVENT_OBJNAME "Event"

typedef struct Cuda_Stream
{
	PyObject_HEAD
	cudaStream_t stream;
}
Cuda_Stream;

typedef struct Cuda_Event
{
	PyObject_HEAD
	cudaEvent_t event;
}
Cuda_Event;

extern PyTypeObject *Cuda_Stream_Type, *Cuda_Event_Type;
bool Cuda_Stream_moduleInit(PyObject *m);
void Cuda_Stream_moduleDealloc(void);


// Buffer
#define CUDA_BUFFER_OBJNAME "Buffer"

typedef struct Cuda_Buffer
{
	PyObject_HEAD

	Cuda_Ptr ptr;
	size_t size;

	PyObject *parent;

	int device;
	bool ipc;
}
Cuda_Buffer;

Cuda_Buffer *Cuda_Buffer_new(Cuda_Ptr ptr, size_t size, int device, PyObject *parent);
Cuda_Buffer *Cuda_Buffer_newFromIPCHandle(cudaIpcMemHandle_t handle, size_t size);
Cuda_Buffer *Cuda_Buffer_getSlice(Cuda_Buffer *self, size_t start, size_t size);

inline static bool Cuda_Buffer_fillD8(Cuda_Buffer *self, unsigned char uc, size_t size, Cuda_Stream *stream)
{
	size = (size == (size_t)-1) ? self->size : size;
	CUresult status;

	if (stream == NULL)
		status = cuMemsetD8((CUdeviceptr)self->ptr, uc, size);
	else
		status = cuMemsetD8Async((CUdeviceptr)self->ptr, uc, size, stream->stream);

	CU_CHECK(status, return false);
	return true;
}

inline static bool Cuda_Buffer_get(Cuda_Buffer *self, void *ptr, size_t size, Cuda_Stream *stream)
{
	size = (size == (size_t)-1) ? self->size : size;
	CUresult status;

	if (stream == NULL)
		status = cuMemcpyDtoH(ptr, (CUdeviceptr)self->ptr, size);
	else
		status = cuMemcpyDtoHAsync(ptr, (CUdeviceptr)self->ptr, size, stream->stream);

	CU_CHECK(status, return false);
	return true;
}

inline static bool Cuda_Buffer_set(Cuda_Buffer *self, void *ptr, size_t size, Cuda_Stream *stream)
{
	size = (size == (size_t)-1) ? self->size : size;
	CUresult status;

	if (stream == NULL)
		status = cuMemcpyHtoD((CUdeviceptr)self->ptr, ptr, size);
	else
		status = cuMemcpyHtoDAsync((CUdeviceptr)self->ptr, ptr, size, stream->stream);

	CU_CHECK(status, return false);
	return true;
}

inline static bool Cuda_Buffer_copy(Cuda_Buffer *self, Cuda_Buffer *dst, size_t size, Cuda_Stream *stream)
{
	size = (size == (size_t)-1) ? self->size : size;
	CUresult status;

	if (stream == NULL)
		status = cuMemcpyDtoD((CUdeviceptr)dst->ptr, (CUdeviceptr)self->ptr, size);
	else
		status = cuMemcpyDtoDAsync((CUdeviceptr)dst->ptr, (CUdeviceptr)self->ptr, size, stream->stream);

	CU_CHECK(status, return false);
	return true;
}

extern PyTypeObject *Cuda_Buffer_Type;
bool Cuda_Buffer_moduleInit(PyObject *m);
void Cuda_Buffer_moduleDealloc(void);


// Allocator
#define CUDA_MEMORY_POOL_OBJNAME "MemoryPool"

enum
{
	MEMORY_POOL_MANTISSA_BITS = 2,

	MEMORY_POOL_MIN_BIN_BITS = 4,
	MEMORY_POOL_OFFSET = MEMORY_POOL_MIN_BIN_BITS * (1 << MEMORY_POOL_MANTISSA_BITS),

	MEMORY_POOL_NUM_OF_BINS = sizeof(uint64_t) * 8 * (1 << MEMORY_POOL_MANTISSA_BITS) - MEMORY_POOL_OFFSET
};

typedef struct Cuda_MemoryPool
{
	PyObject_HEAD

	Cuda_AllocVector bins[MEMORY_POOL_NUM_OF_BINS];
	size_t activeBlocks, heldBlocks;

	int device;
	bool holding;
}
Cuda_MemoryPool;

Cuda_Buffer *Cuda_MemoryPool_allocate(Cuda_MemoryPool *self, size_t nbytes);
void Cuda_MemoryPool_hold(Cuda_MemoryPool *self, Cuda_Buffer *pybuf);

extern PyTypeObject *Cuda_MemoryPool_Type;
bool Cuda_Allocator_moduleInit(PyObject *m);
void Cuda_Allocator_moduleDealloc(void);


// Array
#define CUDA_GPUARRAY_OBJNAME "GPUArray"
#define CUDA_GPUARRAY_FULLNAME CUDA_DRIVER_NAME "." CUDA_GPUARRAY_OBJNAME

typedef enum Cuda_DataType
{
	DTYPE_INVALID,

	DTYPE_INT8,
	DTYPE_UINT8,

	DTYPE_INT16,
	DTYPE_UINT16,

	DTYPE_INT32,
	DTYPE_UINT32,

	DTYPE_INT64,
	DTYPE_UINT64,

	DTYPE_FLOAT16,
	DTYPE_FLOAT32,
	DTYPE_FLOAT64,

#if UINTPTR_MAX > UINT_MAX
	DTYPE_INTP = DTYPE_INT64,
	DTYPE_UINTP = DTYPE_UINT64

#else
	DTYPE_INTP = DTYPE_INT32,
	DTYPE_UINTP = DTYPE_UINT32

#endif
}
Cuda_DataType;

inline static size_t Cuda_dtypeSize(Cuda_DataType dtype)
{
	switch (dtype)
	{
		case DTYPE_FLOAT32: return sizeof(float);
		case DTYPE_INT32:   return sizeof(int32_t);
		case DTYPE_INT8:    return sizeof(int8_t);
		case DTYPE_FLOAT16: return sizeof(int16_t);
		case DTYPE_INT64:   return sizeof(int64_t);
		case DTYPE_FLOAT64: return sizeof(double);
		case DTYPE_UINT32:  return sizeof(uint32_t);
		case DTYPE_UINT8:   return sizeof(uint8_t);
		case DTYPE_UINT64:  return sizeof(uint64_t);
		case DTYPE_INT16:   return sizeof(int16_t);
		case DTYPE_UINT16:  return sizeof(uint16_t);
		default:            assert(false); return (size_t)-1;
	}
}

inline static Cuda_DataType Cuda_numpyToDataType(int typenum)
{
	switch (typenum)
	{
		case NPY_FLOAT32: return DTYPE_FLOAT32;
		case NPY_INT32:   return DTYPE_INT32;
		case NPY_INT8:    return DTYPE_INT8;
		case NPY_FLOAT16: return DTYPE_FLOAT16;
		case NPY_INT64:   return DTYPE_INT64;
		case NPY_FLOAT64: return DTYPE_FLOAT64;
		case NPY_UINT32:  return DTYPE_UINT32;
		case NPY_UINT8:   return DTYPE_UINT8;
		case NPY_UINT64:  return DTYPE_UINT64;
		case NPY_INT16:   return DTYPE_INT16;
		case NPY_UINT16:  return DTYPE_UINT16;
		default:          PyErr_SetString(PyExc_ValueError, "unsupported datatype"); return DTYPE_INVALID;
	}
}

enum
{
	GPUARRAY_NDIM_LIMIT = 32
};

typedef struct Cuda_ArraySpec
{
	size_t shape[GPUARRAY_NDIM_LIMIT], strides[GPUARRAY_NDIM_LIMIT];
	size_t ndim, size;

	Cuda_DataType dtype;
	bool contiguous;
}
Cuda_ArraySpec;

inline static void Cuda_copyShapeAsContiguous(size_t *outshape, size_t *outstrides, const size_t *inshape, size_t ndim,
											  size_t laststride)
{
	size_t lastdim = 1;

	for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; i -= 1)
	{
		outshape[i] = inshape[i];
		outstrides[i] = lastdim * laststride;

		lastdim = outshape[i], laststride = outstrides[i];
	}
}

inline static void Cuda_ArraySpec_initAsContiguous(Cuda_ArraySpec *self, const size_t *shape, size_t ndim,
												   Cuda_DataType dtype)
{
	Cuda_copyShapeAsContiguous(self->shape, self->strides, shape, ndim, Cuda_dtypeSize(dtype));

	self->ndim = ndim;
	self->size = self->strides[0] * self->shape[0] / Cuda_dtypeSize(dtype);

	self->dtype = dtype;
	self->contiguous = true;
}

#if defined(_MSC_VER)
	#pragma warning(push)
	#pragma warning(disable: 4200)
#endif

typedef struct Cuda_GPUArray
{
	PyObject_HEAD

	size_t ndim, size;
	Cuda_Buffer *gpudata;

	Cuda_DataType dtype;
	bool contiguous;

	size_t shapeAndStrides[];
}
Cuda_GPUArray;

#if defined(_MSC_VER)
	#pragma warning(pop)
#endif

#define CUDA_GPUARRAY_SHAPE(self) ((self)->shapeAndStrides)
#define CUDA_GPUARRAY_STRIDES(self) ((self)->shapeAndStrides + (self)->ndim)
#define CUDA_GPUARRAY_NBYTES(self) ((self)->size * Cuda_dtypeSize((self)->dtype))

Cuda_GPUArray *Cuda_GPUArray_newWithAllocator(Cuda_MemoryPool *allocator, Cuda_Buffer *gpudata,
											  const Cuda_ArraySpec *spec);

extern PyTypeObject *Cuda_GPUArray_Type;
bool Cuda_GPUArray_moduleInit(PyObject *m);
void Cuda_GPUArray_moduleDealloc(void);


// Module
#define CUDA_MODULE_OBJNAME "Module"
#define CUDA_FUNCTION_OBJNAME "Function"

typedef struct Cuda_Module
{
	PyObject_HEAD
	CUmodule module;
}
Cuda_Module;

typedef struct Cuda_Function
{
	PyObject_HEAD

	Cuda_Module *module;
	CUfunction function;
}
Cuda_Function;

extern PyTypeObject *Cuda_Module_Type, *Cuda_Function_Type;
bool Cuda_Module_moduleInit(PyObject *m);
void Cuda_Module_moduleDealloc(void);


// Driver
#define CUDA_DRIVER_NAME CUDA_BACKEND_NAME "Driver"
#define CUDA_ERROR_NAME CUDA_BACKEND_NAME "Error"
#define CUDA_NVRTC_ERROR_NAME "RtcError"

Cuda_Buffer *Cuda_Driver_allocateWithKnownDevice(size_t nbytes, int device);
Cuda_Buffer *Cuda_Driver_allocate(size_t nbytes);
