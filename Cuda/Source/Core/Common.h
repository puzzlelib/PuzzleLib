#pragma once

#include <assert.h>
#include <stdbool.h>

#define TO_STRING(s) #s


#if defined(__HIP_PLATFORM_HCC__)
	#include "HipDefines.h"

	#define CUDA_BACKEND_IS_HIP
	#define CUDA_BACKEND_NAME "Hip"
	#define CURAND_BACKEND_NAME "HipRand"
	#define CUBLAS_BACKEND_NAME "RocBlas"

#else
	#if defined(__clang__)
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wunused-function"

	#elif defined(_MSC_VER)
		#pragma warning(push)
		#pragma warning(disable: 4201 4505)
	#endif

	#include <cuda.h>

	#undef __cdecl
	#include <cuda_runtime.h>

	#include <nvrtc.h>
	#include <cuda_profiler_api.h>

	#include <curand.h>
	#include <cublas_v2.h>
	#include <cudnn.h>

	#if defined(__clang__)
		#pragma GCC diagnostic pop

	#elif defined(_MSC_VER)
		#pragma warning(pop)

	#endif

	#define CUDA_BACKEND_IS_CUDA
	#define CUDA_BACKEND_NAME "Cuda"
	#define CURAND_BACKEND_NAME "CuRand"
	#define CUBLAS_BACKEND_NAME "CuBlas"
	#define CUDNN_BACKEND_NAME "CuDnn"

inline static const char *curandGetErrorString(curandStatus_t code)
{
	switch (code)
	{
		case CURAND_STATUS_SUCCESS:                   return TO_STRING(CURAND_STATUS_SUCCESS);
		case CURAND_STATUS_VERSION_MISMATCH:          return TO_STRING(CURAND_STATUS_VERSION_MISMATCH);
		case CURAND_STATUS_NOT_INITIALIZED:           return TO_STRING(CURAND_STATUS_NOT_INITIALIZED);
		case CURAND_STATUS_ALLOCATION_FAILED:         return TO_STRING(CURAND_STATUS_ALLOCATION_FAILED);
		case CURAND_STATUS_TYPE_ERROR:                return TO_STRING(CURAND_STATUS_TYPE_ERROR);
		case CURAND_STATUS_OUT_OF_RANGE:              return TO_STRING(CURAND_STATUS_OUT_OF_RANGE);
		case CURAND_STATUS_LENGTH_NOT_MULTIPLE:       return TO_STRING(CURAND_STATUS_LENGTH_NOT_MULTIPLE);
		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return TO_STRING(CURAND_STATUS_DOUBLE_PRECISION_REQUIRED);
		case CURAND_STATUS_LAUNCH_FAILURE:            return TO_STRING(CURAND_STATUS_LAUNCH_FAILURE);
		case CURAND_STATUS_PREEXISTING_FAILURE:       return TO_STRING(CURAND_STATUS_PREEXISTING_FAILURE);
		case CURAND_STATUS_INITIALIZATION_FAILED:     return TO_STRING(CURAND_STATUS_INITIALIZATION_FAILED);
		case CURAND_STATUS_ARCH_MISMATCH:             return TO_STRING(CURAND_STATUS_ARCH_MISMATCH);
		case CURAND_STATUS_INTERNAL_ERROR:            return TO_STRING(CURAND_STATUS_INTERNAL_ERROR);
		default:                                      assert(false); return "UNKNOWN_ERROR";
	}
}

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

#endif


#include "PyDefines.gen.h"


#if !defined(CUDA_DRIVER_IMPORT_ARRAY)
	#define NO_IMPORT_ARRAY
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL CUDA_DRIVER_ARRAY_API

#if defined(__clang__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#if defined(__clang__)
	#pragma GCC diagnostic pop
#endif


extern PyObject *Cuda_Error;


inline static bool cudaCheckStatus(cudaError_t code, const char *file, int line)
{
	if (code == cudaSuccess)
		return true;

	const char *error = cudaGetErrorString(code), *name = cudaGetErrorName(code);
	PyErr_Format(Cuda_Error, "%s (%s) (%s:%d)", error, name, file, line);

	return false;
}


#define CUDA_CHECK(status, atexit) do { if (!cudaCheckStatus((status), __FILE__, __LINE__)) { atexit; } } while (0)
#define CUDA_ENFORCE(status) CUDA_CHECK(status, return NULL)
#define CUDA_ASSERT(status) do { cudaError_t code = (status); (void)code; assert(code == cudaSuccess); } while (0)

#define CU_CHECK(status, atexit) CUDA_CHECK((cudaError_t)status, atexit)
#define CU_ENFORCE(status) CU_CHECK(status, return NULL)
#define CU_ASSERT(status) CUDA_ASSERT((cudaError_t)status)
