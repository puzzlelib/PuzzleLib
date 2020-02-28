#pragma once

#include <stdbool.h>
#include <assert.h>


#if defined(__clang__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wvisibility"
	#pragma GCC diagnostic ignored "-Wunused-function"

#elif defined(_MSC_VER)
	#pragma warning(push)
	#pragma warning(disable: 4115)
	#pragma warning(disable: 4201)
	#pragma warning(disable: 4505)

#endif

#include <Python.h>
#include <structmember.h>

#undef __cdecl

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <curand.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <cuda_profiler_api.h>

#define CUDA_BACKEND_IS_CUDA
#define CUDA_BACKEND_NAME "Cuda"
#define CURAND_BACKEND_NAME "CuRand"
#define CUBLAS_BACKEND_NAME "CuBlas"
#define CUDNN_BACKEND_NAME "CuDnn"

#if defined(__clang__)
	#pragma GCC diagnostic pop

#elif defined(_MSC_VER)
	#pragma warning(pop)

#endif


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


inline static bool createPyClass(PyObject *module, const char *name, PyType_Spec *spec, PyTypeObject **pType)
{
	PyTypeObject *type = (PyTypeObject *)PyType_FromSpec(spec);
	if (type == NULL)
		return false;

	if (PyModule_AddObject(module, name, (PyObject *)type) < 0)
	{
		Py_DECREF(type);
		return false;
	}

	Py_INCREF(type);
	*pType = type;

	return true;
}

inline static bool createPyExc(PyObject *module, const char *name, const char *fullname, PyObject **pExc)
{
	PyObject *exc = PyErr_NewException(fullname, NULL, NULL);
	if (exc == NULL)
		return false;

	if (PyModule_AddObject(module, name, exc) < 0)
	{
		Py_DECREF(exc);
		return false;
	}

	Py_INCREF(exc);
	*pExc = exc;

	return true;
}

inline static bool unpackPyOptional(PyObject **pObj, PyTypeObject *type, const char *key)
{
	PyObject *obj = *pObj;

	if (obj != NULL && Py_TYPE(obj) != type && obj != Py_None)
	{
		PyErr_Format(
			PyExc_TypeError, "%s must be %s or %s, not %s",
			key, type->tp_name, Py_TYPE(Py_None)->tp_name, Py_TYPE(obj)->tp_name
		);
		return false;
	}

	*pObj = (obj == Py_None) ? NULL : obj;
	return true;
}

#define REMOVE_PY_OBJECT(pObj) do { PyObject *obj = (PyObject *)*(pObj); Py_DECREF(obj); *(pObj) = NULL; } while (0)


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
