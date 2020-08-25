#define CUDA_DRIVER_IMPORT_ARRAY
#include "Driver.h"

#include "../Libs/Libs.h"
#include "../TraceMalloc/TraceMalloc.gen.h"


PyObject *Cuda_Error = NULL, *Nvrtc_Error = NULL;


inline static bool nvrtcCheckStatus(nvrtcResult code, const char *file, int line)
{
	if (code == NVRTC_SUCCESS)
		return true;

	const char *error = nvrtcGetErrorString(code);
	PyErr_Format(Nvrtc_Error, "%s (%s:%d)", error, file, line);

	return false;
}


#define NVRTC_CHECK(status, atexit) do { if (!nvrtcCheckStatus(status, __FILE__, __LINE__)) { atexit; } } while (0)
#define NVRTC_ENFORCE(status) NVRTC_CHECK(status, return NULL)
#define NVRTC_ASSERT(status) do { nvrtcResult code = (status); (void)code; assert(code == NVRTC_SUCCESS); } while (0)


PyDoc_STRVAR(Cuda_Driver_getDriverVersion_doc, "getDriverVersion() -> int");
static PyObject *Cuda_Driver_getDriverVersion(PyObject *self, PyObject *args)
{
	(void)self, (void)args;

	int version;
	CUDA_ENFORCE(cudaDriverGetVersion(&version));

	return Py_BuildValue("i", version);
}


PyDoc_STRVAR(Cuda_Driver_getMemoryInfo_doc, "getMemoryInfo() -> Tuple[int, int]");
static PyObject *Cuda_Driver_getMemoryInfo(PyObject *self, PyObject *args)
{
	(void)self, (void)args;

	size_t free, total;
	CUDA_ENFORCE(cudaMemGetInfo(&free, &total));

	Py_ssize_t pyfree = free, pytotal = total;
	return Py_BuildValue("(nn)", pyfree, pytotal);
}


Cuda_Buffer *Cuda_Driver_allocateWithKnownDevice(size_t nbytes, int device)
{
	return Cuda_Buffer_new(0, nbytes, device, NULL);
}


Cuda_Buffer *Cuda_Driver_allocate(size_t nbytes)
{
	int device;
	CUDA_ENFORCE(cudaGetDevice(&device));

	return Cuda_Driver_allocateWithKnownDevice(nbytes, device);
}


PyDoc_STRVAR(Cuda_Driver_pyAllocate_doc, "allocate(nbytes) -> " CUDA_DRIVER_NAME "." CUDA_BUFFER_OBJNAME);
static PyObject *Cuda_Driver_pyAllocate(PyObject *self, PyObject *args)
{
	(void)self;
	Py_ssize_t pysize;

	if (!PyArg_ParseTuple(args, "n", &pysize))
		return NULL;

	return (PyObject *)Cuda_Driver_allocate(pysize);
}


PyDoc_STRVAR(
	Cuda_Driver_allocateFromIPCHandle_doc,
	"allocateFromIPCHandle(handle, nbytes) -> " CUDA_DRIVER_NAME "." CUDA_BUFFER_OBJNAME
);
static PyObject *Cuda_Driver_allocateFromIPCHandle(PyObject *type, PyObject *args)
{
	(void)type;

	const char *buffer;
	Py_ssize_clean_t length;

	Py_ssize_t pysize;

	if (!PyArg_ParseTuple(args, "y#n", &buffer, &length, &pysize))
		return NULL;

	cudaIpcMemHandle_t handle;

	if (length != sizeof(handle))
	{
		PyErr_SetString(PyExc_ValueError, "invalid ipc handle length");
		return NULL;
	}
	memcpy(&handle, buffer, sizeof(handle));

	return (PyObject *)Cuda_Buffer_newFromIPCHandle(handle, pysize);
}


typedef struct Cuda_Driver_Memcpy
{
	void *dst, *src;
	enum cudaMemcpyKind kind;
}
Cuda_Driver_Memcpy;


static bool Cuda_Driver_memcpyCheckBuffers(PyObject *dst, PyObject *src, Cuda_Driver_Memcpy *memcpy)
{
	bool dstIsDevice = true, srcIsDevice = true;

	if (Py_TYPE(dst) == Cuda_Buffer_Type)
	{
		Cuda_Buffer *dstbuf = (Cuda_Buffer *)dst;
		memcpy->dst = dstbuf->ptr;
	}
	else if (PyArray_CheckExact(dst))
	{
		PyArrayObject *dstary = (PyArrayObject *)dst;

		memcpy->dst = PyArray_DATA(dstary);
		dstIsDevice = false;
	}
	else
	{
		PyErr_Format(
			PyExc_TypeError, "destination must be %s or %s, not %s",
			Cuda_Buffer_Type->tp_name, (&PyArray_Type)->tp_name, Py_TYPE(src)->tp_name
		);
		return false;
	}

	if (Py_TYPE(src) == Cuda_Buffer_Type)
	{
		Cuda_Buffer *srcbuf = (Cuda_Buffer *)src;
		memcpy->src = srcbuf->ptr;
	}
	else if (PyArray_CheckExact(src))
	{
		PyArrayObject *srcary = (PyArrayObject *)src;

		memcpy->src = PyArray_DATA(srcary);
		srcIsDevice = false;
	}
	else
	{
		PyErr_Format(
			PyExc_TypeError, "source must be %s or %s, not %s",
			Cuda_Buffer_Type->tp_name, (&PyArray_Type)->tp_name, Py_TYPE(src)->tp_name
		);
		return false;
	}

	if (dstIsDevice)
		memcpy->kind = srcIsDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
	else
		memcpy->kind = srcIsDevice ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;

	return true;
}


PyDoc_STRVAR(
	Cuda_Driver_memcpy2D_doc, "memcpy2D(width, height, src, srcPitch, dst, dstPitch, srcX=0, srcY=0, dstX=0, dstY=0)"
);
static PyObject *Cuda_Driver_memcpy2D(PyObject *self, PyObject *args, PyObject *kwds)
{
	(void)self;
	const char *kwlist[] = {
		"width", "height", "src", "srcPitch", "dst", "dstPitch", "srcX", "srcY", "dstX", "dstY", NULL
	};

	Py_ssize_t width, height, srcPitch, dstPitch;
	PyObject *src, *dst;

	Py_ssize_t srcX = 0, srcY = 0, dstX = 0, dstY = 0;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "nnOnOn|nnnn", (char **)kwlist,
		&width, &height, &src, &srcPitch, &dst, &dstPitch, &srcX, &srcY, &dstX, &dstY
	))
		return NULL;

	Cuda_Driver_Memcpy memcpy;

	if (!Cuda_Driver_memcpyCheckBuffers(dst, src, &memcpy))
		return NULL;

	CUDA_ENFORCE(cudaMemcpy2D(
		(char *)memcpy.dst + dstY * dstPitch + dstX, dstPitch,
		(char *)memcpy.src + srcY * srcPitch + srcX, srcPitch,
		width, height, memcpy.kind
	));
	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Compiler_getCompilerVersion_doc, "getCompilerVersion() -> Tuple[int, int]");
static PyObject *Cuda_Compiler_getCompilerVersion(PyObject *self, PyObject *args)
{
	(void)self, (void)args;

	int major, minor;
	NVRTC_ENFORCE(nvrtcVersion(&major, &minor));

	return Py_BuildValue("(ii)", major, minor);
}


typedef struct Cuda_Compiler_Options
{
	int numOptions;
	const char **options;
}
Cuda_Compiler_Options;


static bool Cuda_Compiler_parseOptions(PyObject *pylist, Cuda_Compiler_Options *options)
{
	if (pylist == NULL || pylist == Py_None)
	{
		options->numOptions = 0;
		options->options = NULL;

		return true;
	}

	if (Py_TYPE(pylist) != &PyList_Type)
	{
		PyErr_Format(
			PyExc_TypeError, "options must be %s, not %s", (&PyList_Type)->tp_name, Py_TYPE(pylist)->tp_name
		);
		goto error_1;
	}

	Py_ssize_t size; size = PyList_GET_SIZE(pylist);

	options->numOptions = (int)size;
	options->options = (const char **)TRACE_MALLOC(sizeof(*options->options) * size);

	for (Py_ssize_t i = 0; i < size; i += 1)
	{
		PyObject *item = PyList_GET_ITEM(pylist, i);

		if (Py_TYPE(item) != &PyUnicode_Type)
		{
			PyErr_Format(
				PyExc_TypeError, "options #%d item must be %s, not %s", (int)(i + 1),
				(&PyUnicode_Type)->tp_name, Py_TYPE(item)->tp_name
			);
			goto error_2;
		}

		options->options[i] = PyUnicode_AsUTF8AndSize(item, NULL);
	}

	return true;

error_2:
	TRACE_FREE((void *)options->options);

error_1:
	return false;
}


typedef struct Cuda_Compiler_Includes
{
	int numHeaders;
	const char **headers, **includeNames;
}
Cuda_Compiler_Includes;


static bool Cuda_Compiler_parseIncludes(PyObject *pydict, Cuda_Compiler_Includes *includes)
{
	if (pydict == NULL || pydict == Py_None)
	{
		includes->numHeaders = 0;
		includes->headers = includes->includeNames = NULL;

		return true;
	}

	if (Py_TYPE(pydict) != &PyDict_Type)
	{
		PyErr_Format(
			PyExc_TypeError, "includes must be %s, not %s", (&PyDict_Type)->tp_name, Py_TYPE(pydict)->tp_name
		);
		goto error_1;
	}

	Py_ssize_t size; size = PyDict_Size(pydict);
	includes->numHeaders = (int)size;

	includes->headers = (const char **)TRACE_MALLOC(sizeof(*includes->headers) * size);
	includes->includeNames = (const char **)TRACE_MALLOC(sizeof(*includes->includeNames) * size);

	PyObject *key, *value;

	int index; index = 0;
	Py_ssize_t pos; pos = 0;

	while (PyDict_Next(pydict, &pos, &key, &value))
	{
		if (Py_TYPE(key) != &PyUnicode_Type)
		{
			PyErr_Format(
				PyExc_TypeError, "includes key must be %s, not %s",
				(&PyUnicode_Type)->tp_name, Py_TYPE(key)->tp_name
			);
			goto error_2;
		}

		if (Py_TYPE(value) != &PyUnicode_Type)
		{
			PyErr_Format(
				PyExc_TypeError, "includes value must be %s, not %s",
				(&PyUnicode_Type)->tp_name, Py_TYPE(value)->tp_name
			);
			goto error_2;
		}

		includes->includeNames[index] = PyUnicode_AsUTF8AndSize(key, NULL);
		includes->headers[index] = PyUnicode_AsUTF8AndSize(value, NULL);

		index += 1;
	}

	return true;

error_2:
	TRACE_FREE((void *)includes->headers);
	TRACE_FREE((void *)includes->includeNames);

error_1:
	return false;
}


PyDoc_STRVAR(
	Cuda_Compiler_compile_doc,
	"compile(source, options=None, includes=None, name=None) -> Tuple[bytes, Optional[bytes]]"
);
static PyObject *Cuda_Compiler_compile(PyObject *self, PyObject *args, PyObject *kwds)
{
	(void)self;
	const char *kwlist[] = {"", "options", "includes", "name", NULL};

	const char *source;

	PyObject *pylist = NULL, *pydict = NULL;
	const char *name = NULL;

	PyObject *retval = NULL;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|$OOz", (char **)kwlist, &source, &pylist, &pydict, &name))
		goto error_1;

#if defined(TRACE_CUDA_DRIVER)
	fprintf(stderr, "[" CUDA_DRIVER_NAME "] Compiling program '%s' ...\n", (name == NULL) ? "<NONAME>" : name);
#endif

	Cuda_Compiler_Options options;

	if (!Cuda_Compiler_parseOptions(pylist, &options))
		goto error_1;

	Cuda_Compiler_Includes includes;

	if (!Cuda_Compiler_parseIncludes(pydict, &includes))
		goto error_2;

	nvrtcProgram prog;

	NVRTC_CHECK(nvrtcCreateProgram(
		&prog, source, name, includes.numHeaders, includes.headers, includes.includeNames
	), goto error_3);

	nvrtcResult status; status = nvrtcCompileProgram(prog, options.numOptions, options.options);

	if (status != NVRTC_ERROR_COMPILATION && status != NVRTC_SUCCESS)
	{
		PyErr_Format(Nvrtc_Error, "%s (%s:%d)", nvrtcGetErrorString(status), __FILE__, __LINE__);
		goto error_4;
	}

	char *ptx, *log;
	ptx = NULL, log = NULL;

	size_t ptxSizeRet; ptxSizeRet = 0;
	if (status == NVRTC_SUCCESS)
	{
		NVRTC_ASSERT(nvrtcGetPTXSize(prog, &ptxSizeRet));

		ptx = (char *)TRACE_MALLOC(sizeof(*ptx) * ptxSizeRet);
		NVRTC_ASSERT(nvrtcGetPTX(prog, ptx));
	}

	size_t logSizeRet;
	NVRTC_ASSERT(nvrtcGetProgramLogSize(prog, &logSizeRet));

	if (logSizeRet > 2)
	{
		log = (char *)TRACE_MALLOC(sizeof(*log) * logSizeRet);
		NVRTC_ASSERT(nvrtcGetProgramLog(prog, log));
	}

	retval = Py_BuildValue("(y#z)", ptx, (Py_ssize_clean_t)(ptxSizeRet > 0 ? ptxSizeRet - 1 : 0), log);
	TRACE_FREE(ptx), TRACE_FREE(log);

error_4:
	NVRTC_ASSERT(nvrtcDestroyProgram(&prog));

error_3:
	TRACE_FREE((void *)includes.headers);
	TRACE_FREE((void *)includes.includeNames);

error_2:
	TRACE_FREE((void *)options.options);

error_1:
	return retval;
}


#if defined(CUDA_BACKEND_IS_CUDA)

PyDoc_STRVAR(Cuda_Driver_profilerStop_doc, "profilerStop()");
static PyObject *Cuda_Driver_profilerStop(PyObject *self, PyObject *args)
{
	(void)self, (void)args;

	CU_ENFORCE(cudaProfilerStop());
	Py_RETURN_NONE;
}

#endif


PyDoc_STRVAR(Cuda_Driver_traceLeaks_doc, "traceLeaks() -> List[Tuple[int, str, int]]");
static PyObject *Cuda_Driver_traceLeaks(PyObject *self, PyObject *args)
{
	(void)self, (void)args;
	size_t nleaks = TraceMalloc_traceLeaks();

	PyObject *leaks = PyList_New(nleaks);
	if (leaks == NULL)
		return NULL;

	size_t index = 0;
	if (!TraceMalloc_Iterator_init())
		return leaks;

	do
	{
		size_t size;
		const char *file;
		int line;

		TraceMalloc_Iterator_item(&size, &file, &line);

		PyObject *leak = Py_BuildValue("(nsi)", (Py_ssize_t)size, file, line);
		if (leak == NULL)
			goto error;

		PyList_SET_ITEM(leaks, index, leak);
		index += 1;
	}
	while (TraceMalloc_Iterator_move());

	TraceMalloc_Iterator_dealloc();
	return leaks;

error:
	TraceMalloc_Iterator_dealloc();
	Py_DECREF(leaks);

	return NULL;
}


#if defined(__GNUC__)
	#pragma GCC diagnostic push
	#if __GNUC__ >= 8
		#pragma GCC diagnostic ignored "-Wcast-function-type"
	#endif
#endif

static PyMethodDef Cuda_Driver_methods[] = {
	{"getDriverVersion", Cuda_Driver_getDriverVersion, METH_NOARGS, Cuda_Driver_getDriverVersion_doc},
	{"getMemoryInfo", Cuda_Driver_getMemoryInfo, METH_NOARGS, Cuda_Driver_getMemoryInfo_doc},

	{"allocate", Cuda_Driver_pyAllocate, METH_VARARGS, Cuda_Driver_pyAllocate_doc},
	{"allocateFromIPCHandle", Cuda_Driver_allocateFromIPCHandle, METH_VARARGS, Cuda_Driver_allocateFromIPCHandle_doc},
	{"memcpy2D", (PyCFunction)Cuda_Driver_memcpy2D, METH_VARARGS | METH_KEYWORDS, Cuda_Driver_memcpy2D_doc},

	{"getCompilerVersion", Cuda_Compiler_getCompilerVersion, METH_NOARGS, Cuda_Compiler_getCompilerVersion_doc},
	{"compile", (PyCFunction)Cuda_Compiler_compile, METH_VARARGS | METH_KEYWORDS, Cuda_Compiler_compile_doc},

#if defined(CUDA_BACKEND_IS_CUDA)
	{"profilerStop", Cuda_Driver_profilerStop, METH_NOARGS, Cuda_Driver_profilerStop_doc},
#endif

	{"traceLeaks", Cuda_Driver_traceLeaks, METH_NOARGS, Cuda_Driver_traceLeaks_doc},
	{NULL, NULL, 0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif

static PyModuleDef Cuda_Driver_module = {
	PyModuleDef_HEAD_INIT,
	CUDA_DRIVER_NAME,
	NULL, 0,
	Cuda_Driver_methods,
	NULL, NULL, NULL, NULL
};


PyMODINIT_FUNC PyInit_Driver(void)
{
	import_array();

	PyObject *m = PyModule_Create(&Cuda_Driver_module);
	if (m == NULL)
		goto error_1;

	if (!Cuda_Device_moduleInit(m))    goto error_2;
	if (!Cuda_Stream_moduleInit(m))    goto error_3;
	if (!Cuda_Buffer_moduleInit(m))    goto error_4;
	if (!Cuda_GPUArray_moduleInit(m))  goto error_5;
	if (!Cuda_Allocator_moduleInit(m)) goto error_6;
	if (!Cuda_Module_moduleInit(m))    goto error_7;
	if (!CuRand_moduleInit(m))         goto error_8;
	if (!CuBlas_moduleInit(m))         goto error_9;

#if defined(CUDA_BACKEND_IS_CUDA)
	if (!CuDnn_moduleInit(m))          goto error_10;
#endif

	if (!createPyExc(m, CUDA_ERROR_NAME, CUDA_DRIVER_NAME "." CUDA_ERROR_NAME, &Cuda_Error))              goto error_11;
	if (!createPyExc(m, CUDA_NVRTC_ERROR_NAME, CUDA_DRIVER_NAME "." CUDA_NVRTC_ERROR_NAME, &Nvrtc_Error)) goto error_12;

	return m;

error_12:
	REMOVE_PY_OBJECT(&Cuda_Error);

error_11:
#if defined(CUDA_BACKEND_IS_CUDA)
	CuDnn_moduleDealloc();
error_10:
#endif

	CuBlas_moduleDealloc();
error_9:
	CuRand_moduleDealloc();
error_8:
	Cuda_Module_moduleDealloc();
error_7:
	Cuda_Allocator_moduleDealloc();
error_6:
	Cuda_GPUArray_moduleDealloc();
error_5:
	Cuda_Buffer_moduleDealloc();
error_4:
	Cuda_Stream_moduleDealloc();
error_3:
	Cuda_Device_moduleDealloc();
error_2:
	Py_DECREF(m);
error_1:
	return NULL;
}
