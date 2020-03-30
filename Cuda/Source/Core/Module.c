#include "Driver.h"


enum
{
	KERNEL_ARGS_LIMIT = 64
};


static const char *blockKey = "block", *gridKey = "grid", *streamKey = "stream";
static PyObject *pyBlockKey = NULL, *pyGridKey = NULL, *pyStreamKey = NULL;


static bool Cuda_Function_init(Cuda_Function *self, Cuda_Module *pymod, const char *name)
{
	CU_CHECK(cuModuleGetFunction(&self->function, pymod->module, name), goto error);

	Py_INCREF(pymod);
	self->module = pymod;

	return true;

error:
	self->module = NULL;
	return false;
}


static Cuda_Function *Cuda_Function_new(Cuda_Module *pymod, const char *name)
{
	Cuda_Function *self = PyObject_NEW(Cuda_Function, Cuda_Function_Type);
	if (self == NULL)
		return NULL;

	if (!Cuda_Function_init(self, pymod, name))
		goto error;

	return self;

error:
	Py_DECREF(self);
	return NULL;
}


static void Cuda_Function_dealloc(PyObject *self)
{
	Cuda_Function *pyfunc = (Cuda_Function *)self;

	if (pyfunc->module != NULL)
		Py_DECREF(pyfunc->module);

	PyObject_Del(self);
}


inline static bool Cuda_Function_unpackKernelArgs(Cuda_Function *self, PyObject *args,
												  void *kernelArgs[KERNEL_ARGS_LIMIT])
{
	(void)self;
	Py_ssize_t size = PyTuple_GET_SIZE(args);

	if (size > KERNEL_ARGS_LIMIT)
	{
		PyErr_Format(PyExc_ValueError, "kernel arguments overflow (limit is %d)", KERNEL_ARGS_LIMIT);
		return false;
	}

	for (Py_ssize_t i = 0; i < size; i += 1)
	{
		PyObject *item = PyTuple_GET_ITEM(args, i);
		PyTypeObject *type = Py_TYPE(item);

		if (type == Cuda_GPUArray_Type)
		{
			Cuda_GPUArray *ary = (Cuda_GPUArray *)item;
			kernelArgs[i] = &ary->gpudata->ptr;

#if defined(TRACE_CUDA_DRIVER)
			fprintf(
				stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is gpuarray "
				"0x%" PRIXMAX "\n", (size_t)self, (int)(i + 1), (size_t)ary
			);
#endif
		}
		else if (
			type == &PyFloat32ArrType_Type || type == &PyInt32ArrType_Type || type == &PyHalfArrType_Type ||
			type == &PyInt8ArrType_Type || type == &PyInt64ArrType_Type || type == &PyUInt32ArrType_Type ||
			type == &PyUInt8ArrType_Type || type == &PyUInt64ArrType_Type || type == &PyInt16ArrType_Type ||
			type == &PyUInt16ArrType_Type || type == &PyFloat64ArrType_Type
		)
		{
			kernelArgs[i] = &PyArrayScalar_VAL(item, );

#if defined(TRACE_CUDA_DRIVER)
			if (type == &PyFloat32ArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is float32 %lf\n",
					(size_t)self, (int)(i + 1), (double)((PyFloat32ScalarObject *)item)->obval
				);
			else if (type == &PyFloat64ArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is float64 %lf\n",
					(size_t)self, (int)(i + 1), (double)((PyFloat64ScalarObject *)item)->obval
				);
			else if (type == &PyHalfArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is float16 0x%" PRIx16 "\n",
					(size_t)self, (int)(i + 1), (int16_t)((PyHalfScalarObject *)item)->obval
				);
			else if (type == &PyInt32ArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is int32 %" PRId64 "\n",
					(size_t)self, (int)(i + 1), (int64_t)((PyInt32ScalarObject *)item)->obval
				);
			else if (type == &PyInt64ArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is int64 %" PRId64 "\n",
					(size_t)self, (int)(i + 1), (int64_t)((PyInt64ScalarObject *)item)->obval
				);
			else if (type == &PyInt8ArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is int8 %" PRId64 "\n",
					(size_t)self, (int)(i + 1), (int64_t)((PyInt8ScalarObject *)item)->obval
				);
			else if (type == &PyInt16ArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is int16 %" PRId64 "\n",
					(size_t)self, (int)(i + 1), (int64_t)((PyInt16ScalarObject *)item)->obval
				);
			else if (type == &PyUInt32ArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is uint32 %" PRIu64 "\n",
					(size_t)self, (int)(i + 1), (uint64_t)((PyUInt32ScalarObject *)item)->obval
				);
			else if (type == &PyUInt64ArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is uint64 %" PRIu64 "\n",
					(size_t)self, (int)(i + 1), (uint64_t)((PyUInt64ScalarObject *)item)->obval
				);
			else if (type == &PyUInt8ArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is uint8 %" PRIu64 "\n",
					(size_t)self, (int)(i + 1), (uint64_t)((PyUInt8ScalarObject *)item)->obval
				);
			else if (type == &PyUInt16ArrType_Type)
				fprintf(
					stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is uint16 %" PRIu64 "\n",
					(size_t)self, (int)(i + 1), (uint64_t)((PyUInt16ScalarObject *)item)->obval
				);
#endif
		}
		else if (type == &PyBytes_Type)
		{
			PyBytesObject *ary = (PyBytesObject *)item;
			kernelArgs[i] = ary->ob_sval;

#if defined(TRACE_CUDA_DRIVER)
			fprintf(
				stderr,
				"[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is bytes object of %" PRIuMAX " bytes\n",
				(size_t)self, (int)(i + 1), (size_t)PyBytes_Size(item)
			);
#endif
		}
		else if (type == &PyByteArray_Type)
		{
			PyByteArrayObject *ary = (PyByteArrayObject *)item;
			kernelArgs[i] = ary->ob_start;

#if defined(TRACE_CUDA_DRIVER)
			fprintf(
				stderr,
				"[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is bytearray object of %" PRIuMAX
				" bytes\n", (size_t)self, (int)(i + 1), (size_t)PyByteArray_Size(item)
			);
#endif
		}
		else if (type == Cuda_Buffer_Type)
		{
			Cuda_Buffer *buffer = (Cuda_Buffer *)item;
			kernelArgs[i] = &buffer->ptr;

#if defined(TRACE_CUDA_DRIVER)
			fprintf(
				stderr, "[" CUDA_FUNCTION_OBJNAME "] (0x%" PRIXMAX ") Call: item #%d is buffer 0x%" PRIXMAX "\n",
				(size_t)self, (int)(i + 1), (size_t)buffer
			);
#endif
		}
		else
		{
			PyErr_Format(PyExc_TypeError, "unrecognized kernel argument #%d", (int)(i + 1));
			return false;
		}
	}

	return true;
}


inline static bool Cuda_Function_unpackBlockOrGrid(unsigned ary[3], PyObject *dict, const char *key, PyObject *pyKey)
{
	PyObject *pyary = PyDict_GetItem(dict, pyKey);
	if (pyary == NULL)
	{
		PyErr_SetString(PyExc_KeyError, key);
		return false;
	}

	if (!PyTuple_CheckExact(pyary) || PyTuple_GET_SIZE(pyary) != 3)
	{
		PyErr_Format(
			PyExc_TypeError, "%s must be 3-int %s, not %s", key, (&PyTuple_Type)->tp_name, Py_TYPE(pyary)->tp_name
		);
		return false;
	}

	for (Py_ssize_t i = 0; i < 3; i += 1)
	{
		PyObject *item = PyTuple_GET_ITEM(pyary, i);
		unsigned long value = PyLong_AsUnsignedLong(item);

		if (value == (unsigned long)-1 && PyErr_Occurred())
			return false;

		ary[i] = (unsigned)value;
	}

	return true;
}


static bool Cuda_Function_unpackStream(cudaStream_t *stream, PyObject *dict)
{
	PyObject *obj = PyDict_GetItem(dict, pyStreamKey);
	if (obj == NULL)
	{
		*stream = NULL;
		return true;
	}

	if (Py_TYPE(obj) != Cuda_Stream_Type)
	{
		PyErr_Format(
			PyExc_TypeError, "%s must be %s, not %s", streamKey, Cuda_Stream_Type->tp_name, Py_TYPE(obj)->tp_name
		);
		return false;
	}

	Cuda_Stream *pystream = (Cuda_Stream *)obj;
	*stream = pystream->stream;

	return true;
}


static PyObject *Cuda_Function_call(PyObject *self, PyObject *args, PyObject *kwds)
{
	(void)kwds;
	Cuda_Function *pyfunc = (Cuda_Function *)self;

	void *kernelArgs[KERNEL_ARGS_LIMIT];
	unsigned block[3], grid[3];

	if (!Cuda_Function_unpackKernelArgs(pyfunc, args, kernelArgs))
		return NULL;

	if (kwds == NULL)
	{
		PyErr_SetString(PyExc_ValueError, "must pass keyword arguments to kernel call");
		return NULL;
	}

	if (!Cuda_Function_unpackBlockOrGrid(block, kwds, blockKey, pyBlockKey))
		return NULL;

	if (!Cuda_Function_unpackBlockOrGrid(grid, kwds, gridKey, pyGridKey))
		return NULL;

	cudaStream_t stream;
	if (!Cuda_Function_unpackStream(&stream, kwds))
		return NULL;

	CU_ENFORCE(cuLaunchKernel(
		pyfunc->function, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, stream, kernelArgs, NULL
	));

	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Function_getAttribute_doc, "getAttribute(self, attrib) -> int");
static PyObject *Cuda_Function_getAttribute(PyObject *self, PyObject *args)
{
	Cuda_Function *pyfunc = (Cuda_Function *)self;
	int attrib;

	if (!PyArg_ParseTuple(args, "i", &attrib))
		return NULL;

	int value;
	CU_ENFORCE(cuFuncGetAttribute(&value, (CUfunction_attribute)attrib, pyfunc->function));

	return Py_BuildValue("i", value);
}


static PyMemberDef Cuda_Function_members[] = {
	{(char *)"module", T_OBJECT_EX, offsetof(Cuda_Function, module), READONLY, NULL},
	{NULL, 0, 0, 0, NULL}
};

static PyMethodDef Cuda_Function_methods[] = {
	{"getAttribute", Cuda_Function_getAttribute, METH_VARARGS, Cuda_Function_getAttribute_doc},
	{NULL, NULL, 0, NULL}
};

static PyType_Slot Cuda_Function_slots[] = {
	{Py_tp_dealloc, (void *)Cuda_Function_dealloc},
	{Py_tp_call, (void *)Cuda_Function_call},
	{Py_tp_members, Cuda_Function_members},
	{Py_tp_methods, Cuda_Function_methods},
	{0, NULL}
};

static PyType_Spec Cuda_Function_TypeSpec = {
	CUDA_DRIVER_NAME "." CUDA_FUNCTION_OBJNAME,
	sizeof(Cuda_Function),
	0,
	Py_TPFLAGS_DEFAULT,
	Cuda_Function_slots
};


PyTypeObject *Cuda_Function_Type = NULL;


static PyObject *Cuda_Module_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)kwds;
	Py_buffer image;

	if (!PyArg_ParseTuple(args, "s*", &image))
		goto error_1;

	Cuda_Module *self; self = (Cuda_Module *)type->tp_alloc(type, 0);
	if (self == NULL)
		goto error_2;

	CU_CHECK(cuModuleLoadData(&self->module, image.buf), goto error_3);

#if defined(TRACE_CUDA_DRIVER)
	fprintf(stderr, "[" CUDA_MODULE_OBJNAME "] (0x%" PRIXMAX ") Allocated module\n", (size_t)self);
#endif

	PyBuffer_Release(&image);
	return (PyObject *)self;

error_3:
	self->module = NULL;
	Py_DECREF(self);

error_2:
	PyBuffer_Release(&image);

error_1:
	return NULL;
}


static void Cuda_Module_dealloc(PyObject *self)
{
	Cuda_Module *pymod = (Cuda_Module *)self;

	if (pymod->module != NULL)
	{
		CU_ASSERT(cuModuleUnload(pymod->module));

#if defined(TRACE_CUDA_DRIVER)
		fprintf(stderr, "[" CUDA_MODULE_OBJNAME "] (0x%" PRIXMAX ") Deallocated module\n", (size_t)pymod);
#endif
	}

	Py_TYPE(self)->tp_free(self);
}


PyDoc_STRVAR(Cuda_Module_getFunction_doc, "getFunction(self, name) -> " CUDA_DRIVER_NAME "." CUDA_FUNCTION_OBJNAME);
static PyObject *Cuda_Module_getFunction(PyObject *self, PyObject *args)
{
	Cuda_Module *pymod = (Cuda_Module *)self;
	const char *name;

	if (!PyArg_ParseTuple(args, "s", &name))
		return NULL;

#if defined(TRACE_CUDA_DRIVER)
	fprintf(stderr, "[" CUDA_MODULE_OBJNAME "] (0x%" PRIXMAX ") Getting function '%s' ...\n", (size_t)pymod, name);
#endif

	return (PyObject *)Cuda_Function_new(pymod, name);
}


static PyMethodDef Cuda_Module_methods[] = {
	{"getFunction", Cuda_Module_getFunction, METH_VARARGS, Cuda_Module_getFunction_doc},
	{NULL, NULL, 0, NULL}
};

static PyType_Slot Cuda_Module_slots[] = {
	{Py_tp_new, (void *)Cuda_Module_new},
	{Py_tp_dealloc, (void *)Cuda_Module_dealloc},
	{Py_tp_methods, Cuda_Module_methods},
	{0, NULL}
};

static PyType_Spec Cuda_Module_TypeSpec = {
	CUDA_DRIVER_NAME "." CUDA_MODULE_OBJNAME,
	sizeof(Cuda_Module),
	0,
	Py_TPFLAGS_DEFAULT,
	Cuda_Module_slots
};


PyTypeObject *Cuda_Module_Type = NULL;


bool Cuda_Module_moduleInit(PyObject *m)
{
	if (!createPyClass(m, CUDA_MODULE_OBJNAME, &Cuda_Module_TypeSpec, &Cuda_Module_Type))       goto error_1;
	if (!createPyClass(m, CUDA_FUNCTION_OBJNAME, &Cuda_Function_TypeSpec, &Cuda_Function_Type)) goto error_2;

	pyBlockKey = PyUnicode_FromString(blockKey);
	if (pyBlockKey == NULL) goto error_3;

	pyGridKey = PyUnicode_FromString(gridKey);
	if (pyGridKey == NULL) goto error_4;

	pyStreamKey = PyUnicode_FromString(streamKey);
	if (pyStreamKey == NULL) goto error_5;

	PyModule_AddIntConstant(m, "FUNC_ATTR_MAX_THREADS_PER_BLOCK", CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
	PyModule_AddIntConstant(m, "FUNC_ATTR_SHARED_SIZE_BYTES", CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
	PyModule_AddIntConstant(m, "FUNC_ATTR_CONST_SIZE_BYTES", CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
	PyModule_AddIntConstant(m, "FUNC_ATTR_LOCAL_SIZE_BYTES", CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
	PyModule_AddIntConstant(m, "FUNC_ATTR_NUM_REGS", CU_FUNC_ATTRIBUTE_NUM_REGS);

	return true;

error_5:
	Py_DECREF(pyGridKey);
error_4:
	Py_DECREF(pyBlockKey);
error_3:
	REMOVE_PY_OBJECT(&Cuda_Function_Type);
error_2:
	REMOVE_PY_OBJECT(&Cuda_Module_Type);
error_1:
	return false;
}


void Cuda_Module_moduleDealloc(void)
{
	REMOVE_PY_OBJECT(&Cuda_Function_Type);
	REMOVE_PY_OBJECT(&Cuda_Module_Type);
}
