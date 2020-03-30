#include "Driver.h"


static bool Cuda_Buffer_init(Cuda_Buffer *self, Cuda_Ptr ptr, size_t size, int device, PyObject *parent)
{
	if (ptr == NULL)
	{
		assert(parent == NULL || Py_TYPE(parent) == Cuda_MemoryPool_Type);
		cudaError_t status = cudaMalloc(&ptr, size);

		if (status == cudaErrorMemoryAllocation)
		{
			PyErr_Format(PyExc_MemoryError, "out of device memory while allocating %" PRIuMAX " bytes", size);
			goto error;
		}

		CUDA_CHECK(status, goto error);

#if defined(TRACE_CUDA_DRIVER)
		fprintf(
			stderr, "[" CUDA_BUFFER_OBJNAME "] (0x%" PRIXMAX ") Allocated buffer of %" PRIuMAX " bytes\n",
			(size_t)self, size
		);
#endif
	}
	else if (parent != NULL)
	{
#if defined(TRACE_CUDA_DRIVER)
		if (Py_TYPE(parent) == Cuda_Buffer_Type)
			fprintf(
				stderr, "[" CUDA_BUFFER_OBJNAME "] (0x%" PRIXMAX ") Allocated buffer of %" PRIuMAX
				" bytes as SLICE of 0x%" PRIXMAX "\n", (size_t)self, size, (size_t)parent
			);
		else
			fprintf(
				stderr, "[" CUDA_BUFFER_OBJNAME "] (0x%" PRIXMAX ") Allocated buffer of %" PRIuMAX
				" bytes from MEMORY POOL 0x%" PRIXMAX "\n", (size_t)self, size, (size_t)parent
			);
#endif
	}

	self->ptr = ptr;
	self->size = size;

	if (parent != NULL)
		Py_INCREF(parent);

	self->parent = parent;

	self->device = device;
	self->ipc = false;

	return true;

error:
	self->ptr = NULL;
	return false;
}


static bool Cuda_Buffer_initFromIPCHandle(Cuda_Buffer *self, cudaIpcMemHandle_t handle, size_t size, int device)
{
	Cuda_Ptr ptr;
	CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess), goto error);

#if defined(TRACE_CUDA_DRIVER)
	fprintf(
		stderr, "[" CUDA_BUFFER_OBJNAME "] (0x%" PRIXMAX ") Allocated ipc buffer of %" PRIuMAX " bytes\n",
		(size_t)self, size
	);
#endif

	self->ptr = ptr;
	self->size = size;

	self->parent = NULL;

	self->device = device;
	self->ipc = true;

	return true;

error:
	self->ptr = NULL;
	return false;
}


void Cuda_Buffer_free(Cuda_Buffer *self)
{
	if (self->ptr == NULL)
		return;

	if (self->parent == NULL)
	{
		if (self->ipc)
		{
			CUDA_ASSERT(cudaIpcCloseMemHandle(self->ptr));

#if defined(TRACE_CUDA_DRIVER)
			fprintf(
				stderr, "[" CUDA_BUFFER_OBJNAME "] (0x%" PRIXMAX ") Deallocated ipc buffer of %" PRIuMAX " bytes\n",
				(size_t)self, self->size
			);
#endif
		}
		else
		{
			CUDA_ASSERT(cudaFree(self->ptr));

#if defined(TRACE_CUDA_DRIVER)
			fprintf(
				stderr, "[" CUDA_BUFFER_OBJNAME "] (0x%" PRIXMAX ") Deallocated buffer of %" PRIuMAX " bytes\n",
				(size_t)self, self->size
			);
#endif
		}
	}
	else if (Py_TYPE(self->parent) == Cuda_MemoryPool_Type)
	{
		Cuda_MemoryPool *pypool = (Cuda_MemoryPool *)self->parent;
		Cuda_MemoryPool_hold(pypool, self);

#if defined(TRACE_CUDA_DRIVER)
		fprintf(
			stderr, "[" CUDA_BUFFER_OBJNAME "] (0x%" PRIXMAX ") Returned buffer of %" PRIuMAX
			" bytes to memory pool 0x%" PRIXMAX "\n", (size_t)self, self->size, (size_t)pypool
		);
#endif
	}
	else
	{
#if defined(TRACE_CUDA_DRIVER)
		fprintf(
			stderr, "[" CUDA_BUFFER_OBJNAME "] (0x%" PRIXMAX ") Deallocated buffer of %" PRIuMAX
			" bytes as slice of 0x%" PRIXMAX "\n", (size_t)self, self->size, (size_t)self->parent
		);
#endif
	}

	self->ptr = NULL;

	if (self->parent != NULL)
		Py_DECREF(self->parent);

	self->parent = NULL;
}


static void Cuda_Buffer_dealloc(PyObject *self)
{
	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;
	Cuda_Buffer_free(pybuf);

	PyObject_Del(self);
}


Cuda_Buffer *Cuda_Buffer_new(Cuda_Ptr ptr, size_t size, int device, PyObject *parent)
{
	Cuda_Buffer *self = PyObject_NEW(Cuda_Buffer, Cuda_Buffer_Type);
	if (self == NULL)
		return NULL;

	if (!Cuda_Buffer_init(self, ptr, size, device, parent))
		goto error;

	return self;

error:
	Py_DECREF(self);
	return NULL;
}


Cuda_Buffer *Cuda_Buffer_newFromIPCHandle(cudaIpcMemHandle_t handle, size_t size)
{
	int device;
	CUDA_ENFORCE(cudaGetDevice(&device));

	Cuda_Buffer *self = PyObject_NEW(Cuda_Buffer, Cuda_Buffer_Type);
	if (self == NULL)
		return NULL;

	if (!Cuda_Buffer_initFromIPCHandle(self, handle, size, device))
		goto error;

	return self;

error:
	Py_DECREF(self);
	return NULL;
}


PyDoc_STRVAR(Cuda_Buffer_pyFree_doc, "free(self)");
static PyObject *Cuda_Buffer_pyFree(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_Buffer_free((Cuda_Buffer *)self);

	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Buffer_get_doc, "get(self) -> numpy.ndarray");
static PyObject *Cuda_Buffer_get(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;

	npy_intp dims[1];
	dims[0] = pybuf->size;

	PyArrayObject *ary = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_UBYTE, 0);
	if (ary == NULL)
		goto error_1;

	CU_CHECK(cuMemcpyDtoH(PyArray_DATA(ary), (CUdeviceptr)pybuf->ptr, pybuf->size), goto error_2);
	return (PyObject *)ary;

error_2:
	Py_DECREF(ary);

error_1:
	return NULL;
}


PyDoc_STRVAR(Cuda_Buffer_set_doc, "set(self, ary)");
static PyObject *Cuda_Buffer_set(PyObject *self, PyObject *args)
{
	PyArrayObject *ary;

	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &ary))
		return NULL;

	size_t nbytes = PyArray_NBYTES(ary);
	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;

	if (nbytes != pybuf->size)
	{
		PyErr_Format(
			PyExc_ValueError, "cannot copy input array of %" PRIuMAX " bytes to buffer of %" PRIuMAX " bytes",
			nbytes, pybuf->size
		);
		return NULL;
	}

	if (!PyArray_IS_C_CONTIGUOUS(ary))
	{
		PyErr_SetString(PyExc_ValueError, "input array is not contiguous");
		return NULL;
	}

	CU_ENFORCE(cuMemcpyHtoD((CUdeviceptr)pybuf->ptr, PyArray_DATA(ary), pybuf->size));
	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Buffer_fillD8_doc, "fillD8(self, byte)");
static PyObject *Cuda_Buffer_fillD8(PyObject *self, PyObject *args)
{
	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;
	unsigned char uc;

	if (!PyArg_ParseTuple(args, "B", &uc))
		return NULL;

	CU_ENFORCE(cuMemsetD8((CUdeviceptr)pybuf->ptr, uc, pybuf->size));
	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Buffer_fillD16_doc, "fillD16(self, word)");
static PyObject *Cuda_Buffer_fillD16(PyObject *self, PyObject *args)
{
	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;
	unsigned short us;

	if (pybuf->size % 2 != 0)
	{
		PyErr_Format(PyExc_ValueError, "buffer size (%" PRIuMAX ") is not multiple of 2", pybuf->size);
		return NULL;
	}

	if (!PyArg_ParseTuple(args, "H", &us))
		return NULL;

	CU_ENFORCE(cuMemsetD16((CUdeviceptr)pybuf->ptr, us, pybuf->size / 2));
	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Buffer_fillD32_doc, "fillD32(self, dword)");
static PyObject *Cuda_Buffer_fillD32(PyObject *self, PyObject *args)
{
	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;
	unsigned int ui;

	if (pybuf->size % 4 != 0)
	{
		PyErr_Format(PyExc_ValueError, "buffer size (%" PRIuMAX ") is not multiple of 4", pybuf->size);
		return NULL;
	}

	if (!PyArg_ParseTuple(args, "I", &ui))
		return NULL;

	CU_ENFORCE(cuMemsetD32((CUdeviceptr)pybuf->ptr, ui, pybuf->size / 4));
	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Buffer_copy_doc, "copy(self, dst=None, allocator=None) -> " CUDA_DRIVER_NAME "." CUDA_BUFFER_OBJNAME);
static PyObject *Cuda_Buffer_copy(PyObject *self, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"dst", "allocator", NULL};

	Cuda_Buffer *dst = NULL;
	Cuda_MemoryPool *allocator = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "|O!O!", (char **)kwlist, Cuda_Buffer_Type, &dst, Cuda_MemoryPool_Type, &allocator
	))
		goto error_1;

	Cuda_Buffer *pybuf; pybuf = (Cuda_Buffer *)self;
	size_t nbytes; nbytes = pybuf->size;

	if (dst == NULL)
		dst = (allocator != NULL) ? Cuda_MemoryPool_allocate(allocator, nbytes) :
			Cuda_Driver_allocateWithKnownDevice(nbytes, pybuf->device);

	else
	{
		if (pybuf->size != dst->size)
		{
			PyErr_Format(
				PyExc_ValueError, "source size (%" PRIuMAX ") and destination size (%" PRIuMAX ") are not equal",
				pybuf->size, dst->size
			);
			goto error_1;
		}

		Py_INCREF(dst);
	}

	if (dst == NULL)
		goto error_1;

	CU_CHECK(cuMemcpyDtoD((CUdeviceptr)dst->ptr, (CUdeviceptr)pybuf->ptr, nbytes), goto error_2);
	return (PyObject *)dst;

error_2:
	Py_DECREF(dst);

error_1:
	return NULL;
}


PyDoc_STRVAR(Cuda_Buffer_getIPCHandle_doc, "getIPCHandle(self) -> bytes");
static PyObject *Cuda_Buffer_getIPCHandle(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;

	cudaIpcMemHandle_t handle;
	CUDA_ENFORCE(cudaIpcGetMemHandle(&handle, pybuf->ptr));

	char buffer[sizeof(handle)];
	memcpy(buffer, &handle, sizeof(handle));

	return Py_BuildValue("y#", buffer, sizeof(buffer));
}


Cuda_Buffer *Cuda_Buffer_getSlice(Cuda_Buffer *self, size_t start, size_t size)
{
	return Cuda_Buffer_new((char *)self->ptr + start, size, self->device, (PyObject *)self);
}


static PyObject *Cuda_Buffer_pyGetSlice(PyObject *self, PyObject *slice)
{
	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;
	Py_ssize_t start, stop, step;

	if (Py_TYPE(slice) != &PySlice_Type)
	{
		PyErr_Format(PyExc_TypeError, "subscript must be %s, not %s", (&PySlice_Type)->tp_name, Py_TYPE(self)->tp_name);
		return NULL;
	}

	Py_ssize_t length = pybuf->size, slicelength;
	if (PySlice_GetIndicesEx(slice, length, &start, &stop, &step, &slicelength) < 0)
		return NULL;

	if (step != 1)
	{
		PyErr_Format(PyExc_ValueError, "slice step %" PRIuMAX " is not contiguous", (size_t)step);
		return NULL;
	}

	return (PyObject *)Cuda_Buffer_getSlice(pybuf, start, slicelength);
}


static PyObject *Cuda_Buffer_toString(PyObject *self)
{
	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;
	PyObject *str = NULL;

	if (pybuf->ptr == NULL)
	{
		PyErr_SetString(Cuda_Error, "buffer is freed");
		goto error_1;
	}

	int device;
	CUDA_CHECK(cudaGetDevice(&device), goto error_1);
	CUDA_CHECK(cudaSetDevice(pybuf->device), goto error_1);

	PyObject *ary; ary = Cuda_Buffer_get(self, NULL);
	if (ary == NULL)
		goto error_2;

	str = PyObject_Str(ary);
	Py_DECREF(ary);

error_2:
	if (pybuf->device != device)
		CUDA_ASSERT(cudaSetDevice(device));

error_1:
	return str;
}


static Py_ssize_t Cuda_Buffer_getSize(PyObject *self)
{
	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;
	return pybuf->size;
}


static PyObject *Cuda_Buffer_getPtr(PyObject *self, void *closure)
{
	(void)closure;

	Cuda_Buffer *pybuf = (Cuda_Buffer *)self;
	return PyLong_FromSize_t((size_t)pybuf->ptr);
}


static PyGetSetDef Cuda_Buffer_getset[] = {
	{(char *)"ptr", Cuda_Buffer_getPtr, NULL, NULL, NULL},
	{NULL, NULL, NULL, NULL, NULL}
};

static PyMemberDef Cuda_Buffer_members[] = {
	{(char *)"parent", T_OBJECT, offsetof(Cuda_Buffer, parent), READONLY, NULL},
	{(char *)"size", T_PYSSIZET, offsetof(Cuda_Buffer, size), READONLY, NULL},
	{(char *)"device", T_INT, offsetof(Cuda_Buffer, device), READONLY, NULL},
	{(char *)"ipc", T_BOOL, offsetof(Cuda_Buffer, ipc), READONLY, NULL},
	{NULL, 0, 0, 0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic push
	#if __GNUC__ >= 8
		#pragma GCC diagnostic ignored "-Wcast-function-type"
	#endif
#endif

static PyMethodDef Cuda_Buffer_methods[] = {
	{"free", Cuda_Buffer_pyFree, METH_NOARGS, Cuda_Buffer_pyFree_doc},

	{"get", Cuda_Buffer_get, METH_NOARGS, Cuda_Buffer_get_doc},
	{"set", Cuda_Buffer_set, METH_VARARGS, Cuda_Buffer_set_doc},

	{"fillD8", Cuda_Buffer_fillD8, METH_VARARGS, Cuda_Buffer_fillD8_doc},
	{"fillD16", Cuda_Buffer_fillD16, METH_VARARGS, Cuda_Buffer_fillD16_doc},
	{"fillD32", Cuda_Buffer_fillD32, METH_VARARGS, Cuda_Buffer_fillD32_doc},

	{"copy", (PyCFunction)Cuda_Buffer_copy, METH_VARARGS | METH_KEYWORDS, Cuda_Buffer_copy_doc},
	{"getIPCHandle", Cuda_Buffer_getIPCHandle, METH_NOARGS, Cuda_Buffer_getIPCHandle_doc},

	{NULL, NULL, 0, NULL}
};

static PyType_Slot Cuda_Buffer_slots[] = {
	{Py_tp_str, (void *)Cuda_Buffer_toString},
	{Py_tp_repr, (void *)Cuda_Buffer_toString},
	{Py_tp_dealloc, (void *)Cuda_Buffer_dealloc},
	{Py_mp_length, (void *)Cuda_Buffer_getSize},
	{Py_mp_subscript, (void *)Cuda_Buffer_pyGetSlice},
	{Py_tp_getset, Cuda_Buffer_getset},
	{Py_tp_members, Cuda_Buffer_members},
	{Py_tp_methods, Cuda_Buffer_methods},
	{0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif

static PyType_Spec Cuda_Buffer_TypeSpec = {
	CUDA_DRIVER_NAME "." CUDA_BUFFER_OBJNAME,
	sizeof(Cuda_Buffer),
	0,
	Py_TPFLAGS_DEFAULT,
	Cuda_Buffer_slots
};


PyTypeObject *Cuda_Buffer_Type = NULL;


bool Cuda_Buffer_moduleInit(PyObject *m)
{
	if (!createPyClass(m, CUDA_BUFFER_OBJNAME, &Cuda_Buffer_TypeSpec, &Cuda_Buffer_Type))
		return false;

	return true;
}


void Cuda_Buffer_moduleDealloc(void)
{
	REMOVE_PY_OBJECT(&Cuda_Buffer_Type);
}
