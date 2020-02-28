#if defined(_MSC_VER)
	#include <intrin.h>
#endif

#include "Driver.h"


inline static uint32_t bitlog2(size_t mask)
{
#if defined(__GNUC__) || defined(__clang__)
	uint32_t index = (mask != 0) ? (sizeof(mask) * 8 - 1) - __builtin_clzll(mask) : 0;

#elif defined(_MSC_VER)
	unsigned long ul;
	if (_BitScanReverse64(&ul, mask) == 0)
		return 0;

	uint32_t index = (uint32_t)ul;

#else
	#error

#endif

	return index;
}


static uint32_t Cuda_MemoryPool_binNumber(size_t nbytes)
{
	if (nbytes > (1 << MEMORY_POOL_MIN_BIN_BITS))
	{
		uint32_t msb = bitlog2(nbytes);
		uint32_t exponent = msb << MEMORY_POOL_MANTISSA_BITS;

		uint32_t shift = msb - MEMORY_POOL_MANTISSA_BITS;
		uint32_t mask = (1 << MEMORY_POOL_MANTISSA_BITS) - 1;

		uint32_t mantissa = (nbytes >> shift) & mask;

		uint32_t bin = exponent | mantissa;
		bin = nbytes & (((size_t)1 << shift) - 1) ? bin + 1 : bin;

		bin -= MEMORY_POOL_OFFSET;
		return bin;
	}
	else return 0;
}


static size_t Cuda_MemoryPool_allocSize(uint32_t bin)
{
	if (bin > 0)
	{
		bin += MEMORY_POOL_OFFSET;

		uint32_t msb = bin >> MEMORY_POOL_MANTISSA_BITS;
		uint32_t shift = msb - MEMORY_POOL_MANTISSA_BITS;

		uint32_t mask = (1 << MEMORY_POOL_MANTISSA_BITS) - 1;
		uint32_t mantissa = bin & mask;

		size_t allocSize = (((size_t)1 << MEMORY_POOL_MANTISSA_BITS) | mantissa) << shift;
		return allocSize;
	}
	else return 1 << MEMORY_POOL_MIN_BIN_BITS;
}


typedef struct Cuda_MemoryPool_Result
{
	Cuda_Ptr ptr;
	size_t nbytes;
}
Cuda_MemoryPool_Result;


static void Cuda_MemoryPool_initBins(Cuda_MemoryPool *self)
{
	for (size_t i = 0; i < MEMORY_POOL_NUM_OF_BINS; i += 1)
		Cuda_AllocVector_init(&self->bins[i]);
}


static Cuda_MemoryPool_Result Cuda_MemoryPool_get(Cuda_MemoryPool *self, size_t nbytes)
{
	assert(nbytes < 1ULL << (sizeof(nbytes) * 8 - 1));
	uint32_t bin = Cuda_MemoryPool_binNumber(nbytes);

	size_t allocSize = Cuda_MemoryPool_allocSize(bin);
	assert(Cuda_MemoryPool_binNumber(allocSize) == bin);

	Cuda_AllocVector *vector = &self->bins[bin];

	Cuda_Ptr ptr = NULL;
	Cuda_AllocVector_pop(vector, &ptr);

	Cuda_MemoryPool_Result result;
	result.ptr = ptr;
	result.nbytes = allocSize;

	return result;
}


static void Cuda_MemoryPool_insert(Cuda_MemoryPool *self, size_t nbytes, Cuda_Ptr ptr)
{
	uint32_t bin = Cuda_MemoryPool_binNumber(nbytes);
	assert(Cuda_MemoryPool_allocSize(bin) == nbytes);

	Cuda_AllocVector_append(&self->bins[bin], ptr);
}


static void Cuda_MemoryPool_clear(Cuda_MemoryPool *self)
{
	for (size_t i = 0; i < MEMORY_POOL_NUM_OF_BINS; i += 1)
		Cuda_AllocVector_clear(&self->bins[i]);
}


static void Cuda_MemoryPool_deallocBins(Cuda_MemoryPool *self)
{
	for (size_t i = 0; i < MEMORY_POOL_NUM_OF_BINS; i += 1)
		Cuda_AllocVector_dealloc(&self->bins[i]);
}


static PyObject *Cuda_MemoryPool_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)args, (void)kwds;

	int device;
	CUDA_ENFORCE(cudaGetDevice(&device));

	Cuda_MemoryPool *self = (Cuda_MemoryPool *)type->tp_alloc(type, 0);
	if (self == NULL)
		return NULL;

	Cuda_MemoryPool_initBins(self);
	self->activeBlocks = self->heldBlocks = 0;

	self->device = device;
	self->holding = true;

#if defined(TRACE_CUDA_DRIVER)
	fprintf(stderr, "[" CUDA_MEMORY_POOL_OBJNAME "] (0x%" PRIXMAX ") Allocated memory pool\n", (size_t)self);
#endif

	return (PyObject *)self;
}


static void Cuda_MemoryPool_dealloc(PyObject *self)
{
	Cuda_MemoryPool *pypool = (Cuda_MemoryPool *)self;
	Cuda_MemoryPool_deallocBins(pypool);

#if defined(TRACE_CUDA_DRIVER)
	fprintf(stderr, "[" CUDA_MEMORY_POOL_OBJNAME "] (0x%" PRIXMAX ") Deallocated memory pool\n", (size_t)pypool);
#endif

	Py_TYPE(self)->tp_free(self);
}


inline static void cudaAssertDeviceIsActive(int device)
{
	(void)device;

#if !defined(NDEBUG)
	int current;
	CUDA_ASSERT(cudaGetDevice(&current));

	assert(device == current);
#endif
}


Cuda_Buffer *Cuda_MemoryPool_allocate(Cuda_MemoryPool *self, size_t nbytes)
{
	cudaAssertDeviceIsActive(self->device);
	Cuda_MemoryPool_Result result = Cuda_MemoryPool_get(self, nbytes);

	if (result.ptr == NULL)
	{
#if defined(TRACE_CUDA_DRIVER)
		fprintf(
			stderr, "[" CUDA_MEMORY_POOL_OBJNAME "] (0x%" PRIXMAX ") MISSED allocation of %" PRIuMAX " bytes\n",
			(size_t)self, nbytes
		);
#endif
	}
	else
	{
#if defined(TRACE_CUDA_DRIVER)
		fprintf(
			stderr, "[" CUDA_MEMORY_POOL_OBJNAME "] (0x%" PRIXMAX ") GAVE allocation of %" PRIuMAX
			" bytes (requested %" PRIuMAX ")\n", (size_t)self, result.nbytes, nbytes
		);
#endif
	}

	Cuda_Buffer *pybuf = Cuda_Buffer_new(result.ptr, result.nbytes, self->device, (PyObject *)self);

	if (pybuf == NULL)
		goto error;

	if (result.ptr != NULL)
		self->heldBlocks -= 1;

	self->activeBlocks += 1;
	return pybuf;

error:
	if (result.ptr != NULL)
		Cuda_MemoryPool_insert(self, result.nbytes, result.ptr);

	return NULL;
}


PyDoc_STRVAR(Cuda_MemoryPool_pyAllocate_doc, "allocate(self, nbytes) -> " CUDA_GPUARRAY_FULLNAME);
static PyObject *Cuda_MemoryPool_pyAllocate(PyObject *self, PyObject *args)
{
	Py_ssize_t pysize;

	if (!PyArg_ParseTuple(args, "n", &pysize))
		return NULL;

	if (pysize < 0)
	{
		PyErr_Format(PyExc_ValueError, "invalid allocation size %" PRIdMAX, (ptrdiff_t)pysize);
		return NULL;
	}

	return (PyObject *)Cuda_MemoryPool_allocate((Cuda_MemoryPool *)self, pysize);
}


void Cuda_MemoryPool_hold(Cuda_MemoryPool *self, Cuda_Buffer *pybuf)
{
	self->activeBlocks -= 1;
	bool holding = self->holding;

	if (holding)
	{
		Cuda_MemoryPool_insert(self, pybuf->size, pybuf->ptr);
		self->heldBlocks += 1;

#if defined(TRACE_CUDA_DRIVER)
		fprintf(
			stderr, "[" CUDA_MEMORY_POOL_OBJNAME "] (0x%" PRIXMAX ") Got allocation of %" PRIuMAX
			" bytes from buffer 0x%" PRIXMAX "\n", (size_t)self, pybuf->size, (size_t)pybuf
		);
#endif
	}
	else
	{
		CUDA_ASSERT(cudaFree(pybuf->ptr));

#if defined(TRACE_CUDA_DRIVER)
		fprintf(
			stderr, "[" CUDA_MEMORY_POOL_OBJNAME "] (0x%" PRIXMAX ") Released allocation of %" PRIuMAX
			" bytes from buffer 0x%" PRIXMAX "\n", (size_t)self, pybuf->size, (size_t)pybuf
		);
#endif
	}
}


PyDoc_STRVAR(Cuda_MemoryPool_stopHolding_doc, "stopHolding(self)");
static PyObject *Cuda_MemoryPool_stopHolding(PyObject *self, PyObject *args)
{
	(void)args;

	Cuda_MemoryPool *pypool = (Cuda_MemoryPool *)self;
	pypool->holding = false;

	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_MemoryPool_freeHeld_doc, "freeHeld(self)");
static PyObject *Cuda_MemoryPool_freeHeld(PyObject *self, PyObject *args)
{
	(void)self, (void)args;

	Cuda_MemoryPool *pypool = (Cuda_MemoryPool *)self;
	Cuda_MemoryPool_clear(pypool);

	pypool->heldBlocks = 0;
	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_MemoryPool_getStats_doc, "getStats(self) -> Dict");
static PyObject *Cuda_MemoryPool_getStats(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_MemoryPool *pypool = (Cuda_MemoryPool *)self;

	PyObject *bins = PyDict_New();
	if (bins == NULL)
		return NULL;

	size_t poolSize = 0;

	for (uint32_t i = 0; i < MEMORY_POOL_NUM_OF_BINS; i += 1)
	{
		size_t allocSize = Cuda_MemoryPool_allocSize(i), binSize = pypool->bins[i].size;
		poolSize += binSize * allocSize;

		PyObject *pyAllocSize = PyLong_FromSize_t(allocSize);
		if (pyAllocSize == NULL)
			goto error_1;

		PyObject *pyBinSize = PyLong_FromSize_t(binSize);
		if (pyBinSize == NULL)
			goto error_2;

		if (PyDict_SetItem(bins, pyAllocSize, pyBinSize) < 0)
			goto error_3;

		Py_DECREF(pyBinSize);
		continue;

error_3:
		Py_DECREF(pyBinSize);

error_2:
		Py_DECREF(pyAllocSize);
		goto error_1;
	}

	PyObject *stats; stats = Py_BuildValue(
		"{sNsnsnsn}",
		"bins", bins, "poolSize", (Py_ssize_t)poolSize,
		"activeBlocks", (Py_ssize_t)pypool->activeBlocks, "heldBlocks", (Py_ssize_t)pypool->heldBlocks
	);

	return stats;

error_1:
	Py_DECREF(bins);
	return NULL;
}


static PyMemberDef Cuda_MemoryPool_members[] = {
	{(char *)"activeBlocks", T_PYSSIZET, offsetof(Cuda_MemoryPool, activeBlocks), READONLY, NULL},
	{(char *)"heldBlocks", T_PYSSIZET, offsetof(Cuda_MemoryPool, heldBlocks), READONLY, NULL},
	{(char *)"device", T_INT, offsetof(Cuda_MemoryPool, device), READONLY, NULL},
	{(char *)"isHolding", T_BOOL, offsetof(Cuda_MemoryPool, holding), READONLY, NULL},
	{NULL, 0, 0, 0, NULL}
};

static PyMethodDef Cuda_MemoryPool_methods[] = {
	{"allocate", Cuda_MemoryPool_pyAllocate, METH_VARARGS, Cuda_MemoryPool_pyAllocate_doc},
	{"stopHolding", Cuda_MemoryPool_stopHolding, METH_NOARGS, Cuda_MemoryPool_stopHolding_doc},
	{"freeHeld", Cuda_MemoryPool_freeHeld, METH_NOARGS, Cuda_MemoryPool_freeHeld_doc},
	{"getStats", Cuda_MemoryPool_getStats, METH_NOARGS, Cuda_MemoryPool_getStats_doc},
	{NULL, NULL, 0, NULL}
};

static PyType_Slot Cuda_MemoryPool_slots[] = {
	{Py_tp_new, (void *)Cuda_MemoryPool_new},
	{Py_tp_dealloc, (void *)Cuda_MemoryPool_dealloc},
	{Py_tp_members, Cuda_MemoryPool_members},
	{Py_tp_methods, Cuda_MemoryPool_methods},
	{0, NULL}
};

static PyType_Spec Cuda_MemoryPool_TypeSpec = {
	CUDA_DRIVER_NAME "." CUDA_MEMORY_POOL_OBJNAME,
	sizeof(Cuda_MemoryPool),
	0,
	Py_TPFLAGS_DEFAULT,
	Cuda_MemoryPool_slots
};


PyTypeObject *Cuda_MemoryPool_Type = NULL;


bool Cuda_Allocator_moduleInit(PyObject *m)
{
	if (!createPyClass(m, CUDA_MEMORY_POOL_OBJNAME, &Cuda_MemoryPool_TypeSpec, &Cuda_MemoryPool_Type))
		return false;

	return true;
}


void Cuda_Allocator_moduleDealloc(void)
{
	REMOVE_PY_OBJECT(&Cuda_MemoryPool_Type);
}
