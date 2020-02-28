#include "Libs.h"


PyObject *CuRand_Error = NULL;


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


inline static bool curandCheckStatus(curandStatus_t code, const char *file, int line)
{
	if (code == CURAND_STATUS_SUCCESS)
		return true;

	const char *error = curandGetErrorString(code);
	PyErr_Format(CuRand_Error, "%s (%s:%d)\n", error, file, line);

	return false;
}


#define CURAND_CHECK(status, atexit) do { if (!curandCheckStatus((status), __FILE__, __LINE__)) { atexit; } } while (0)
#define CURAND_ENFORCE(status) CURAND_CHECK(status, return NULL)
#define CURAND_ASSERT(status) \
do { curandStatus_t code = (status); (void)code; assert(code == CURAND_STATUS_SUCCESS); } while (0)


PyDoc_STRVAR(CuRand_getVersion_doc, "getVersion() -> int");
static PyObject *CuRand_getVersion(PyObject *self, PyObject *args)
{
	(void)self, (void)args;

	int version;
	CURAND_ENFORCE(curandGetVersion(&version));

	return Py_BuildValue("i", version);
}


typedef struct CuRand_RNG
{
	PyObject_HEAD

	curandGenerator_t generator;
	curandRngType_t type;
	unsigned long long seed;
}
CuRand_RNG;


static PyObject *CuRand_RNG_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	const char *kwlist[] = {"type", "seed", "offset", NULL};

	int pytype;
	unsigned long long seed, offset = 0;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iK|K", (char **)kwlist, &pytype, &seed, &offset))
		goto error_1;

	CuRand_RNG *self; self = (CuRand_RNG *)type->tp_alloc(type, 0);
	if (self == NULL)
		goto error_1;

	self->type = (curandRngType_t)pytype;
	self->seed = seed;

	CURAND_CHECK(curandCreateGenerator(&self->generator, self->type), goto error_2);
	CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(self->generator, seed), goto error_3);
	CURAND_CHECK(curandSetGeneratorOffset(self->generator, offset), goto error_3);

#if defined(TRACE_CUDA_CURAND)
	fprintf(stderr, "[" CURAND_RNG_OBJNAME "] (0x%" PRIXMAX ") Allocated rng\n", (size_t)self);
#endif

	return (PyObject *)self;

error_3:
	CURAND_ASSERT(curandDestroyGenerator(self->generator));

error_2:
	self->generator = NULL;
	Py_DECREF(self);

error_1:
	return NULL;
}


static void CuRand_RNG_dealloc(PyObject *self)
{
	CuRand_RNG *pygen = (CuRand_RNG *)self;

	if (pygen->generator != NULL)
	{
		CURAND_ASSERT(curandDestroyGenerator(pygen->generator));

#if defined(TRACE_CUDA_CURAND)
		fprintf(stderr, "[" CURAND_RNG_OBJNAME "] (0x%" PRIXMAX ") Deallocated rng\n", (size_t)self);
#endif
	}

	Py_TYPE(self)->tp_free(self);
}


PyDoc_STRVAR(CuRand_RNG_move_doc, "move(self, offset)");
static PyObject *Curand_RNG_move(PyObject *self, PyObject *args)
{
	CuRand_RNG *pygen = (CuRand_RNG *)self;
	Py_ssize_t pyoffset;

	if (!PyArg_ParseTuple(args, "n", &pyoffset))
		return NULL;

	CURAND_ENFORCE(curandSetGeneratorOffset(pygen->generator, (unsigned long long)pyoffset));
	Py_RETURN_NONE;
}


PyDoc_STRVAR(CuRand_RNG_fillInteger_doc, "fillInteger(self, ary)");
static PyObject *CuRand_RNG_fillInteger(PyObject *self, PyObject *args)
{
	CuRand_RNG *pygen = (CuRand_RNG *)self;

	Cuda_GPUArray *pyary;
	if (!PyArg_ParseTuple(args, "O!", Cuda_GPUArray_Type, &pyary))
		return NULL;

	switch (pyary->dtype)
	{
		case DTYPE_INT32:
		case DTYPE_UINT32:
		{
			CURAND_ENFORCE(curandGenerate(pygen->generator, (unsigned *)pyary->gpudata->ptr, pyary->size));
			break;
		}
		default:
		{
			PyErr_SetString(PyExc_ValueError, "unsupported gpuarray dtype");
			return NULL;
		}
	}

	Py_RETURN_NONE;
}


PyDoc_STRVAR(CuRand_RNG_fillUniform_doc, "fillUniform(self, ary)");
static PyObject *CuRand_RNG_fillUniform(PyObject *self, PyObject *args)
{
	CuRand_RNG *pygen = (CuRand_RNG *)self;

	Cuda_GPUArray *pyary;
	if (!PyArg_ParseTuple(args, "O!", Cuda_GPUArray_Type, &pyary))
		return NULL;

	switch (pyary->dtype)
	{
		case DTYPE_FLOAT32:
		{
			CURAND_ENFORCE(curandGenerateUniform(pygen->generator, (float *)pyary->gpudata->ptr, pyary->size));
			break;
		}
		case DTYPE_FLOAT64:
		{
			CURAND_ENFORCE(curandGenerateUniformDouble(pygen->generator, (double *)pyary->gpudata->ptr, pyary->size));
			break;
		}
		default:
		{
			PyErr_SetString(PyExc_ValueError, "unsupported gpuarray dtype");
			return NULL;
		}
	}

	Py_RETURN_NONE;
}


PyDoc_STRVAR(CuRand_RNG_fillNormal_doc, "fillNormal(self, ary, mean=0.0, stddev=1.0)");
static PyObject *CuRand_RNG_fillNormal(PyObject *self, PyObject *args, PyObject *kwds)
{
	CuRand_RNG *pygen = (CuRand_RNG *)self;
	const char *kwlist[] = {"", "mean", "stddev", NULL};

	Cuda_GPUArray *pyary;
	double mean = 0.0, stddev = 1.0;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|$dd", (char **)kwlist, Cuda_GPUArray_Type, &pyary, &mean, &stddev))
		return NULL;

	switch (pyary->dtype)
	{
		case DTYPE_FLOAT32:
		{
			CURAND_ENFORCE(curandGenerateNormal(
				pygen->generator, (float *)pyary->gpudata->ptr, pyary->size, (float)mean, (float)stddev
			));
			break;
		}
		case DTYPE_FLOAT64:
		{
			CURAND_ENFORCE(curandGenerateNormalDouble(
				pygen->generator, (double *)pyary->gpudata->ptr, pyary->size, mean, stddev
			));
			break;
		}
		default:
		{
			PyErr_SetString(PyExc_ValueError, "unsupported gpuarray dtype");
			return NULL;
		}
	}

	Py_RETURN_NONE;
}


static PyMemberDef Cuda_RNG_members[] = {
	{(char *)"type", T_INT, offsetof(CuRand_RNG, type), READONLY, NULL},
	{(char *)"seed", T_ULONGLONG, offsetof(CuRand_RNG, seed), READONLY, NULL},
	{NULL, 0, 0, 0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic push
	#if __GNUC__ >= 8
		#pragma GCC diagnostic ignored "-Wcast-function-type"
	#endif
#endif

static PyMethodDef Cuda_RNG_methods[] = {
	{"move", Curand_RNG_move, METH_VARARGS, CuRand_RNG_move_doc},
	{"fillInteger", CuRand_RNG_fillInteger, METH_VARARGS, CuRand_RNG_fillInteger_doc},
	{"fillUniform", CuRand_RNG_fillUniform, METH_VARARGS, CuRand_RNG_fillUniform_doc},
	{"fillNormal", (PyCFunction)CuRand_RNG_fillNormal, METH_VARARGS | METH_KEYWORDS, CuRand_RNG_fillNormal_doc},
	{NULL, NULL, 0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif

static PyType_Slot CuRand_RNG_slots[] = {
	{Py_tp_new, (void *)CuRand_RNG_new},
	{Py_tp_dealloc, (void *)CuRand_RNG_dealloc},
	{Py_tp_members, Cuda_RNG_members},
	{Py_tp_methods, Cuda_RNG_methods},
	{0, NULL}
};

static PyType_Spec CuRand_RNG_TypeSpec = {
	CURAND_BACKEND_NAME "." CURAND_RNG_OBJNAME,
	sizeof(CuRand_RNG),
	0,
	Py_TPFLAGS_DEFAULT,
	CuRand_RNG_slots
};


PyTypeObject *CuRand_RNG_Type = NULL;


static PyMethodDef CuRand_methods[] = {
	{"getVersion", CuRand_getVersion, METH_NOARGS, CuRand_getVersion_doc},
	{NULL, NULL, 0, NULL}
};

static PyModuleDef CuRand_module = {
	PyModuleDef_HEAD_INIT,
	CURAND_BACKEND_NAME,
	NULL, 0,
	CuRand_methods,
	NULL, NULL, NULL, NULL
};


bool CuRand_moduleInit(PyObject *module)
{
	PyObject *m = PyModule_Create(&CuRand_module);
	if (m == NULL)
		goto error_1;

	if (!createPyClass(m, CURAND_RNG_OBJNAME, &CuRand_RNG_TypeSpec, &CuRand_RNG_Type))                goto error_2;
	if (!createPyExc(m, CURAND_ERROR_NAME, CURAND_BACKEND_NAME "." CURAND_ERROR_NAME, &CuRand_Error)) goto error_3;

	PyModule_AddIntConstant(m, "RAND_RNG_TEST", CURAND_RNG_TEST);
	PyModule_AddIntConstant(m, "RAND_RNG_PSEUDO_DEFAULT", CURAND_RNG_PSEUDO_DEFAULT);
	PyModule_AddIntConstant(m, "RAND_RNG_PSEUDO_XORWOW", CURAND_RNG_PSEUDO_XORWOW);
	PyModule_AddIntConstant(m, "RAND_RNG_PSEUDO_MRG32K3A", CURAND_RNG_PSEUDO_MRG32K3A);
	PyModule_AddIntConstant(m, "RAND_RNG_PSEUDO_MTGP32", CURAND_RNG_PSEUDO_MTGP32);
	PyModule_AddIntConstant(m, "RAND_RNG_PSEUDO_MT19937", CURAND_RNG_PSEUDO_MT19937);
	PyModule_AddIntConstant(m, "RAND_RNG_PSEUDO_PHILOX4_32_10", CURAND_RNG_PSEUDO_PHILOX4_32_10);

	PyModule_AddIntConstant(m, "RAND_RNG_QUASI_DEFAULT", CURAND_RNG_QUASI_DEFAULT);
	PyModule_AddIntConstant(m, "RAND_RNG_QUASI_SOBOL32", CURAND_RNG_QUASI_SOBOL32);
	PyModule_AddIntConstant(m, "RAND_RNG_QUASI_SCRAMBLED_SOBOL32", CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
	PyModule_AddIntConstant(m, "RAND_RNG_QUASI_SOBOL64", CURAND_RNG_QUASI_SOBOL64);
	PyModule_AddIntConstant(m, "RAND_RNG_QUASI_SCRAMBLED_SOBOL64", CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);

	if (PyModule_AddObject(module, CURAND_BACKEND_NAME, m) < 0)
		goto error_4;

	return true;

error_4:
	REMOVE_PY_OBJECT(&CuRand_Error);
error_3:
	REMOVE_PY_OBJECT(&CuRand_RNG_Type);
error_2:
	Py_DECREF(m);
error_1:
	return false;
}


void CuRand_moduleDealloc(void)
{
	REMOVE_PY_OBJECT(&CuRand_Error);
	REMOVE_PY_OBJECT(&CuRand_RNG_Type);
}
