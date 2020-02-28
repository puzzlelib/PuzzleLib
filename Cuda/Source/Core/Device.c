#include "Driver.h"


static PyObject *Cuda_Device_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)kwds;
	int device;

	if (!PyArg_ParseTuple(args, "i", &device))
		return NULL;

	Cuda_Device *self = (Cuda_Device *)type->tp_alloc(type, 0);
	if (self == NULL)
		return NULL;

	self->index = device;
	return (PyObject *)self;
}


PyDoc_STRVAR(Cuda_Device_set_doc, "set(self)");
static PyObject *Cuda_Device_set(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_Device *pydevice = (Cuda_Device *)self;

	CUDA_ENFORCE(cudaSetDevice(pydevice->index));

#if defined(TRACE_CUDA_DRIVER)
	fprintf(stderr, "[" CUDA_DEVICE_OBJNAME "] (0x%" PRIXMAX ") Set device #%d\n", (size_t)self, pydevice->index);
#endif

	Py_INCREF(self);
	return self;
}


PyDoc_STRVAR(Cuda_Device_getAttribute_doc, "getAttribute(self, attrib) -> int");
static PyObject *Cuda_Device_getAttribute(PyObject *self, PyObject *args)
{
	Cuda_Device *pydevice = (Cuda_Device *)self;
	int attrib;

	if (!PyArg_ParseTuple(args, "i", &attrib))
		return NULL;

	int value;
	CUDA_ENFORCE(cudaDeviceGetAttribute(&value, (enum cudaDeviceAttr)attrib, pydevice->index));

	return Py_BuildValue("i", value);
}


PyDoc_STRVAR(Cuda_Device_name_doc, "name(self) -> str");
static PyObject *Cuda_Device_name(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_Device *pydevice = (Cuda_Device *)self;

	CUdevice device;
	CU_ENFORCE(cuDeviceGet(&device, pydevice->index));

	char name[256];
	CU_ENFORCE(cuDeviceGetName(name, sizeof(name), device));

	name[sizeof(name) - 1] = '\0';
	return Py_BuildValue("s", name);
}


PyDoc_STRVAR(Cuda_Device_computeCapability_doc, "computeCapability(self) -> Tuple[int, int]");
static PyObject *Cuda_Device_computeCapability(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_Device *pydevice = (Cuda_Device *)self;

	int major, minor;

	CUDA_ENFORCE(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, pydevice->index));
	CUDA_ENFORCE(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, pydevice->index));

	return Py_BuildValue("(ii)", major, minor);
}


PyDoc_STRVAR(Cuda_Device_globalMemory_doc, "globalMemory(self) -> int");
static PyObject *Cuda_Device_globalMemory(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_Device *pydevice = (Cuda_Device *)self;

	CUdevice device;
	CU_ENFORCE(cuDeviceGet(&device, pydevice->index));

	size_t size;
	CU_ENFORCE(cuDeviceTotalMem(&size, device));

	Py_ssize_t pysize = size;
	return Py_BuildValue("n", pysize);
}


PyDoc_STRVAR(Cuda_Device_synchronize_doc, "synchronize()");
static PyObject *Cuda_Device_synchronize(PyObject *type, PyObject *args)
{
	(void)type, (void)args;

	CUDA_ENFORCE(cudaDeviceSynchronize());
	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Device_pyGetCurrent_doc, "getCurrent() -> int");
static PyObject *Cuda_Device_pyGetCurrent(PyObject *type, PyObject *args)
{
	(void)type, (void)args;

	int device;
	CUDA_ENFORCE(cudaGetDevice(&device));

	return Py_BuildValue("i", device);
}


PyDoc_STRVAR(Cuda_Device_count_doc, "count() -> int");
static PyObject *Cuda_Device_count(PyObject *type, PyObject *args)
{
	(void)type, (void)args;

	int count;
	CUDA_ENFORCE(cudaGetDeviceCount(&count));

	return Py_BuildValue("i", count);
}


static PyMemberDef Cuda_Device_members[] = {
	{(char *)"index", T_INT, offsetof(Cuda_Device, index), READONLY, NULL},
	{NULL, 0, 0, 0, NULL}
};

static PyMethodDef Cuda_Device_methods[] = {
	{"set", Cuda_Device_set, METH_NOARGS, Cuda_Device_set_doc},

	{"getAttribute", Cuda_Device_getAttribute, METH_VARARGS, Cuda_Device_getAttribute_doc},
	{"name", Cuda_Device_name, METH_NOARGS, Cuda_Device_name_doc},
	{"computeCapability", Cuda_Device_computeCapability, METH_NOARGS, Cuda_Device_computeCapability_doc},
	{"globalMemory", Cuda_Device_globalMemory, METH_NOARGS, Cuda_Device_globalMemory_doc},

	{"synchronize", Cuda_Device_synchronize, METH_STATIC | METH_NOARGS, Cuda_Device_synchronize_doc},
	{"getCurrent", Cuda_Device_pyGetCurrent, METH_STATIC | METH_NOARGS, Cuda_Device_pyGetCurrent_doc},
	{"count", Cuda_Device_count, METH_STATIC | METH_NOARGS, Cuda_Device_count_doc},

	{NULL, NULL, 0, NULL}
};

static PyType_Slot Cuda_Device_slots[] = {
	{Py_tp_new, (void *)Cuda_Device_new},
	{Py_tp_members, Cuda_Device_members},
	{Py_tp_methods, Cuda_Device_methods},
	{0, NULL}
};

static PyType_Spec Cuda_Device_TypeSpec = {
	CUDA_DRIVER_NAME "." CUDA_DEVICE_OBJNAME,
	sizeof(Cuda_Device),
	0,
	Py_TPFLAGS_DEFAULT,
	Cuda_Device_slots
};


PyTypeObject *Cuda_Device_Type = NULL;


bool Cuda_Device_moduleInit(PyObject *m)
{
	if (!createPyClass(m, CUDA_DEVICE_OBJNAME, &Cuda_Device_TypeSpec, &Cuda_Device_Type))
		return false;

	PyModule_AddIntConstant(m, "DEV_ATTR_COMPUTE_CAPABILITY_MAJOR", cudaDevAttrComputeCapabilityMajor);
	PyModule_AddIntConstant(m, "DEV_ATTR_COMPUTE_CAPABILITY_MINOR", cudaDevAttrComputeCapabilityMinor);

	PyModule_AddIntConstant(m, "DEV_ATTR_MAX_BLOCK_DIM_X", cudaDevAttrMaxBlockDimX);
	PyModule_AddIntConstant(m, "DEV_ATTR_MAX_BLOCK_DIM_Y", cudaDevAttrMaxBlockDimY);
	PyModule_AddIntConstant(m, "DEV_ATTR_MAX_BLOCK_DIM_Z", cudaDevAttrMaxBlockDimZ);

	PyModule_AddIntConstant(m, "DEV_ATTR_MAX_GRID_DIM_X", cudaDevAttrMaxGridDimX);
	PyModule_AddIntConstant(m, "DEV_ATTR_MAX_GRID_DIM_Y", cudaDevAttrMaxGridDimY);
	PyModule_AddIntConstant(m, "DEV_ATTR_MAX_GRID_DIM_Z", cudaDevAttrMaxGridDimZ);

	PyModule_AddIntConstant(m, "DEV_ATTR_WARP_SIZE", cudaDevAttrWarpSize);
	PyModule_AddIntConstant(m, "DEV_ATTR_MAX_THREADS_PER_BLOCK", cudaDevAttrMaxThreadsPerBlock);
	PyModule_AddIntConstant(m, "DEV_ATTR_MAX_REGISTERS_PER_BLOCK", cudaDevAttrMaxRegistersPerBlock);
	PyModule_AddIntConstant(m, "DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK", cudaDevAttrMaxSharedMemoryPerBlock);
	PyModule_AddIntConstant(m, "DEV_ATTR_TOTAL_CONSTANT_MEMORY", cudaDevAttrTotalConstantMemory);

	PyModule_AddIntConstant(
		m, "DEV_ATTR_MULTIPROCESSOR_COUNT", cudaDevAttrMultiProcessorCount
	);
	PyModule_AddIntConstant(
		m, "DEV_ATTR_MAX_THREADS_PER_MULTIPROCESSOR", cudaDevAttrMaxThreadsPerMultiProcessor
	);
	PyModule_AddIntConstant(
		m, "DEV_ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR", cudaDevAttrMaxSharedMemoryPerMultiprocessor
	);

	PyModule_AddIntConstant(m, "DEV_ATTR_COOPERATIVE_LAUNCH", cudaDevAttrCooperativeLaunch);
	PyModule_AddIntConstant(m, "DEV_ATTR_MAX_PITCH", cudaDevAttrMaxPitch);

	return true;
}


void Cuda_Device_moduleDealloc(void)
{
	REMOVE_PY_OBJECT(&Cuda_Device_Type);
}
