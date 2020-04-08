#define Py_LIMITED_API
#include <Python.h>

#include "IntVector.gen.h"


typedef struct PyIntVector
{
	PyObject_HEAD
	IntVector vector;
}
PyIntVector;


static PyObject *PyIntVector_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)args, (void)kwds;

	PyIntVector *self = PyObject_New(PyIntVector, type);
	if (self == NULL)
		return NULL;

	IntVector_init(&self->vector);
	return (PyObject *)self;
}


static void PyIntVector_dealloc(PyObject *self)
{
	PyIntVector *pyvec = (PyIntVector *)self;
	IntVector_dealloc(&pyvec->vector);

	PyObject_Del(self);
}


static PyObject *PyIntVector_append(PyObject *self, PyObject *args)
{
	(void)self, (void)args;
	int value;

	if (!PyArg_ParseTuple(args, "i", &value))
		return NULL;

	PyIntVector *pyvec = (PyIntVector *)self;
	IntVector_append(&pyvec->vector, value);

	Py_RETURN_NONE;
}


static PyObject *PyIntVector_pop(PyObject *self, PyObject *args)
{
	(void)args;
	PyIntVector *pyvec = (PyIntVector *)self;

	int value;
	if (!IntVector_pop(&pyvec->vector, &value))
		Py_RETURN_NONE;

	return Py_BuildValue("i", value);
}


static Py_ssize_t PyIntVector_size(PyObject *self)
{
	PyIntVector *pyvec = (PyIntVector *)self;
	return pyvec->vector.size;
}


static PyObject *PyIntVector_getItem(PyObject *self, Py_ssize_t index)
{
	PyIntVector *pyvec = (PyIntVector *)self;
	int value;

	if (!IntVector_get(&pyvec->vector, index, &value))
	{
		PyErr_SetString(PyExc_ValueError, "index out of bounds");
		return NULL;
	}

	return Py_BuildValue("i", value);
}


static PyType_Spec PyIntVector_TypeSpec = {
	.name = "PyIntVector.IntVector",
	.basicsize = sizeof(PyIntVector),
	.flags = Py_TPFLAGS_DEFAULT,
	.slots = (PyType_Slot[]){
		{Py_tp_new, (void *)PyIntVector_new},
		{Py_tp_dealloc, (void *)PyIntVector_dealloc},
		{Py_sq_length, (void *)PyIntVector_size},
		{Py_sq_item, (void *)PyIntVector_getItem},
		{Py_tp_methods, (PyMethodDef[]){
			{"append", PyIntVector_append, METH_VARARGS, NULL},
			{"pop", PyIntVector_pop, METH_NOARGS, NULL},
			{NULL, NULL, 0, NULL}
		}},
		{0, NULL}
	}
};


static PyModuleDef IntVector_moduleDef = {
	PyModuleDef_HEAD_INIT,
	.m_name = "IntVector"
};


PyMODINIT_FUNC PyInit_IntVector(void)
{
	PyObject *module = PyModule_Create(&IntVector_moduleDef);
	PyModule_AddObject(module, "IntVector", PyType_FromSpec(&PyIntVector_TypeSpec));

	return module;
}
