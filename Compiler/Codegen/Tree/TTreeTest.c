#define Py_LIMITED_API
#include <Python.h>

#include "IntTree.gen.h"


typedef struct PyIntTree
{
	PyObject_HEAD
	IntTree tree;
}
PyIntTree;


static PyObject *PyIntTree_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)args, (void)kwds;

	PyIntTree *self = PyObject_New(PyIntTree, type);
	if (self == NULL)
		return NULL;

	IntTree_init(&self->tree);
	return (PyObject *)self;
}


static void PyIntTree_dealloc(PyObject *self)
{
	PyIntTree *pytree = (PyIntTree *)self;
	IntTree_dealloc(&pytree->tree);

	PyObject_Del(self);
}


static Py_ssize_t PyIntTree_size(PyObject *self)
{
	PyIntTree *pytree = (PyIntTree *)self;
	return pytree->tree.size;
}


static PyObject *PyIntTree_validate(PyObject *self, PyObject *args)
{
	(void)args;
	PyIntTree *pytree = (PyIntTree *)self;

	if (IntTree_validate(&pytree->tree))
		Py_RETURN_TRUE;
	else
		Py_RETURN_FALSE;
}


static int PyIntTree_setItem(PyObject *self, PyObject *pykey, PyObject *pyvalue)
{
	PyIntTree *pytree = (PyIntTree *)self;

	if (!PyLong_CheckExact(pykey))
	{
		PyErr_SetString(PyExc_ValueError, "invalid key type");
		return -1;
	}

	int key = PyLong_AsLong(pykey);
	if (key == -1 && PyErr_Occurred())
		return -1;

	if (pyvalue == NULL)
	{
		if (!IntTree_delete(&pytree->tree, key))
		{
			PyErr_SetObject(PyExc_KeyError, pykey);
			return -1;
		}

		return 0;
	}
	else if (!PyLong_CheckExact(pyvalue))
	{
		PyErr_SetString(PyExc_ValueError, "invalid value type");
		return -1;
	}

	int value = PyLong_AsLong(pyvalue);
	if (value == -1 && PyErr_Occurred())
		return -1;

	IntTree_insert(&pytree->tree, key, value);
	return 0;
}


static PyObject *PyIntTree_getItem(PyObject *self, PyObject *pykey)
{
	if (!PyLong_CheckExact(pykey))
	{
		PyErr_SetString(PyExc_ValueError, "invalid key type");
		return NULL;
	}

	int key = PyLong_AsLong(pykey);
	if (key == -1 && PyErr_Occurred())
		return NULL;

	PyIntTree *pytree = (PyIntTree *)self;
	int value;

	if (!IntTree_get(&pytree->tree, key, &value))
	{
		PyErr_SetObject(PyExc_KeyError, pykey);
		return NULL;
	}

	return Py_BuildValue("i", value);
}


static PyType_Spec PyIntTree_TypeSpec = {
	.name = "PyIntTree.IntTree",
	.basicsize = sizeof(PyIntTree),
	.itemsize = 0,
	.flags = Py_TPFLAGS_DEFAULT,
	.slots = (PyType_Slot[]){
		{Py_tp_new, (void *)PyIntTree_new},
		{Py_tp_dealloc, (void *)PyIntTree_dealloc},
		{Py_mp_length, (void *)PyIntTree_size},
		{Py_mp_ass_subscript, (void *)PyIntTree_setItem},
		{Py_mp_subscript, (void *)PyIntTree_getItem},
		{Py_tp_methods, (PyMethodDef[]){
			{"validate", PyIntTree_validate, METH_VARARGS, NULL},
			{NULL, NULL, 0, NULL}
		}},
		{0, NULL}
	}
};


static PyModuleDef IntTree_moduleDef = {
	PyModuleDef_HEAD_INIT,
	.m_name = "IntTree"
};


PyMODINIT_FUNC PyInit_IntTree(void)
{
	PyObject *module = PyModule_Create(&IntTree_moduleDef);
	if (module == NULL)
		return NULL;

	PyModule_AddObject(module, "IntTree", PyType_FromSpec(&PyIntTree_TypeSpec));
	return module;
}
