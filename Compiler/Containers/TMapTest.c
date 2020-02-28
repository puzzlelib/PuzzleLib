#define Py_LIMITED_API
#include <Python.h>

#include "../TestData/IntMap.gen.h"


typedef struct PyIntMap
{
	PyObject_HEAD
	IntMap map;
}
PyIntMap;


static PyObject *PyIntMap_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)args, (void)kwds;

	PyIntMap *self = PyObject_New(PyIntMap, type);
	if (self == NULL)
		return NULL;

	IntMap_init(&self->map);
	return (PyObject *)self;
}


static void PyIntMap_dealloc(PyObject *self)
{
	PyIntMap *pymap = (PyIntMap *)self;
	IntMap_dealloc(&pymap->map);

	PyObject_Free(self);
}


static Py_ssize_t PyIntMap_size(PyObject *self)
{
	PyIntMap *pymap = (PyIntMap *)self;
	return pymap->map.size;
}


static int PyIntMap_setItem(PyObject *self, PyObject *pykey, PyObject *pyvalue)
{
	PyIntMap *pymap = (PyIntMap *)self;

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
		if (!IntMap_delete(&pymap->map, key))
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

	IntMap_insert(&pymap->map, key, value);
	return 0;
}


static PyObject *PyIntMap_getItem(PyObject *self, PyObject *pykey)
{
	if (!PyLong_CheckExact(pykey))
	{
		PyErr_SetString(PyExc_ValueError, "invalid key type");
		return NULL;
	}

	int key = PyLong_AsLong(pykey);
	if (key == -1 && PyErr_Occurred())
		return NULL;

	PyIntMap *pymap = (PyIntMap *)self;
	int value;

	if (!IntMap_get(&pymap->map, key, &value))
	{
		PyErr_SetObject(PyExc_KeyError, pykey);
		return NULL;
	}

	return Py_BuildValue("i", value);
}


static PyType_Spec PyIntMap_TypeSpec = {
	.name = "PyIntMap.IntMap",
	.basicsize = sizeof(PyIntMap),
	.itemsize = 0,
	.flags = Py_TPFLAGS_DEFAULT,
	.slots = (PyType_Slot[]){
		{Py_tp_new, (void *)PyIntMap_new},
		{Py_tp_dealloc, (void *)PyIntMap_dealloc},
		{Py_mp_length, (void *)PyIntMap_size},
		{Py_mp_ass_subscript, (void *)PyIntMap_setItem},
		{Py_mp_subscript, (void *)PyIntMap_getItem},
		{0, NULL}
	}
};


static PyModuleDef IntMap_moduleDef = {
	PyModuleDef_HEAD_INIT,
	.m_name = "IntMap"
};


PyMODINIT_FUNC PyInit_IntMap(void)
{
	PyObject *module = PyModule_Create(&IntMap_moduleDef);
	if (module == NULL)
		return NULL;

	PyModule_AddObject(module, "IntMap", PyType_FromSpec(&PyIntMap_TypeSpec));
	return module;
}
