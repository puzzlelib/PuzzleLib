#define Py_LIMITED_API
#include <Python.h>

#include "../TestData/TraceMalloc.gen.h"


static PyObject *PyTraceMalloc_malloc(PyObject *self, PyObject *args)
{
	(void)self, (void)args;
	Py_ssize_t nbytes;

	if (!PyArg_ParseTuple(args, "n", &nbytes))
		return NULL;

	void *ptr = TRACE_MALLOC(nbytes);
	return Py_BuildValue("n", (Py_ssize_t)ptr);
}


static PyObject *PyTraceMalloc_free(PyObject *self, PyObject *args)
{
	(void)self, (void)args;
	Py_ssize_t ptr;

	if (!PyArg_ParseTuple(args, "n", &ptr))
		return NULL;

	TRACE_FREE((void *)ptr);
	Py_RETURN_NONE;
}


static PyObject *PyTraceMalloc_traceLeaks(PyObject *self, PyObject *args)
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

		PyList_SetItem(leaks, index, leak);
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


static PyModuleDef PyTraceMalloc_moduleDef = {
	PyModuleDef_HEAD_INIT,
	.m_name = "TraceMalloc",
	.m_methods = (PyMethodDef[]){
		{"malloc", PyTraceMalloc_malloc, METH_VARARGS, NULL},
		{"free", PyTraceMalloc_free, METH_VARARGS, NULL},
		{"traceLeaks", PyTraceMalloc_traceLeaks, METH_NOARGS, NULL},
		{NULL, NULL, 0, NULL}
	},
	.m_slots = NULL
};


PyMODINIT_FUNC PyInit_TraceMalloc(void)
{
	return PyModule_Create(&PyTraceMalloc_moduleDef);
}
