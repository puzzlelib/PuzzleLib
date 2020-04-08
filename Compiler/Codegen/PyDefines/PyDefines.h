#pragma once
#include <stdbool.h>


#if defined(__clang__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wvisibility"

#elif defined(_MSC_VER)
	#pragma warning(push)
	#pragma warning(disable: 4115)

#endif

#include <Python.h>
#include <structmember.h>

#if defined(__clang__)
	#pragma GCC diagnostic pop

#elif defined(_MSC_VER)
	#pragma warning(pop)

#endif


inline static bool createPyClass(PyObject *module, const char *name, PyType_Spec *spec, PyTypeObject **pType)
{
	PyTypeObject *type = (PyTypeObject *)PyType_FromSpec(spec);
	if (type == NULL)
		return false;

	if (PyModule_AddObject(module, name, (PyObject *)type) < 0)
	{
		Py_DECREF(type);
		return false;
	}

	Py_INCREF(type);
	*pType = type;

	return true;
}

inline static bool createPyExc(PyObject *module, const char *name, const char *fullname, PyObject **pExc)
{
	PyObject *exc = PyErr_NewException(fullname, NULL, NULL);
	if (exc == NULL)
		return false;

	if (PyModule_AddObject(module, name, exc) < 0)
	{
		Py_DECREF(exc);
		return false;
	}

	Py_INCREF(exc);
	*pExc = exc;

	return true;
}

inline static bool unpackPyOptional(PyObject **pObj, PyTypeObject *type, const char *key)
{
	PyObject *obj = *pObj;

	if (obj != NULL && Py_TYPE(obj) != type && obj != Py_None)
	{
		PyErr_Format(
			PyExc_TypeError, "%s must be %s or %s, not %s",
			key, type->tp_name, Py_TYPE(Py_None)->tp_name, Py_TYPE(obj)->tp_name
		);
		return false;
	}

	*pObj = (obj == Py_None) ? NULL : obj;
	return true;
}

#define REMOVE_PY_OBJECT(pObj) do { PyObject *obj = (PyObject *)*(pObj); Py_DECREF(obj); *(pObj) = NULL; } while (0)
