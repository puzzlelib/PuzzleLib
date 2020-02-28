#include "Driver.h"


static PyObject *Cuda_Stream_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)args, (void)kwds;

	Cuda_Stream *self = (Cuda_Stream *)type->tp_alloc(type, 0);
	if (self == NULL)
		goto error_1;

	CUDA_CHECK(cudaStreamCreate(&self->stream), goto error_2);

#if defined(TRACE_CUDA_DRIVER)
	fprintf(stderr, "[" CUDA_STREAM_OBJNAME "] (0x%" PRIXMAX ") Allocated stream\n", (size_t)self);
#endif

	return (PyObject *)self;

error_2:
	self->stream = NULL;
	Py_DECREF(self);

error_1:
	return NULL;
}


static void Cuda_Stream_dealloc(PyObject *self)
{
	Cuda_Stream *pystream = (Cuda_Stream *)self;

	if (pystream->stream != NULL)
	{
		CUDA_ASSERT(cudaStreamDestroy(pystream->stream));

#if defined(TRACE_CUDA_DRIVER)
		fprintf(stderr, "[" CUDA_STREAM_OBJNAME "] (0x%" PRIXMAX ") Deallocated stream\n", (size_t)pystream);
#endif
	}

	Py_TYPE(self)->tp_free(self);
}


PyDoc_STRVAR(Cuda_Stream_synchronize_doc, "synchronize(self)");
static PyObject *Cuda_Stream_synchronize(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_Stream *pystream = (Cuda_Stream *)self;

	CUDA_ENFORCE(cudaStreamSynchronize(pystream->stream));
	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Stream_isDone_doc, "isDone(self) -> bool");
static PyObject *Cuda_Stream_isDone(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_Stream *pystream = (Cuda_Stream *)self;

	cudaError_t status = cudaStreamQuery(pystream->stream);
	if (status == cudaErrorNotReady)
		Py_RETURN_FALSE;

	CUDA_ENFORCE(status);
	Py_RETURN_TRUE;
}


PyDoc_STRVAR(Cuda_Stream_waitForEvent_doc, "waitForEvent(self, event)");
static PyObject *Cuda_Stream_waitForEvent(PyObject *self, PyObject *args)
{
	Cuda_Stream *pystream = (Cuda_Stream *)self;
	Cuda_Event *pyevent;

	if (!PyArg_ParseTuple(args, "O!", Cuda_Event_Type, &pyevent))
		return NULL;

	CUDA_ENFORCE(cudaStreamWaitEvent(pystream->stream, pyevent->event, 0));
	Py_RETURN_NONE;
}


static PyObject *Cuda_Stream_getHandle(PyObject *self, void *closure)
{
	(void)closure;

	Cuda_Stream *pystream = (Cuda_Stream *)self;
	return PyLong_FromSize_t((size_t)pystream->stream);
}


static PyGetSetDef Cuda_Stream_getset[] = {
	{(char *)"handle", Cuda_Stream_getHandle, NULL, NULL, NULL},
	{NULL, NULL, NULL, NULL, NULL}
};

static PyMethodDef Cuda_Stream_methods[] = {
	{"synchronize", Cuda_Stream_synchronize, METH_NOARGS, Cuda_Stream_synchronize_doc},
	{"isDone", Cuda_Stream_isDone, METH_NOARGS, Cuda_Stream_isDone_doc},
	{"waitForEvent", Cuda_Stream_waitForEvent, METH_VARARGS, Cuda_Stream_waitForEvent_doc},
	{NULL, NULL, 0, NULL}
};

static PyType_Slot Cuda_Stream_slots[] = {
	{Py_tp_new, (void *)Cuda_Stream_new},
	{Py_tp_dealloc, (void *)Cuda_Stream_dealloc},
	{Py_tp_getset, Cuda_Stream_getset},
	{Py_tp_methods, Cuda_Stream_methods},
	{0, NULL}
};

static PyType_Spec Cuda_Stream_TypeSpec = {
	CUDA_DRIVER_NAME "." CUDA_STREAM_OBJNAME,
	sizeof(Cuda_Stream),
	0,
	Py_TPFLAGS_DEFAULT,
	Cuda_Stream_slots
};


PyTypeObject *Cuda_Stream_Type = NULL;


static PyObject *Cuda_Event_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)args, (void)kwds;

	Cuda_Event *self = (Cuda_Event *)type->tp_alloc(type, 0);
	if (self == NULL)
		goto error_1;

	CUDA_CHECK(cudaEventCreateWithFlags(&self->event, cudaEventDefault), goto error_2);

#if defined(TRACE_CUDA_DRIVER)
	fprintf(stderr, "[" CUDA_EVENT_OBJNAME "] (0x%" PRIXMAX ") Allocated event\n", (size_t)self);
#endif

	return (PyObject *)self;

error_2:
	self->event = NULL;
	Py_DECREF(self);

error_1:
	return NULL;
}


static void Cuda_Event_free(Cuda_Event *self)
{
	if (self->event != NULL)
	{
		CUDA_ASSERT(cudaEventDestroy(self->event));

#if defined(TRACE_CUDA_DRIVER)
		fprintf(stderr, "[" CUDA_EVENT_OBJNAME "] (0x%" PRIXMAX ") Deallocated event\n", (size_t)self);
#endif

		self->event = NULL;
	}
}


static void Cuda_Event_dealloc(PyObject *self)
{
	Cuda_Event *pyevent = (Cuda_Event *)self;
	Cuda_Event_free(pyevent);

	Py_TYPE(self)->tp_free(self);
}


PyDoc_STRVAR(Cuda_Event_record_doc, "record(self, stream=None)");
static PyObject *Cuda_Event_record(PyObject *self, PyObject *args)
{
	Cuda_Stream *pystream = NULL;

	if (!PyArg_ParseTuple(args, "|O!", Cuda_Stream_Type, &pystream))
		return NULL;

	cudaStream_t stream = (pystream != NULL) ? pystream->stream : NULL;

	Cuda_Event *pyevent = (Cuda_Event *)self;
	CUDA_ENFORCE(cudaEventRecord(pyevent->event, stream));

	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Event_synchronize_doc, "synchronize(self)");
static PyObject *Cuda_Event_synchronize(PyObject *self, PyObject *args)
{
	(void)args;
	Cuda_Event *pyevent = (Cuda_Event *)self;

	CUDA_ENFORCE(cudaEventSynchronize(pyevent->event));
	Py_RETURN_NONE;
}


PyDoc_STRVAR(Cuda_Event_timeSince_doc, "timeSince(self, event) -> float");
static PyObject *Cuda_Event_timeSince(PyObject *self, PyObject *args)
{
	Cuda_Event *pyevent;

	if (!PyArg_ParseTuple(args, "O!", Cuda_Event_Type, &pyevent))
		return NULL;

	float milli;
	CUDA_ENFORCE(cudaEventElapsedTime(&milli, pyevent->event, ((Cuda_Event *)self)->event));

	return Py_BuildValue("f", milli);
}


PyDoc_STRVAR(Cuda_Event_timeTill_doc, "timeTill(self, event) -> float");
static PyObject *Cuda_Event_timeTill(PyObject *self, PyObject *args)
{
	Cuda_Event *pyevent;

	if (!PyArg_ParseTuple(args, "O!", Cuda_Event_Type, &pyevent))
		return NULL;

	float milli;
	CUDA_ENFORCE(cudaEventElapsedTime(&milli, ((Cuda_Event *)self)->event, pyevent->event));

	return Py_BuildValue("f", milli);
}


static PyMethodDef Cuda_Event_methods[] = {
	{"record", Cuda_Event_record, METH_VARARGS, Cuda_Event_record_doc},
	{"synchronize", Cuda_Event_synchronize, METH_NOARGS, Cuda_Event_synchronize_doc},

	{"timeSince", Cuda_Event_timeSince, METH_VARARGS, Cuda_Event_timeSince_doc},
	{"timeTill", Cuda_Event_timeTill, METH_VARARGS, Cuda_Event_timeTill_doc},

	{NULL, NULL, 0, NULL}
};

static PyType_Slot Cuda_Event_slots[] = {
	{Py_tp_new, (void *)Cuda_Event_new},
	{Py_tp_dealloc, (void *)Cuda_Event_dealloc},
	{Py_tp_methods, Cuda_Event_methods},
	{0, NULL}
}; 

static PyType_Spec Cuda_Event_TypeSpec = {
	CUDA_DRIVER_NAME "." CUDA_EVENT_OBJNAME,
	sizeof(Cuda_Event),
	0,
	Py_TPFLAGS_DEFAULT,
	Cuda_Event_slots
};


PyTypeObject *Cuda_Event_Type = NULL;


bool Cuda_Stream_moduleInit(PyObject *m)
{
	if (!createPyClass(m, CUDA_STREAM_OBJNAME, &Cuda_Stream_TypeSpec, &Cuda_Stream_Type)) goto error_1;
	if (!createPyClass(m, CUDA_EVENT_OBJNAME, &Cuda_Event_TypeSpec, &Cuda_Event_Type))    goto error_2;

	return true;

error_2:
	REMOVE_PY_OBJECT(&Cuda_Stream_Type);
error_1:
	return false;
}


void Cuda_Stream_moduleDealloc(void)
{
	REMOVE_PY_OBJECT(&Cuda_Event_Type);
	REMOVE_PY_OBJECT(&Cuda_Stream_Type);
}
