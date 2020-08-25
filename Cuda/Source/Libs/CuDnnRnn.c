#include "Libs.h"
#include "../TraceMalloc/TraceMalloc.gen.h"


#define CUDNN_RNN_OBJNAME "Rnn"


typedef struct CuDnn_Dropout
{
	cudnnDropoutDescriptor_t desc;
	Cuda_Buffer *states;

	unsigned long long seed;
	float rate;
}
CuDnn_Dropout;


inline static bool CuDnn_Dropout_init(CuDnn_Dropout *self, cudnnHandle_t handle, float dropout, unsigned long long seed)
{
	size_t size;

	CUDNN_ASSERT(cudnnDropoutGetStatesSize(handle, &size));
	assert(size > 0);

	self->states = Cuda_Driver_allocate(size);
	if (self->states == NULL)
		goto error_1;

	CUDNN_CHECK(cudnnCreateDropoutDescriptor(&self->desc), goto error_2);
	CUDNN_CHECK(cudnnSetDropoutDescriptor(self->desc, handle, dropout, self->states->ptr, size, seed), goto error_3);

	self->seed = seed;
	self->rate = dropout;

	return true;

error_3:
	cudnnDestroyDropoutDescriptor(self->desc);
error_2:
	Py_DECREF(self->states);
error_1:
	self->states = NULL;
	return false;
}


inline static void CuDnn_Dropout_dealloc(CuDnn_Dropout *self)
{
	Py_DECREF(self->states);
	CUDNN_ASSERT(cudnnDestroyDropoutDescriptor(self->desc));
}


typedef struct CuDnn_Rnn
{
	PyObject_HEAD

	CuDnn_Context *context;

	CuDnn_Dropout dropout;
	cudnnRNNDescriptor_t desc;
	cudnnPersistentRNNPlan_t plan;

	size_t wsize;
	cudnnFilterDescriptor_t wDesc;

	size_t insize, hsize, layers, batchsize;
	Cuda_DataType dtype;

	cudnnRNNAlgo_t algo;
	cudnnRNNMode_t mode;
	cudnnDirectionMode_t direction;
}
CuDnn_Rnn;


inline static bool CuDnn_describeRnnInput(cudnnTensorDescriptor_t *desc, size_t batchsize, size_t insize,
										  Cuda_DataType dtype)
{
	size_t shape[3], strides[3];
	shape[0] = batchsize, shape[1] = insize, shape[2] = 1;

	size_t itemsize = Cuda_dtypeSize(dtype);
	strides[0] = insize * itemsize, strides[1] = itemsize, strides[2] = itemsize;

	if (!CuDnn_describeTensorFromShape(desc, shape, strides, 3, dtype))
		return false;

	return true;
}


inline static bool CuDnn_Rnn_validateWeights(CuDnn_Rnn *self, const Cuda_GPUArray *W)
{
	if (W->ndim != 1 || !W->contiguous || CUDA_GPUARRAY_SHAPE(W)[0] != self->wsize || W->dtype != self->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid weights gpuarray");
		return false;
	}

	return true;
}


inline static bool CuDnn_Rnn_describeWeights(CuDnn_Rnn *self, CuDnn_Context *context, Cuda_DataType dtype)
{
	bool status = false;
	size_t itemsize = Cuda_dtypeSize(dtype);

	cudnnTensorDescriptor_t dataDesc;
	if (!CuDnn_describeRnnInput(&dataDesc, 1, self->insize, dtype))
		goto error_1;

	size_t nbytes;

	CUDNN_CHECK(
		cudnnGetRNNParamsSize(context->handle, self->desc, dataDesc, &nbytes, CuDnn_dtypeToDnn(dtype)
	), goto error_2);
	assert(nbytes % itemsize == 0);

	self->wsize = nbytes / itemsize;

	size_t shape[3];
	shape[0] = self->wsize, shape[1] = 1, shape[2] = 1;

	if (!CuDnn_describeFilterFromShape(&self->wDesc, shape, 3, dtype))
		goto error_2;

	status = true;

error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
error_1:
	return status;
}


inline static bool CuDnn_Rnn_init(CuDnn_Rnn *self, CuDnn_Context *context, size_t insize, size_t hsize,
								  Cuda_DataType dtype, size_t layers, cudnnRNNAlgo_t algo, cudnnRNNMode_t mode,
								  cudnnDirectionMode_t direction, size_t batchsize,
								  float dropout, unsigned long long seed)
{
	if (!CuDnn_isValidDtype(dtype))
	{
		PyErr_SetString(PyExc_ValueError, "invalid datatype");
		goto error_1;
	}

	cudnnDataType_t dnntype; dnntype = CuDnn_dtypeToDnn(dtype);
	algo = (batchsize > 0) ? CUDNN_RNN_ALGO_PERSIST_DYNAMIC : algo;

	if (!CuDnn_Dropout_init(&self->dropout, context->handle, dropout, seed))
		goto error_1;

	CUDNN_CHECK(cudnnCreateRNNDescriptor(&self->desc), goto error_2);

	CUDNN_CHECK(cudnnSetRNNDescriptor(
		context->handle, self->desc, (int)hsize, (int)layers, self->dropout.desc, CUDNN_LINEAR_INPUT,
		direction, mode, algo, dnntype
	), goto error_3);

	CUDNN_CHECK(cudnnSetRNNBiasMode(self->desc, CUDNN_RNN_DOUBLE_BIAS), goto error_3);
	CUDNN_CHECK(cudnnSetRNNMatrixMathType(self->desc, context->mathType), goto error_3);

	if (batchsize > 0)
	{
		CUDNN_CHECK(cudnnCreatePersistentRNNPlan(self->desc, (int)batchsize, dnntype, &self->plan), goto error_3);
		CUDNN_CHECK(cudnnSetPersistentRNNPlan(self->desc, self->plan), goto error_4);
	}

	self->insize = insize, self->hsize = hsize, self->layers = layers, self->batchsize = batchsize;
	self->dtype = dtype;

	self->algo = algo, self->mode = mode, self->direction = direction;

	if (!CuDnn_Rnn_describeWeights(self, context, dtype))
		goto error_4;

	Py_INCREF(context);
	self->context = context;

	return true;

error_4:
	CUDNN_ASSERT(cudnnDestroyPersistentRNNPlan(self->plan));
error_3:
	CUDNN_ASSERT(cudnnDestroyRNNDescriptor(self->desc));
error_2:
	CuDnn_Dropout_dealloc(&self->dropout);

error_1:
	self->context = NULL;
	return false;
}


static PyObject *CuDnn_Rnn_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	(void)type;
	const char *kwlist[] = {
		"context", "insize", "hsize", "dtype", "layers", "algo", "mode", "direction",
		"dropout", "seed", "batchsize", NULL
	};

	CuDnn_Context *context;
	Py_ssize_t insize, hsize, layers = 1, batchsize = 0;

	float dropout = 0.0f;
	unsigned long long seed = 0;

	int algo = CUDNN_RNN_ALGO_STANDARD, mode = CUDNN_LSTM, direction = CUDNN_UNIDIRECTIONAL;
	PyArray_Descr *pytype;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!nnO&|niiifKn", (char **)kwlist, CuDnn_Context_Type, &context, &insize, &hsize,
		PyArray_DescrConverter2, &pytype, &layers, &algo, &mode, &direction, &dropout, &seed, &batchsize
	))
		return NULL;

	CuDnn_Rnn *self = NULL;

	Cuda_DataType dtype = Cuda_numpyToDataType(pytype->type_num);
	if (dtype == DTYPE_INVALID)
		goto error_1;

	self = (CuDnn_Rnn *)type->tp_alloc(type, 0);
	if (self == NULL)
		goto error_1;

	if (!CuDnn_Rnn_init(
		self, context, insize, hsize, dtype, layers,
		(cudnnRNNAlgo_t)algo, (cudnnRNNMode_t)mode, (cudnnDirectionMode_t)direction, batchsize, dropout, seed
	))
	{
		Py_DECREF(self);
		self = NULL;
	}

error_1:
	Py_DECREF(pytype);
	return (PyObject *)self;
}


static void CuDnn_Rnn_dealloc(PyObject *self)
{
	CuDnn_Rnn *rnn = (CuDnn_Rnn *)self;

	if (rnn->context != NULL)
	{
		CUDNN_ASSERT(cudnnDestroyFilterDescriptor(rnn->wDesc));

		CUDNN_ASSERT(cudnnDestroyPersistentRNNPlan(rnn->plan));
		CUDNN_ASSERT(cudnnDestroyRNNDescriptor(rnn->desc));

		CuDnn_Dropout_dealloc(&rnn->dropout);
		Py_DECREF(rnn->context);
	}

	Py_TYPE(self)->tp_free(self);
}


PyDoc_STRVAR(CuDnn_Rnn_getParam_doc, "getParam(self, W, layer, linLayer) -> Tuple[Tuple[int, int], Tuple[int, int]]");
static PyObject *CuDnn_Rnn_getParam(PyObject *self, PyObject *args)
{
	CuDnn_Rnn *rnn = (CuDnn_Rnn *)self;

	Cuda_GPUArray *W;
	int layer, linLayer;

	if (!PyArg_ParseTuple(args, "O!ii", Cuda_GPUArray_Type, &W, &layer, &linLayer))
		return NULL;

	if (!CuDnn_Rnn_validateWeights(rnn, W))
		return NULL;

	cudnnTensorDescriptor_t dataDesc;
	if (!CuDnn_describeRnnInput(&dataDesc, 1, rnn->insize, rnn->dtype))
		return NULL;

	ptrdiff_t Woffset = 0, biasOffset = 0;
	size_t wsize = 0, biasSize = 0;

	bool status = false;

	cudnnFilterDescriptor_t desc;
	CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc), goto error);

	cudnnTensorFormat_t format;
	cudnnDataType_t dtype;
	int nbDims, filterDimA[3];

	void *Wptr;
	CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
		rnn->context->handle, rnn->desc, layer, dataDesc, rnn->wDesc, W->gpudata->ptr, linLayer, desc, &Wptr
	), goto error);

	CUDNN_CHECK(cudnnGetFilterNdDescriptor(desc, 3, &dtype, &format, &nbDims, filterDimA), goto error);

	Woffset = (char *)Wptr - (char *)W->gpudata->ptr;
	wsize = filterDimA[0] * filterDimA[1] * filterDimA[2];

	void *biasptr;
	CUDNN_CHECK(cudnnGetRNNLinLayerBiasParams(
		rnn->context->handle, rnn->desc, layer, dataDesc, rnn->wDesc, W->gpudata->ptr, linLayer, desc, &biasptr
	), goto error);

	if (biasptr != NULL)
	{
		CUDNN_CHECK(cudnnGetFilterNdDescriptor(desc, 3, &dtype, &format, &nbDims, filterDimA), goto error);

		biasOffset = (char *)biasptr - (char *)W->gpudata->ptr;
		biasSize = filterDimA[0] * filterDimA[1] * filterDimA[2];
	}

	status = true;

error:
	CUDNN_ASSERT(cudnnDestroyFilterDescriptor(desc));

	return status ? Py_BuildValue(
		"(nn)(nn)", (Py_ssize_t)Woffset, (Py_ssize_t)wsize, (Py_ssize_t)biasOffset, (Py_ssize_t)biasSize
	) : NULL;
}


typedef struct CuDnn_RnnCells
{
	cudnnTensorDescriptor_t desc;
	const void *hptr, *cptr;
}
CuDnn_RnnCells;


static bool CuDnn_RnnCells_init(CuDnn_RnnCells *self, CuDnn_Rnn *rnn, size_t batchsize,
								const Cuda_GPUArray *hidden, const Cuda_GPUArray *cells)
{
	size_t itemsize = Cuda_dtypeSize(rnn->dtype), shape[3], strides[3];

	shape[0] = (rnn->direction == CUDNN_UNIDIRECTIONAL) ? rnn->layers : 2 * rnn->layers;
	shape[1] = batchsize, shape[2] = rnn->hsize;

	strides[0] = batchsize * rnn->hsize * itemsize, strides[1] = rnn->hsize * itemsize, strides[2] = itemsize;

	const size_t *hshape = NULL;
	if (hidden != NULL)
	{
		if (hidden->ndim != 3 || hidden->dtype != rnn->dtype || !hidden->contiguous)
		{
			PyErr_SetString(PyExc_ValueError, "invalid input gpuarray hidden layer layout");
			return false;
		}

		hshape = CUDA_GPUARRAY_SHAPE(hidden);
	}

	const size_t *cshape = NULL;
	if (cells != NULL)
	{
		if (cells->ndim != 3 || cells->dtype != rnn->dtype || !cells->contiguous)
		{
			PyErr_SetString(PyExc_ValueError, "invalid input gpuarray cells layer layout");
			return false;
		}

		cshape = CUDA_GPUARRAY_SHAPE(cells);
	}

	for (size_t i = 0; i < 3; i += 1)
	{
		if ((hshape != NULL && hshape[i] != shape[i]) || (cshape != NULL && cshape[i] != shape[i]))
		{
			PyErr_SetString(PyExc_ValueError, "invalid input gpuarray cells dims");
			return false;
		}
	}

	if (!CuDnn_describeTensorFromShape(&self->desc, shape, strides, 3, rnn->dtype))
		return false;

	self->hptr = (hidden == NULL) ? NULL : hidden->gpudata->ptr;
	self->cptr = (cells == NULL) ? NULL : cells->gpudata->ptr;

	return true;
}


static void CuDnn_RnnCells_dealloc(CuDnn_RnnCells *self)
{
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(self->desc));
}


typedef struct CuDnn_RnnBwdCells
{
	CuDnn_RnnCells base;
	Cuda_GPUArray *dhidden, *dcells;
}
CuDnn_RnnBwdCells;


static bool CuDnn_RnnBwdCells_init(CuDnn_RnnBwdCells *self, CuDnn_Rnn *rnn, size_t batchsize,
								   const Cuda_GPUArray *hidden, const Cuda_GPUArray *cells,
								   Cuda_GPUArray *dhidden, Cuda_GPUArray *dcells, Cuda_MemoryPool *allocator)
{
	if (!CuDnn_RnnCells_init(&self->base, rnn, batchsize, hidden, cells))
		goto error_1;

	if (hidden != NULL)
	{
		dhidden = CuDnn_enforceAllocated(
			dhidden, allocator, CUDA_GPUARRAY_SHAPE(hidden), hidden->ndim, hidden->dtype, false
		);
		if (dhidden == NULL) goto error_2;
	}

	if (cells != NULL)
	{
		dcells = CuDnn_enforceAllocated(
			dcells, allocator, CUDA_GPUARRAY_SHAPE(cells), cells->ndim, cells->dtype, false
		);
		if (dcells == NULL) goto error_3;
	}

	self->dhidden = dhidden, self->dcells = dcells;
	return true;

error_3:
	Py_DECREF(dhidden);
error_2:
	CuDnn_RnnCells_dealloc(&self->base);
error_1:
	return false;
}


static void CuDnn_RnnBwdCells_dealloc(CuDnn_RnnBwdCells *self, bool status)
{
	if (!status && self->dhidden != NULL)
	{
		Py_DECREF(self->dhidden);
		self->dhidden = NULL;
	}

	if (!status && self->dcells != NULL)
	{
		Py_DECREF(self->dcells);
		self->dcells = NULL;
	}

	CuDnn_RnnCells_dealloc(&self->base);
}


inline static bool CuDnn_Rnn_describeRnnInOut(CuDnn_Rnn *self, size_t batchsize, Cuda_DataType dtype,
											  cudnnTensorDescriptor_t *pInDesc, cudnnTensorDescriptor_t *pOutDesc)
{
	size_t hsize = (self->direction == CUDNN_UNIDIRECTIONAL) ? self->hsize : 2 * self->hsize;

	size_t itemsize = Cuda_dtypeSize(self->dtype), shape[3], strides[3];
	shape[0] = batchsize, shape[2] = 1;
	strides[1] = itemsize, strides[2] = itemsize;

	cudnnTensorDescriptor_t inDesc, outDesc;

	shape[1] = self->insize, strides[0] = self->insize * itemsize;
	if (!CuDnn_describeTensorFromShape(&inDesc, shape, strides, 3, dtype))
		goto error_1;

	shape[1] = hsize, strides[0] = hsize * itemsize;
	if (!CuDnn_describeTensorFromShape(&outDesc, shape, strides, 3, dtype))
		goto error_2;

	*pInDesc = inDesc, *pOutDesc = outDesc;
	return true;

error_2:
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(inDesc));
error_1:
	return false;
}


static bool CuDnn_Rnn_forward(CuDnn_Rnn *self, const Cuda_GPUArray *data, const Cuda_GPUArray *W, Cuda_GPUArray *out,
							  CuDnn_RnnCells cells, Cuda_Buffer **pReserve, bool test, Cuda_MemoryPool *allocator)
{
	bool status = false;

	const size_t *inshape = CUDA_GPUARRAY_SHAPE(data);
	size_t seqlen = inshape[0], batchsize = inshape[1];

	cudnnTensorDescriptor_t dataDesc, outDesc;
	if (!CuDnn_Rnn_describeRnnInOut(self, batchsize, data->dtype, &dataDesc, &outDesc))
		return false;

	cudnnTensorDescriptor_t *inDescs = (cudnnTensorDescriptor_t *)TRACE_MALLOC(sizeof(*inDescs) * seqlen);
	cudnnTensorDescriptor_t *outDescs = (cudnnTensorDescriptor_t *)TRACE_MALLOC(sizeof(*outDescs) * seqlen);

	for (size_t i = 0; i < seqlen; i += 1)
		inDescs[i] = dataDesc, outDescs[i] = outDesc;

	size_t size;
	CUDNN_CHECK(cudnnGetRNNWorkspaceSize(self->context->handle, self->desc, (int)seqlen, inDescs, &size), goto error_1);

	Cuda_Buffer *workspace; workspace = NULL;
	if (size > 0)
	{
		workspace = Cuda_Buffer_newWithAllocator(size, data->gpudata->device, allocator);
		if (workspace == NULL) goto error_1;
	}

	if (test)
	{
		CUDNN_CHECK(cudnnRNNForwardInference(
			self->context->handle, self->desc, (int)seqlen, inDescs, data->gpudata->ptr,
			cells.desc, cells.hptr, cells.desc, cells.cptr, self->wDesc, W->gpudata->ptr,
			outDescs, out->gpudata->ptr, cells.desc, NULL, cells.desc, NULL,
			workspace == NULL ? NULL : workspace->ptr, size
		), goto error_2);
	}
	else
	{
		size_t reserveSize;
		CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
			self->context->handle, self->desc, (int)seqlen, inDescs, &reserveSize
		), goto error_2);

		Cuda_Buffer *reserve = Cuda_Buffer_newWithAllocator(reserveSize, data->gpudata->device, allocator);
		if (reserve == NULL) goto error_2;

		CUDNN_CHECK(cudnnRNNForwardTraining(
			self->context->handle, self->desc, (int)seqlen, inDescs, data->gpudata->ptr,
			cells.desc, cells.hptr, cells.desc, cells.cptr, self->wDesc, W->gpudata->ptr,
			outDescs, out->gpudata->ptr, cells.desc, NULL, cells.desc, NULL,
			workspace == NULL ? NULL : workspace->ptr, size, reserve->ptr, reserveSize
		), Py_DECREF(reserve); goto error_2);

		*pReserve = reserve;
	}

	status = true;

error_2:
	if (workspace != NULL)
		Py_DECREF(workspace);

error_1:
	TRACE_FREE(inDescs);
	TRACE_FREE(outDescs);

	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));

	return status;
}


PyDoc_STRVAR(
	CuDnn_Rnn_forward_doc,
	"forward(self, data, w, hidden=None, cells=None, test=False, out=None, allocator=None) -> "
	"Union[" CUDA_GPUARRAY_FULLNAME", Tuple[" CUDA_GPUARRAY_FULLNAME ", " CUDA_GPUARRAY_FULLNAME "]]"
);
static PyObject *CuDnn_Rnn_pyForward(PyObject *self, PyObject *args, PyObject *kwds)
{
	CuDnn_Rnn *rnn = (CuDnn_Rnn *)self;
	const char *kwlist[] = {"data", "W", "hidden", "cells", "test", "out", "allocator", NULL};

	Cuda_GPUArray *data, *W, *hidden = NULL, *cells = NULL;
	PyObject *pyout = NULL, *pyalloc = NULL;
	int test = 0;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!|O!O!pOO", (char **)kwlist, Cuda_GPUArray_Type, &data, Cuda_GPUArray_Type, &W,
		Cuda_GPUArray_Type, &hidden, Cuda_GPUArray_Type, &cells, &test, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	const size_t *shape = CUDA_GPUARRAY_SHAPE(data);

	if (data->ndim != 3 || shape[2] != rnn->insize || data->dtype != rnn->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	if (rnn->batchsize > 0 && shape[1] != rnn->batchsize)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray batch size");
		return NULL;
	}

	if (!CuDnn_Rnn_validateWeights(rnn, W))
		return NULL;

	size_t batchsize = shape[1], hsize = (rnn->direction == CUDNN_UNIDIRECTIONAL) ? rnn->hsize : 2 * rnn->hsize;

	CuDnn_RnnCells rnnCells;
	if (!CuDnn_RnnCells_init(&rnnCells, rnn, batchsize, hidden, cells))
		return NULL;

	Cuda_Buffer *reserve = NULL;

	size_t outshape[3];
	outshape[0] = shape[0], outshape[1] = batchsize, outshape[2] = hsize;

	out = CuDnn_enforceAllocated(out, allocator, outshape, data->ndim, data->dtype, false);
	if (out == NULL) goto error;

	if (!CuDnn_Rnn_forward(rnn, data, W, out, rnnCells, &reserve, test, allocator))
	{
		Py_DECREF(out);
		out = NULL;
	}

error:
	CuDnn_RnnCells_dealloc(&rnnCells);
	return test ? (PyObject *)out : Py_BuildValue("NN", out, reserve);
}


static bool CuDnn_Rnn_backwardData(CuDnn_Rnn *self, const Cuda_GPUArray *grad, const Cuda_GPUArray *outdata,
								   const Cuda_GPUArray *W, Cuda_Buffer *reserve, Cuda_GPUArray *out,
								   CuDnn_RnnBwdCells bwdCells, Cuda_MemoryPool *allocator)
{
	bool status = false;

	const size_t *shape = CUDA_GPUARRAY_SHAPE(grad);
	size_t seqlen = shape[0], batchsize = shape[1];

	cudnnTensorDescriptor_t gradDesc, outDesc;
	if (!CuDnn_Rnn_describeRnnInOut(self, batchsize, grad->dtype, &outDesc, &gradDesc))
		return false;

	cudnnTensorDescriptor_t *gradDescs = (cudnnTensorDescriptor_t *)TRACE_MALLOC(sizeof(*gradDescs) * seqlen);
	cudnnTensorDescriptor_t *outDescs = (cudnnTensorDescriptor_t *)TRACE_MALLOC(sizeof(*outDescs) * seqlen);

	for (size_t i = 0; i < seqlen; i += 1)
		gradDescs[i] = gradDesc, outDescs[i] = outDesc;

	size_t size;
	CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
		self->context->handle, self->desc, (int)seqlen, outDescs, &size
	), goto error_1);

	Cuda_Buffer *workspace; workspace = NULL;
	if (size > 0)
	{
		workspace = Cuda_Buffer_newWithAllocator(size, grad->gpudata->device, allocator);
		if (workspace == NULL) goto error_1;
	}

	CuDnn_RnnCells cells; cells = bwdCells.base;
	void *dhptr; dhptr = (bwdCells.dhidden != NULL) ? bwdCells.dhidden->gpudata->ptr : NULL;
	void *dcptr; dcptr = (bwdCells.dcells != NULL) ? bwdCells.dcells->gpudata->ptr : NULL;

	CUDNN_CHECK(cudnnRNNBackwardData(
		self->context->handle, self->desc, (int)seqlen, gradDescs, outdata->gpudata->ptr,
		gradDescs, grad->gpudata->ptr, cells.desc, NULL, cells.desc, NULL, self->wDesc, W->gpudata->ptr,
		cells.desc, cells.hptr, cells.desc, cells.cptr, outDescs, out->gpudata->ptr, cells.desc, dhptr,
		cells.desc, dcptr, workspace == NULL ? NULL : workspace->ptr, size, reserve->ptr, reserve->size
	), goto error_2);

	status = true;

error_2:
	if (workspace != NULL)
		Py_DECREF(workspace);

error_1:
	TRACE_FREE(gradDescs);
	TRACE_FREE(outDescs);

	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(gradDesc));
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));

	return status;
}


PyDoc_STRVAR(
	CuDnn_Rnn_pyBackwardData_doc,
	"backwardData(self, grad, outdata, w, reserve, hidden=None, cells=None, "
	"out=None, dhidden=None, dcells=None, allocator=None) -> Tuple[" CUDA_GPUARRAY_FULLNAME ", "
	"Union[" CUDA_GPUARRAY_FULLNAME ", None], Union[" CUDA_GPUARRAY_FULLNAME ", None]]"
);
static PyObject *CuDnn_Rnn_pyBackwardData(PyObject *self, PyObject *args, PyObject *kwds)
{
	CuDnn_Rnn *rnn = (CuDnn_Rnn *)self;
	const char *kwlist[] = {
		"grad", "outdata", "W", "reserve", "hidden", "cells", "out", "dhidden", "dcells", "allocator", NULL
	};

	Cuda_GPUArray *grad, *outdata, *W, *hidden = NULL, *cells = NULL;
	Cuda_Buffer *reserve;
	PyObject *pyout = NULL, *pydhidden = NULL, *pydcells = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!O!O!|O!O!OOOO", (char **)kwlist, Cuda_GPUArray_Type, &grad, Cuda_GPUArray_Type, &outdata,
		Cuda_GPUArray_Type, &W, Cuda_Buffer_Type, &reserve, Cuda_GPUArray_Type, &hidden, Cuda_GPUArray_Type, &cells,
		&pyout, &pydhidden, &pydcells, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pydhidden, Cuda_GPUArray_Type, "dhidden"))   return NULL;
	if (!unpackPyOptional(&pydcells, Cuda_GPUArray_Type, "dcells"))     return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_GPUArray *dhidden = (Cuda_GPUArray *)pydhidden, *dcells = (Cuda_GPUArray *)pydcells;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	const size_t *shape = CUDA_GPUARRAY_SHAPE(grad);
	size_t batchsize = shape[1];

	if (grad->ndim != 3 || outdata->ndim != 3 || grad->dtype != rnn->dtype || outdata->dtype != rnn->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	if (rnn->batchsize > 0 && shape[1] != rnn->batchsize)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray batch size");
		return NULL;
	}

	if (!CuDnn_Rnn_validateWeights(rnn, W))
		return NULL;

	CuDnn_RnnBwdCells rnnCells;
	if (!CuDnn_RnnBwdCells_init(&rnnCells, rnn, batchsize, hidden, cells, dhidden, dcells, allocator))
		return NULL;

	bool status = false;

	size_t outshape[3];
	outshape[0] = shape[0], outshape[1] = batchsize, outshape[2] = rnn->insize;

	out = CuDnn_enforceAllocated(out, allocator, outshape, grad->ndim, grad->dtype, false);
	if (out == NULL) goto error;

	if (!CuDnn_Rnn_backwardData(rnn, grad, outdata, W, reserve, out, rnnCells, allocator))
	{
		Py_DECREF(out);
		out = NULL;
	}

	status = true;

error:
	CuDnn_RnnBwdCells_dealloc(&rnnCells, status);

	if (!status)
		return NULL;

	PyObject *outhidden = (PyObject *)rnnCells.dhidden, *outcells = (PyObject *)rnnCells.dcells;

	if (outhidden == NULL)
	{
		Py_INCREF(Py_None);
		outhidden = Py_None;
	}

	if (outcells == NULL)
	{
		Py_INCREF(Py_None);
		outcells = Py_None;
	}
	return Py_BuildValue("NNN", out, outhidden, outcells);
}


static bool CuDnn_Rnn_backwardParams(CuDnn_Rnn *self, const Cuda_GPUArray *data, const Cuda_GPUArray *outdata,
									 Cuda_Buffer *reserve, Cuda_GPUArray *out, CuDnn_RnnCells rnnCells,
									 Cuda_MemoryPool *allocator)
{
	bool status = false;

	const size_t *shape = CUDA_GPUARRAY_SHAPE(data);
	size_t seqlen = shape[0], batchsize = shape[1];

	cudnnTensorDescriptor_t dataDesc, outDesc;
	if (!CuDnn_Rnn_describeRnnInOut(self, batchsize, data->dtype, &dataDesc, &outDesc))
		return false;

	cudnnTensorDescriptor_t *dataDescs = (cudnnTensorDescriptor_t *)TRACE_MALLOC(sizeof(*dataDescs) * seqlen);
	cudnnTensorDescriptor_t *outDescs = (cudnnTensorDescriptor_t *)TRACE_MALLOC(sizeof(*outDescs) * seqlen);

	for (size_t i = 0; i < seqlen; i += 1)
		dataDescs[i] = dataDesc, outDescs[i] = outDesc;

	size_t size;
	CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
		self->context->handle, self->desc, (int)seqlen, dataDescs, &size
	), goto error_1);

	Cuda_Buffer *workspace; workspace = NULL;
	if (size > 0)
	{
		workspace = Cuda_Buffer_newWithAllocator(size, data->gpudata->device, allocator);
		if (workspace == NULL) goto error_1;
	}

	CUDNN_CHECK(cudnnRNNBackwardWeights(
		self->context->handle, self->desc, (int)seqlen, dataDescs, data->gpudata->ptr, rnnCells.desc, rnnCells.hptr,
		outDescs, outdata->gpudata->ptr, workspace == NULL ? NULL : workspace->ptr, size,
		self->wDesc, out->gpudata->ptr, reserve->ptr, reserve->size
	), goto error_2);

	status = true;

error_2:
	if (workspace != NULL)
		Py_DECREF(workspace);

error_1:
	TRACE_FREE(dataDescs);
	TRACE_FREE(outDescs);

	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(dataDesc));
	CUDNN_ASSERT(cudnnDestroyTensorDescriptor(outDesc));

	return status;
}


PyDoc_STRVAR(
	CuDnn_Rnn_pyBackwardParams_doc,
	"backwardParams(self, data, outdata, reserve, hidden=None, out=None, allocator=None) -> CUDA_GPUARRAY_FULLNAME"
);
static PyObject *CuDnn_Rnn_pyBackwardParams(PyObject *self, PyObject *args, PyObject *kwds)
{
	CuDnn_Rnn *rnn = (CuDnn_Rnn *)self;
	const char *kwlist[] = {"data", "outdata", "reserve", "hidden", "out", "allocator", NULL};

	Cuda_GPUArray *data, *outdata, *hidden = NULL;
	Cuda_Buffer *reserve;
	PyObject *pyout = NULL, *pyalloc = NULL;

	if (!PyArg_ParseTupleAndKeywords(
		args, kwds, "O!O!O!|O!OO", (char **)kwlist, Cuda_GPUArray_Type, &data, Cuda_GPUArray_Type, &outdata,
		Cuda_Buffer_Type, &reserve, Cuda_GPUArray_Type, &hidden, &pyout, &pyalloc
	))
		return NULL;

	if (!unpackPyOptional(&pyout, Cuda_GPUArray_Type, "out"))           return NULL;
	if (!unpackPyOptional(&pyalloc, Cuda_MemoryPool_Type, "allocator")) return NULL;

	Cuda_GPUArray *out = (Cuda_GPUArray *)pyout;
	Cuda_MemoryPool *allocator = (Cuda_MemoryPool *)pyalloc;

	const size_t *shape = CUDA_GPUARRAY_SHAPE(data);
	size_t batchsize = shape[1];

	if (data->ndim != 3 || outdata->ndim != 3 || data->dtype != rnn->dtype || outdata->dtype != rnn->dtype)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray data layout");
		return NULL;
	}

	if (rnn->batchsize > 0 && shape[1] != rnn->batchsize)
	{
		PyErr_SetString(PyExc_ValueError, "invalid input gpuarray batch size");
		return NULL;
	}

	CuDnn_RnnCells rnnCells;
	if (!CuDnn_RnnCells_init(&rnnCells, rnn, batchsize, hidden, NULL))
		return NULL;

	out = CuDnn_enforceAllocated(out, allocator, &rnn->wsize, 1, rnn->dtype, true);
	if (out == NULL) goto error;

	if (!CuDnn_Rnn_backwardParams(rnn, data, outdata, reserve, out, rnnCells, allocator))
	{
		Py_DECREF(out);
		out = NULL;
	}

error:
	CuDnn_RnnCells_dealloc(&rnnCells);
	return (PyObject *)out;
}


static PyMemberDef CuDnn_Rnn_members[] = {
	{(char *)"context", T_OBJECT_EX, offsetof(CuDnn_Rnn, context), READONLY, NULL},
	{(char *)"wsize", T_PYSSIZET, offsetof(CuDnn_Rnn, wsize), READONLY, NULL},

	{(char *)"states", T_OBJECT_EX, offsetof(CuDnn_Rnn, dropout.states), READONLY, NULL},
	{(char *)"dropout", T_FLOAT, offsetof(CuDnn_Rnn, dropout.rate), READONLY, NULL},
	{(char *)"seed", T_ULONGLONG, offsetof(CuDnn_Rnn, dropout.seed), READONLY, NULL},

	{(char *)"insize", T_PYSSIZET, offsetof(CuDnn_Rnn, insize), READONLY, NULL},
	{(char *)"hsize", T_PYSSIZET, offsetof(CuDnn_Rnn, hsize), READONLY, NULL},
	{(char *)"layers", T_PYSSIZET, offsetof(CuDnn_Rnn, layers), READONLY, NULL},

	{(char *)"algo", T_INT, offsetof(CuDnn_Rnn, algo), READONLY, NULL},
	{(char *)"mode", T_INT, offsetof(CuDnn_Rnn, mode), READONLY, NULL},
	{(char *)"direction", T_INT, offsetof(CuDnn_Rnn, direction), READONLY, NULL},

	{NULL, 0, 0, 0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic push
	#if __GNUC__ >= 8
		#pragma GCC diagnostic ignored "-Wcast-function-type"
	#endif
#endif

static PyMethodDef CuDnn_Rnn_methods[] = {
	{"getParam", CuDnn_Rnn_getParam, METH_VARARGS, CuDnn_Rnn_getParam_doc},
	{"forward", (PyCFunction)CuDnn_Rnn_pyForward, METH_VARARGS | METH_KEYWORDS, CuDnn_Rnn_forward_doc},
	{
		"backwardData", (PyCFunction)CuDnn_Rnn_pyBackwardData, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Rnn_pyBackwardData_doc
	},
	{
		"backwardParams", (PyCFunction)CuDnn_Rnn_pyBackwardParams, METH_VARARGS | METH_KEYWORDS,
		CuDnn_Rnn_pyBackwardParams_doc
	},
	{NULL, NULL, 0, NULL}
};

#if defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif

static PyType_Slot CuDnn_Rnn_slots[] = {
	{Py_tp_new, (void *)CuDnn_Rnn_new},
	{Py_tp_dealloc, (void *)CuDnn_Rnn_dealloc},
	{Py_tp_members, CuDnn_Rnn_members},
	{Py_tp_methods, CuDnn_Rnn_methods},
	{0, NULL}
};

PyType_Spec CuDnn_Rnn_TypeSpec = {
	CUDNN_BACKEND_NAME "." CUDNN_RNN_OBJNAME,
	sizeof(CuDnn_Rnn),
	0,
	Py_TPFLAGS_DEFAULT,
	CuDnn_Rnn_slots
};


PyTypeObject *CuDnn_Rnn_Type = NULL;


bool CuDnnRnn_moduleInit(PyObject *m)
{
	if (!createPyClass(m, CUDNN_RNN_OBJNAME, &CuDnn_Rnn_TypeSpec, &CuDnn_Rnn_Type))
		return false;

	PyModule_AddIntConstant(m, "RNN_ALGO_STANDARD", CUDNN_RNN_ALGO_STANDARD);
	PyModule_AddIntConstant(m, "RNN_ALGO_PERSIST_STATIC", CUDNN_RNN_ALGO_PERSIST_STATIC);
	PyModule_AddIntConstant(m, "RNN_ALGO_PERSIST_DYNAMIC", CUDNN_RNN_ALGO_PERSIST_DYNAMIC);

	PyModule_AddIntConstant(m, "RNN_MODE_RELU", CUDNN_RNN_RELU);
	PyModule_AddIntConstant(m, "RNN_MODE_TANH", CUDNN_RNN_TANH);
	PyModule_AddIntConstant(m, "RNN_MODE_LSTM", CUDNN_LSTM);
	PyModule_AddIntConstant(m, "RNN_MODE_GRU", CUDNN_GRU);

	PyModule_AddIntConstant(m, "RNN_DIRECTION_UNIDIRECTIONAL", CUDNN_UNIDIRECTIONAL);
	PyModule_AddIntConstant(m, "RNN_DIRECTION_BIDIRECTIONAL", CUDNN_BIDIRECTIONAL);

	return true;
}


void CuDnnRnn_moduleDealloc(void)
{
	REMOVE_PY_OBJECT(&CuDnn_Rnn_Type);
}
