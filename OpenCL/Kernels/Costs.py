from string import Template

import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Kernels.Templates import ElementwiseKernel, ReductionKernel
from PuzzleLib.OpenCL.Kernels.Utils import nthreads, roundUp, atomicAddTmpl
from PuzzleLib.OpenCL.Utils import context, queue, memoryPool as memPool
from PuzzleLib.OpenCL.Wrappers.MIOpen import softmax2d


bceKer = ElementwiseKernel(
	"const float *scores, const int *labels, float *totalError, float *grad, int numsamples, int spatialDim",
	"""
	float prob = 1.0f / (1.0f + exp(-scores[i]));
	float error = labels[i] == 1 ? -log(prob) : -log(1.0f - prob);
	atomicAddCAS(totalError, error / spatialDim);
	grad[i] = ((labels[i] == 1) - prob) / numsamples / spatialDim;
	""",
	"bceKer", preamble=atomicAddTmpl
)

hingeKer = ElementwiseKernel(
	"const float *scores, const int *labels, float *totalError, float *grad, int numsamples, int numcases",
	"""
	float score = scores[i];
	int label = labels[i];
	float error = max(0.0f, 1.0f - score * label) / numcases;
	atomicAddCAS(totalError, error);
	grad[i] = score * label < 1.0f ? (float)label / numsamples / numcases : 0.0f;
	""",
	"hingeKer", preamble=atomicAddTmpl
)

smoothL1Ker = ElementwiseKernel(
	"const float *pred, const float *target, float *totalError, float *grad, float norm, float fullnorm",
	"""
	float diff = pred[i] - target[i];
	float sign = pred[i] - target[i] > 0.0f ? 1.0f : -1.0f;
	atomicAddCAS(totalError, diff * sign < 1.0f ? diff * diff / 2.0f * norm : (sign * diff - 0.5f) * norm);
	grad[i] = diff * sign < 1.0f ? diff * fullnorm : sign * fullnorm;
	""",
	"smoothL1Ker", preamble=atomicAddTmpl
)

l1HingeKer = ElementwiseKernel(
	"""
	const float *x1, const float *x2, const int *labels, float *totalError, float *g1, float *g2,
	int numsamples, int numcases
	""",
	"""
	float diff = x1[i] - x2[i];
	float sign = diff > 0.0f ? 1.0f : -1.0f;
	int label = labels[i / numcases];
	float error = (label == 0) ? max(0.0f, 1.0f - fabs(diff)) / numcases : fabs(diff) / numcases;
	atomicAddCAS(totalError, error);
	g1[i] = (label == 0 ? (fabs(diff) < 1.0f) * -sign : sign) / numsamples / numcases;
	g2[i] = (label == 0 ? (fabs(diff) < 1.0f) * sign : -sign) / numsamples / numcases;
	""",
	"l1HingeKer", preamble=atomicAddTmpl
)


costKernelCache = {}


def getAccuracyKernel(name):
	krl = costKernelCache.get(name, None)

	if krl is None:
		from PuzzleLib.OpenCL.Utils import context, queue

		if name == "calcAccuracy":
			krl = ReductionKernel(context, queue, np.float32, neutral="0.0f",
								  reduce_expr="a + b", map_expr="x[i] != y[i]", arguments="const int *x, const int *y")

		elif name == "calcBCEAccuracy":
			krl = ReductionKernel(context, queue, np.float32, neutral="0.0f", reduce_expr="a + b",
								  map_expr="y[i] == 1 ? x[i] <= 0.0f : x[i] > 0.0f",
								  arguments="const float *x, const int *y")

		elif name == "klDivergence":
			krl = ReductionKernel(context, queue, np.float32, neutral="0.0f", reduce_expr="a + b",
								  map_expr="grad[i] = (y[i] - x[i]) * gradnorm,"
										   "y[i] > 0.0f ? y[i] * (log(y[i]) - log(x[i])) : 0.0f",
								  arguments="const float *x, const float *y, float *grad, float gradnorm")

		elif name == "l1HingeAccuracy":
			krl = ReductionKernel(context, queue, np.float32, neutral="0.0f", reduce_expr="a + b",
								  map_expr="(d[i] <= 1.0f) != labels[i]", arguments="const float *d, const int *labels")

		else:
			raise RuntimeError("Unrecognized cost kernel name")

		costKernelCache[name] = krl

	return krl


costLblTmpl = Template("""

$atomicAdd


__kernel void cost(__global const float *scores, __global const int *labels, int loff, int size, int mapStride,
				   int spatialDim, int numCases, int numSamples, __global float *totalError, __global float *grad)
{
	for (int index = get_global_id(0); index < size; index += get_global_size(0))
	{
		int b = index / mapStride;
		int m = index % spatialDim;
		int c = (index / spatialDim) % numCases;

		$logic
	}
}

""")


crossEntropyLogic = """
float score = scores[index];
int label = labels[loff + b * spatialDim + m];
grad[index] = ((c == label) - score) / numSamples;

if (c == label)
{
	float error = -log(score) / spatialDim;
	atomicAddCAS(totalError, error);
}
"""


svmL1Logic = """
float score = scores[index];
int label = labels[loff + b * spatialDim + m];
float cls = (2 * (label == c) - 1);
grad[index] = score * cls < 1.0f ? cls / numCases / numSamples : 0.0f;

float error = max(0.0f, 1.0f - score * cls) / numCases / spatialDim;
atomicAddCAS(totalError, error);
"""


svmL2Logic = """
float score = scores[index];
int label = labels[loff + b * spatialDim + m];
float cls = (2 * (label == c) - 1);
float error = max(0.0f, 1.0f - score * cls);
grad[index] = 2.0f * cls * error / numCases / numSamples;

error = error * error / numCases / spatialDim;
atomicAddCAS(totalError, error);
"""


wceTmpl = Template("""

$atomicAdd


__kernel void cost(__global const float *scores, __global const int *labels, int loff, __global const float *weights,
				   int size, int mapStride, int spatialDim, int numCases, int numSamples, __global float *totalError,
				   __global float *grad)
{
	for (int index = get_global_id(0); index < size; index += get_global_size(0))
	{
		int b = index / mapStride;
		int m = index % spatialDim;
		int c = (index / spatialDim) % numCases;

		float score = scores[index];
		int label = labels[loff + b * spatialDim + m];
		float weight = weights[c];
		grad[index] = weight * ((c == label) - score) / numSamples;

		if (c == label)
		{
			float error = -weight * log(score) / spatialDim;
			atomicAddCAS(totalError, error);
		}
	}
}

""")


if context:
	ceMod = Driver.Program(context, costLblTmpl.substitute(atomicAdd=atomicAddTmpl, logic=crossEntropyLogic)).build()
	wceMod = Driver.Program(context, wceTmpl.substitute(atomicAdd=atomicAddTmpl)).build()

	svmL1Mod = Driver.Program(context, costLblTmpl.substitute(atomicAdd=atomicAddTmpl, logic=svmL1Logic)).build()
	svmL2Mod = Driver.Program(context, costLblTmpl.substitute(atomicAdd=atomicAddTmpl, logic=svmL2Logic)).build()


def crossEntropy(scores, labels, weights=None, error=None):
	assert scores.dtype == np.float32 and labels.dtype == np.int32

	shape = scores.shape
	if scores.ndim < 4:
		scores = scores.reshape(*shape, *(1 for _ in range(4 - scores.ndim)))

	softmax = softmax2d(scores)

	grad = Driver.empty(queue, shape, dtype=np.float32, allocator=memPool)
	if error is None:
		error = Driver.empty(queue, (), dtype=np.float32, allocator=memPool)

	error.fill(0.0)

	size = int(np.prod(scores.shape))
	spatialDim = int(np.prod(scores.shape[2:]))
	mapStride = spatialDim * scores.shape[1]

	block = (nthreads, 1, 1)
	grid = (roundUp(size, nthreads), 1, 1)

	if weights is None:
		ceMod.cost(queue, grid, block, softmax.data, labels.base_data, np.int32(labels.offset // labels.dtype.itemsize),
				   np.int32(size), np.int32(mapStride), np.int32(spatialDim), np.int32(scores.shape[1]),
				   np.int32(scores.shape[0]), error.data, grad.data)

	else:
		wceMod.cost(queue, grid, block, softmax.data, labels.base_data,
					np.int32(labels.offset // labels.dtype.itemsize), weights.data, np.int32(size), np.int32(mapStride),
					np.int32(spatialDim), np.int32(shape[1]), np.int32(shape[0]), error.data, grad.data)

	return error, grad


def svm(scores, labels, mode, error=None):
	assert scores.dtype == np.float32 and labels.dtype == np.int32
	shape = scores.shape

	grad = Driver.empty(queue, shape, dtype=np.float32, allocator=memPool)
	if error is None:
		error = Driver.empty(queue, (), dtype=np.float32, allocator=memPool)

	error.fill(0.0)

	size = int(np.prod(scores.shape))
	spatialDim = int(np.prod(scores.shape[2:]))
	mapStride = spatialDim * scores.shape[1]

	block = (nthreads, 1, 1)
	grid = (roundUp(size, nthreads), 1, 1)

	if mode == "l1":
		krl = svmL1Mod.cost
	elif mode == "l2":
		krl = svmL2Mod.cost
	else:
		raise ValueError()

	krl(queue, grid, block, scores.data, labels.base_data, np.int32(labels.offset // labels.dtype.itemsize),
		np.int32(size), np.int32(mapStride), np.int32(spatialDim), np.int32(shape[1]), np.int32(shape[0]),
		error.data, grad.data)

	return error, grad


def unittest():
	crossEntropyTest()
	svmTest()


def crossEntropyTest():
	scores = Driver.to_device(queue, np.random.randn(20, 10, 3).astype(np.float32))
	labels = Driver.to_device(queue, np.random.randint(low=0, high=10, size=(20, 3)).astype(np.int32))

	error, grad = crossEntropy(scores, labels)

	def softmax(w):
		e = np.exp(w - np.amax(w))
		dist = e / np.sum(e)
		return dist

	def hostCrossEntropy(smax, target):
		smax = np.moveaxis(smax, 1, -1).reshape(-1, smax.shape[1])
		target = target.flatten()
		err = np.sum(np.log(np.array([smax[i, target[i]] for i in range(smax.shape[0])])))

		return -err / target.size

	def hostCrossEntropyGrad(target, smax):
		return np.array([(target == i) - smax[i] for i in range(smax.shape[0])])

	hostSoftmax = np.apply_along_axis(softmax, 1, scores.get())

	hostGrad = np.vstack([hostCrossEntropyGrad(labels.get()[i], hostSoftmax[i]) / scores.shape[0]
						  for i in range(scores.shape[0])]).reshape(*hostSoftmax.shape)

	assert np.allclose(hostGrad, grad.get())

	hostError = hostCrossEntropy(hostSoftmax, labels.get())
	assert np.isclose(hostError, error.get() / scores.shape[0])


def svmTest():
	batchsize, size = 20, 4

	scores = Driver.to_device(queue, np.random.randn(batchsize, size).astype(np.float32))
	labels = Driver.to_device(queue, np.random.randint(low=0, high=size, size=(batchsize, ), dtype=np.int32))

	error, grad = svm(scores, labels, mode="l1")

	hostScores, hostLabels = scores.get(), labels.get()

	hostGrad = np.empty(grad.shape, dtype=np.float32)
	hostError = 0.0

	for b in range(batchsize):
		for n in range(size):
			cls = 2 * (hostLabels[b] == n) - 1
			val = hostScores[b, n] * cls

			hostGrad[b, n] = cls / batchsize / size if val < 1 else 0.0
			hostError += max(0.0, 1.0 - val) / batchsize / size

	assert np.allclose(hostGrad, grad.get())
	assert np.isclose(hostError, error.get() / scores.shape[0])


if __name__ == "__main__":
	unittest()
