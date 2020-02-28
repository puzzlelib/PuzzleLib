from string import Template

import numpy as np

from PuzzleLib.Compiler.Codegen.Types import void_t, int32_t, float_t, ptrdiff_t

from PuzzleLib.CPU.SourceModule import SourceModule, ElementwiseKernel, ReductionKernel
from PuzzleLib.CPU.CPUArray import CPUArray

from PuzzleLib.Intel.Wrappers.DNNL import softmaxNd


bceKer = ElementwiseKernel(
	[
		(float_t.const.ptr, "scores"), (int32_t.const.ptr, "labels"),
		(float_t.ptr, "totalError"), (float_t.ptr, "grad"), (int32_t, "numsamples"), (int32_t, "spatialDim")
	],
	"""
	float prob = 1.0f / (1.0f + expf(-scores[i]));

	float error = labels[i] == 1 ? -logf(prob) : -logf(1.0f - prob);
	*totalError += error / spatialDim;

	grad[i] = ((labels[i] == 1) - prob) / numsamples / spatialDim;
	""",
	"bceKer"
)

hingeKer = ElementwiseKernel(
	[
		(float_t.const.ptr, "scores"), (int32_t.const.ptr, "labels"),
		(float_t.ptr, "totalError"), (float_t.ptr, "grad"), (int32_t, "numsamples"), (int32_t, "numcases")
	],
	"""
	float score = scores[i];
	int label = labels[i];

	float error = 1.0f - score * label;
	*totalError += error > 0.0f ? error / numcases : 0.0f;

	grad[i] = error > 0.0f ? (float)label / numsamples / numcases : 0.0f;
	""",
	"hingeKer"
)

smoothL1Ker = ElementwiseKernel(
	[
		(float_t.const.ptr, "pred"), (float_t.const.ptr, "target"), (float_t.ptr, "totalError"),
		(float_t.ptr, "grad"), (float_t, "norm"), (float_t, "fullnorm")
	],
	"""
	float diff = pred[i] - target[i];

	float sign = diff > 0.0f ? 1.0f : -1.0f;
	diff = fabsf(diff);

	*totalError += diff < 1.0f ? diff * diff / 2.0f * norm : (diff - 0.5f) * norm;
	grad[i] = sign * (diff < 1.0f ? diff * fullnorm : fullnorm);
	""",
	"smoothL1Ker"
)

l1HingeKer = ElementwiseKernel(
	[
		(float_t.const.ptr, "x1"), (float_t.const.ptr, "x2"), (int32_t.const.ptr, "labels"),
		(float_t.ptr, "totalError"), (float_t.ptr, "g1"), (float_t.ptr, "g2"),
		(int32_t, "numsamples"), (int32_t, "numcases")
	],
	"""
	float diff = x1[i] - x2[i];
	float sign = diff > 0.0f ? 1.0f : -1.0f;

	diff = fabsf(diff);
	int label = labels[i / numcases];

	float error = (label == 0) ? (diff > 1.0f ? 0.0f : 1.0f - diff) / numcases : diff / numcases;
	*totalError += error;

	g1[i] = (label == 0 ? (diff < 1.0f) * -sign : sign) / numsamples / numcases;
	g2[i] = (label == 0 ? (diff < 1.0f) * sign : -sign) / numsamples / numcases;
	""",
	"l1HingeKer"
)


accKernelCache = {}


def getAccuracyKernel(name):
	krl = accKernelCache.get(name, None)

	if krl is None:
		if name == "calcAccuracy":
			krl = ReductionKernel(
				np.float32, neutral="0.0f", reduceExpr="a + b", mapExpr="x[i] != y[i]",
				arguments=[(int32_t.const.ptr, "x"), (int32_t.const.ptr, "y")]
			)

		elif name == "calcBCEAccuracy":
			krl = ReductionKernel(
				np.float32, neutral="0.0f", reduceExpr="a + b", mapExpr="y[i] == 1 ? x[i] <= 0.0f : x[i] > 0.0f",
				arguments=[(float_t.const.ptr, "x"), (int32_t.const.ptr, "y")]
			)

		elif name == "klDivergence":
			krl = ReductionKernel(
				np.float32, neutral="0.0f", reduceExpr="a + b",
				mapExpr="grad[i] = (y[i] - x[i]) * gradnorm, y[i] > 0.0f ? y[i] * (logf(y[i]) - logf(x[i])) : 0.0f",
				arguments=[
					(float_t.const.ptr, "x"), (float_t.const.ptr, "y"), (float_t.ptr, "grad"), (float_t, "gradnorm")
				]
			)

		elif name == "l1HingeAccuracy":
			krl = ReductionKernel(
				np.float32, neutral="0.0f", reduceExpr="a + b", mapExpr="(d[i] <= 1.0f) != labels[i]",
				arguments=[(float_t.const.ptr, "d"), (int32_t.const.ptr, "labels")]
			)

		else:
			raise RuntimeError("Unrecognized cost kernel name")

		accKernelCache[name] = krl

	return krl


costLblTmpl = Template("""

static void cost(const float * __restrict scores, const int32_t * __restrict labels, int32_t mapStride,
				 int32_t spatialDim, int32_t numCases, int32_t numSamples, float * __restrict totalError,
				 float * __restrict grad, ptrdiff_t size)
{
	for (ptrdiff_t i = 0; i < size; i++)
	{
		ptrdiff_t b = i / mapStride, m = i % spatialDim, c = (i / spatialDim) % numCases;

		$logic
	}
}

""")


crossEntropyLogic = """
float score = scores[i];
int32_t label = labels[b * spatialDim + m];

grad[i] = ((c == label) - score) / numSamples;

if (c == label)
{
	float error = -logf(score) / spatialDim;
	*totalError += error;
}
"""


svmL1Logic = """
float score = scores[i];
int32_t label = labels[b * spatialDim + m];
float cls = (float)(2 * (label == c) - 1);

float error = 1.0f - score * cls;
*totalError += error > 0.0f ? error / numCases / spatialDim : 0.0f;

grad[i] = error > 0.0f ? cls / numCases / numSamples : 0.0f;
"""


svmL2Logic = """
float score = scores[i];
int32_t label = labels[b * spatialDim + m];
float cls = (float)(2 * (label == c) - 1);

float error = 1.0f - score * cls;
*totalError += error > 0.0f ? error * error / numCases / spatialDim : 0.0f;

grad[i] = error > 0.0f ? 2.0f * cls * error / numCases / numSamples : 0.0f;
"""


wceTmpl = """

static void cost(const float * __restrict scores, const int32_t * __restrict labels, const float * __restrict weights,
				 int32_t mapStride, int32_t spatialDim, int32_t numCases, int32_t numSamples,
				 float * __restrict totalError, float * __restrict grad, ptrdiff_t size)
{
	for (ptrdiff_t i = 0; i < size; i++)
	{
		ptrdiff_t b = i / mapStride, m = i % spatialDim, c = (i / spatialDim) % numCases;

		float score = scores[i];
		int32_t label = labels[b * spatialDim + m];
		float weight = weights[c];

		grad[i] = weight * ((c == label) - score) / numSamples;

		if (c == label)
		{
			float error = -weight * logf(score) / spatialDim;
			*totalError += error;
		}
	}
}

"""


ceMod = SourceModule(costLblTmpl.substitute(logic=crossEntropyLogic), functions=[
	("cost", void_t, [
		(float_t.const.ptr.restrict, "scores"), (int32_t.const.ptr.restrict, "labels"),
		(int32_t, "mapStride"), (int32_t, "spatialDim"), (int32_t, "numCases"), (int32_t, "numSamples"),
		(float_t.ptr.restrict, "totalError"), (float_t.ptr.restrict, "grad"), (ptrdiff_t, "size")
	])
])
wceMod = SourceModule(wceTmpl, functions=[
	("cost", void_t, [
		(float_t.const.ptr.restrict, "scores"), (int32_t.const.ptr.restrict, "labels"),
		(float_t.const.ptr.restrict, "weights"), (int32_t, "mapStride"), (int32_t, "spatialDim"), (int32_t, "numCases"),
		(int32_t, "numSamples"), (float_t.ptr.restrict, "totalError"), (float_t.ptr.restrict, "grad"),
		(ptrdiff_t, "size")
	])
])

svmL1Mod = SourceModule(costLblTmpl.substitute(logic=svmL1Logic), functions=[
	("cost", void_t, [
		(float_t.const.ptr.restrict, "scores"), (int32_t.const.ptr.restrict, "labels"),
		(int32_t, "mapStride"), (int32_t, "spatialDim"), (int32_t, "numCases"), (int32_t, "numSamples"),
		(float_t.ptr.restrict, "totalError"), (float_t.ptr.restrict, "grad"), (ptrdiff_t, "size")
	])
])
svmL2Mod = SourceModule(costLblTmpl.substitute(logic=svmL2Logic), functions=[
	("cost", void_t, [
		(float_t.const.ptr.restrict, "scores"), (int32_t.const.ptr.restrict, "labels"),
		(int32_t, "mapStride"), (int32_t, "spatialDim"), (int32_t, "numCases"), (int32_t, "numSamples"),
		(float_t.ptr.restrict, "totalError"), (float_t.ptr.restrict, "grad"), (ptrdiff_t, "size")
	])
])


def crossEntropy(scores, labels, weights=None, error=None):
	assert scores.dtype == np.float32 and labels.dtype == np.int32

	shape = scores.shape
	if scores.ndim < 4:
		scores = scores.reshape(*shape, *(1 for _ in range(4 - scores.ndim)))

	softmax = softmaxNd(scores)

	grad = CPUArray.empty(shape, dtype=np.float32)
	if error is None:
		error = CPUArray.empty((), dtype=np.float32)

	error.fill(0.0)

	spatialDim = int(np.prod(scores.shape[2:]))
	mapStride = spatialDim * scores.shape[1]

	if weights is None:
		ceMod.cost(
			softmax.data, labels.data, mapStride, spatialDim, scores.shape[1], scores.shape[0], error.data, grad.data,
			softmax.size
		)

	else:
		wceMod.cost(
			softmax.data, labels.data, weights.data,  mapStride, spatialDim, shape[1], shape[0], error.data, grad.data,
			softmax.size
		)

	return error, grad


def svm(scores, labels, mode, error=None):
	assert scores.dtype == np.float32 and labels.dtype == np.int32
	shape = scores.shape

	grad = CPUArray.empty(shape, dtype=np.float32)
	if error is None:
		error = CPUArray.empty((), dtype=np.float32)

	error.fill(0.0)

	spatialDim = int(np.prod(scores.shape[2:]))
	mapStride = spatialDim * scores.shape[1]

	if mode == "l1":
		krl = svmL1Mod.cost
	elif mode == "l2":
		krl = svmL2Mod.cost
	else:
		raise ValueError()

	krl(scores.data, labels.data, mapStride, spatialDim, shape[1], shape[0], error.data, grad.data, scores.size)
	return error, grad


def unittest():
	crossEntropyTest()
	svmTest()


def crossEntropyTest():
	scores = CPUArray.toDevice(np.random.randn(20, 10, 3).astype(np.float32))
	labels = CPUArray.toDevice(np.random.randint(low=0, high=10, size=(20, 3)).astype(np.int32))

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

	scores = CPUArray.toDevice(np.random.randn(batchsize, size).astype(np.float32))
	labels = CPUArray.toDevice(np.random.randint(low=0, high=size, size=(batchsize, ), dtype=np.int32))

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
