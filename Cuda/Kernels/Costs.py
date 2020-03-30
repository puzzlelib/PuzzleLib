from string import Template
import numpy as np

from PuzzleLib.Compiler.Codegen.Types import int_t, float_t
from PuzzleLib.Cuda.Utils import prod, roundUpDiv


def bce(ElementwiseKernel):
	return ElementwiseKernel(
		[
			(float_t.const.ptr, "scores"), (int_t.const.ptr, "labels"),
			(float_t.ptr, "totalError"), (float_t.ptr, "grad"), (int_t, "numsamples"), (int_t, "spatialDim")
		],
		"""
		float prob = 1.0f / (1.0f + exp(-scores[i]));
		float error = labels[i] == 1 ? -log(prob) : -log(1.0f - prob);
		atomicAdd(totalError, error / spatialDim);
		grad[i] = ((labels[i] == 1) - prob) / numsamples / spatialDim;
		""",
		"bceKer"
	)


def hinge(ElementwiseKernel):
	return ElementwiseKernel(
		[
			(float_t.const.ptr, "scores"), (int_t.const.ptr, "labels"),
			(float_t.ptr, "totalError"), (float_t.ptr, "grad"), (int_t, "numsamples"), (int_t, "numcases")
		],
		"""
		float score = scores[i];
		int label = labels[i];
		float error = max(0.0f, 1.0f - score * label) / numcases;
		atomicAdd(totalError, error);
		grad[i] = score * label < 1.0f ? (float)label / numsamples / numcases : 0.0f;
		""",
		"hingeKer"
	)


def smoothL1(ElementwiseKernel):
	return ElementwiseKernel(
		[
			(float_t.const.ptr, "pred"), (float_t.const.ptr, "target"),
			(float_t.ptr, "totalError"), (float_t.ptr, "grad"), (float_t, "norm"), (float_t, "fullnorm")
		],
		"""
		float diff = pred[i] - target[i];
		float sign = pred[i] - target[i] > 0.0f ? 1.0f : -1.0f;
		atomicAdd(totalError, diff * sign < 1.0f ? diff * diff / 2.0f * norm : (sign * diff - 0.5f) * norm);
		grad[i] = diff * sign < 1.0f ? diff * fullnorm : sign * fullnorm;
		""",
		"smoothL1Ker"
	)


def l1Hinge(ElementwiseKernel):
	return ElementwiseKernel(
		[
			(float_t.const.ptr, "x1"), (float_t.const.ptr, "x2"), (int_t.const.ptr, "labels"),
			(float_t.ptr, "totalError"), (float_t.ptr, "g1"), (float_t.ptr, "g2"),
			(int_t, "numsamples"), (int_t, "numcases")
		],
		"""
		float diff = x1[i] - x2[i];
		float sign = diff > 0.0f ? 1.0f : -1.0f;
		int label = labels[i / numcases];
		float error = (label == 0) ? max(0.0f, 1.0f - abs(diff)) / numcases : abs(diff) / numcases;
		atomicAdd(totalError, error);
		g1[i] = (label == 0 ? (abs(diff) < 1.0f) * -sign : sign) / numsamples / numcases;
		g2[i] = (label == 0 ? (abs(diff) < 1.0f) * sign : -sign) / numsamples / numcases;
		""",
		"l1HingeKer"
	)


costLblTmpl = Template("""

extern "C"
__global__ void cost(const float *scores, const int *labels, int size, int mapStride, int spatialDim,
					 int numCases, int numSamples, float *totalError, float *grad)
{
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x)
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
int label = labels[b * spatialDim + m];
grad[index] = ((c == label) - score) / numSamples;

if (c == label)
{
	float error = -log(score) / spatialDim;
	atomicAdd(totalError, error);
}
"""


svmL1Logic = """
float score = scores[index];
int label = labels[b * spatialDim + m];
float cls = (2 * (label == c) - 1);
grad[index] = score * cls < 1.0f ? cls / numCases / numSamples : 0.0f;

float error = max(0.0f, 1.0f - score * cls) / numCases / spatialDim;
atomicAdd(totalError, error);
"""


svmL2Logic = """
float score = scores[index];
int label = labels[b * spatialDim + m];
float cls = (2 * (label == c) - 1);
float error = max(0.0f, 1.0f - score * cls);
grad[index] = 2.0f * cls * error / numCases / numSamples;

error = error * error / numCases / spatialDim;
atomicAdd(totalError, error);
"""


wceTmpl = Template("""

extern "C"
__global__ void cost(const float *scores, const int *labels, const float *weights, int size, int mapStride,
					 int spatialDim, int numCases, int numSamples, float *totalError, float *grad)
{
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < size; index += blockDim.x * gridDim.x)
	{
		int b = index / mapStride;
		int m = index % spatialDim;
		int c = (index / spatialDim) % numCases;

		float score = scores[index];
		int label = labels[b * spatialDim + m];
		float weight = weights[c];
		grad[index] = weight * ((c == label) - score) / numSamples;

		if (c == label)
		{
			float error = -weight * log(score) / spatialDim;
			atomicAdd(totalError, error);
		}
	}
}

""")


class CostModule:
	def __init__(self, backend):
		self.backend, self.GPUArray, self.dnn = backend, backend.GPUArray, backend.dnn
		self.nthreads = backend.nthreads

		self.ceMod = backend.SourceModule(costLblTmpl.substitute(logic=crossEntropyLogic))
		self.wceMod = backend.SourceModule(wceTmpl.substitute())

		self.svmL1Mod = backend.SourceModule(costLblTmpl.substitute(logic=svmL1Logic))
		self.svmL2Mod = backend.SourceModule(costLblTmpl.substitute(logic=svmL2Logic))

		self.accKernelCache = {}


	def getAccuracyKernel(self, name):
		krl = self.accKernelCache.get(name, None)

		if krl is None:
			if name == "calcAccuracy":
				krl = self.backend.ReductionKernel(
					np.float32, neutral="0.0f", reduceExpr="a + b", mapExpr="x[i] != y[i]",
					arguments=[(int_t.const.ptr, "x"), (int_t.const.ptr, "y")], name="calcAccuracy"
				)

			elif name == "calcBCEAccuracy":
				krl = self.backend.ReductionKernel(
					np.float32, neutral="0.0f", reduceExpr="a + b", mapExpr="y[i] == 1 ? x[i] <= 0.0f : x[i] > 0.0f",
					arguments=[(float_t.const.ptr, "x"), (int_t.const.ptr, "y")], name="calcBCEAccuracy"
				)

			elif name == "klDivergence":
				krl = self.backend.ReductionKernel(
					np.float32, neutral="0.0f", reduceExpr="a + b", mapExpr=
					"grad[i] = (y[i] - x[i]) * gradnorm, y[i] > 0.0f ? y[i] * (log(y[i]) - log(x[i])) : 0.0f",
					arguments=[
						(float_t.const.ptr, "x"), (float_t.const.ptr, "y"), (float_t.ptr, "grad"), (float_t, "gradnorm")
					], name="klDivergence"
				)

			elif name == "l1HingeAccuracy":
				krl = self.backend.ReductionKernel(
					np.float32, neutral="0.0f", reduceExpr="a + b", mapExpr="(d[i] <= 1.0f) != labels[i]",
					arguments=[(float_t.const.ptr, "d"), (int_t.const.ptr, "labels")], name="l1HingeAccuracy"
				)

			else:
				raise NotImplementedError(name)

			self.accKernelCache[name] = krl

		return krl


	def crossEntropy(self, scores, labels, weights=None, error=None, allocator=None):
		assert scores.dtype == np.float32 and labels.dtype == np.int32

		shape = scores.shape
		if scores.ndim < 4:
			scores = scores.reshape(*shape, *(1 for _ in range(4 - scores.ndim)))

		softmax = self.dnn.softmaxNd(scores, mode=self.backend.SoftMaxMode.spatial.value, allocator=allocator)

		grad = self.GPUArray.empty(shape, dtype=np.float32, allocator=allocator)
		if error is None:
			error = self.GPUArray.empty((), dtype=np.float32, allocator=allocator)

		error.fill(0.0)

		size = prod(scores.shape)
		spatialDim = prod(scores.shape[2:])
		mapStride = spatialDim * scores.shape[1]

		block = (self.nthreads, 1, 1)
		grid = (roundUpDiv(size, self.nthreads), 1, 1)

		if weights is None:
			self.ceMod.cost(
				softmax, labels, np.int32(size), np.int32(mapStride), np.int32(spatialDim),
				np.int32(scores.shape[1]), np.int32(scores.shape[0]), error, grad, block=block, grid=grid
			)

		else:
			self.wceMod.cost(
				softmax, labels, weights, np.int32(size), np.int32(mapStride), np.int32(spatialDim),
				np.int32(shape[1]), np.int32(shape[0]), error, grad, block=block, grid=grid
			)

		return error, grad


	def svm(self, scores, labels, mode, error=None, allocator=None):
		assert scores.dtype == np.float32 and labels.dtype == np.int32
		shape = scores.shape

		grad = self.GPUArray.empty(shape, dtype=np.float32, allocator=allocator)
		if error is None:
			error = self.GPUArray.empty((), dtype=np.float32, allocator=allocator)

		error.fill(0.0)

		size = prod(scores.shape)
		spatialDim = prod(scores.shape[2:])
		mapStride = spatialDim * scores.shape[1]

		block = (self.nthreads, 1, 1)
		grid = (roundUpDiv(size, self.nthreads), 1, 1)

		mod = {
			"l1": self.svmL1Mod,
			"l2": self.svmL2Mod
		}[mode]

		mod.cost(
			scores, labels, np.int32(size), np.int32(mapStride), np.int32(spatialDim),
			np.int32(shape[1]), np.int32(shape[0]), error, grad, block=block, grid=grid
		)

		return error, grad


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		module = CostModule(Backend.getBackend(deviceIdx, initmode=1))

		crossEntropyTest(module)
		svmTest(module)


def crossEntropyTest(module):
	hostScores = np.random.randn(20, 10, 3).astype(np.float32)
	hostLabels = np.random.randint(low=0, high=10, size=(20, 3)).astype(np.int32)

	scores, labels = module.GPUArray.toGpu(hostScores), module.GPUArray.toGpu(hostLabels)
	error, grad = module.crossEntropy(scores, labels)

	def softmax(w):
		e = np.exp(w - np.amax(w))
		dist = e / np.sum(e)
		return dist

	def hostCrossEntropy(smax, target):
		smax = np.moveaxis(smax, 1, -1).reshape(-1, smax.shape[1])
		target = target.ravel()
		err = np.sum(np.log(np.array([smax[i, target[i]] for i in range(smax.shape[0])])))

		return -err / target.size

	def hostCrossEntropyGrad(target, smax):
		return np.array([(target == i) - smax[i] for i in range(smax.shape[0])])

	hostSoftmax = np.apply_along_axis(softmax, 1, hostScores)

	hostGrad = np.vstack([
		hostCrossEntropyGrad(hostLabels[i], hostSoftmax[i]) / scores.shape[0] for i in range(scores.shape[0])
	]).reshape(*hostSoftmax.shape)

	assert np.allclose(hostGrad, grad.get())

	hostError = hostCrossEntropy(hostSoftmax, hostLabels)
	assert np.isclose(hostError, error.get() / scores.shape[0])


def svmTest(module):
	batchsize, size = 20, 4

	hostScores = np.random.randn(batchsize, size).astype(np.float32)
	hostLabels = np.random.randint(low=0, high=size, size=(batchsize, ), dtype=np.int32)

	scores, labels = module.GPUArray.toGpu(hostScores), module.GPUArray.toGpu(hostLabels)
	error, grad = module.svm(scores, labels, mode="l1")

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
