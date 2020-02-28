import numpy as np
from PuzzleLib.OpenCL.Kernels.Templates import ElementwiseKernel


def memoize(fn):
	cache = {}

	def memoizer(*args):
		obj = cache.get(args, None)
		if obj is not None:
			return obj

		obj = fn(*args)
		cache[args] = obj

		return obj

	return memoizer


@memoize
def sigmoidKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata",
		"outdata[i] = 1.0f / (1.0f + exp(-indata[i]))",
		"sigmoidKer"
	)


@memoize
def sigmoidDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *ingrad, const float *outgrad, const float *outdata",
		"ingrad[i] = outgrad[i] * outdata[i] * (1.0f - outdata[i])",
		"sigmoidDerKer"
	)


@memoize
def	tanhKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata",
		"outdata[i] = tanh(indata[i])",
		"tanhKer"
	)


@memoize
def tanhDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *ingrad, const float *outgrad, const float *outdata",
		"ingrad[i] = outgrad[i] * (1.0f - outdata[i] * outdata[i])",
		"tanhDerKer"
	)


@memoize
def reluKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata",
		"outdata[i] = indata[i] * (indata[i] > 0.0f)",
		"reluKer"
	)


@memoize
def reluDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *ingrad, const float *outgrad, const float *outdata",
		"ingrad[i] = outgrad[i] * (outdata[i] > 0.0f)",
		"reluDerKer"
	)


@memoize
def leakyReluKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata, float a",
		"outdata[i] = indata[i] * ((indata[i] > 0.0f) + a * (indata[i] <= 0.0f))",
		"leakyReluKer"
	)


@memoize
def leakyReluDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *ingrad, const float *outgrad, const float *outdata, float a",
		"ingrad[i] = outgrad[i] * ((outdata[i] > 0.0f) + a * (outdata[i] <= 0.0f))",
		"leakyReluDerKer"
	)


@memoize
def eluKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata, float a",
		"outdata[i] = indata[i] * (indata[i] > 0.0f) + a * (exp(indata[i]) - 1.0f) * (indata[i] <= 0.0f)",
		"eluKer"
	)


@memoize
def eluDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *ingrad, const float *outgrad, const float *outdata, float a",
		"ingrad[i] = outgrad[i] * ((outdata[i] > 0.0f) + (outdata[i] + a) * (outdata[i] <= 0.0f))",
		"eluDerKer"
	)


@memoize
def softPlusKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata",
		"outdata[i] = log(1.0f + exp(indata[i]))",
		"softPlusKer"
	)


@memoize
def softPlusDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *ingrad, const float *outgrad, const float *outdata",
		"ingrad[i] = outgrad[i] * (1.0f - exp(-outdata[i]))",
		"softPlusDerKer"
	)


@memoize
def clipKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata, float a, float b",
		"outdata[i] = min(b, max(a, indata[i]))",
		"clipKer"
	)


@memoize
def clipDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *ingrad, const float *outgrad, const float *outdata, float a, float b",
		"ingrad[i] = outgrad[i] * (outdata[i] > a && outdata[i] < b);",
		"clipDerKer"
	)


@memoize
def dropoutKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata, const float *uni, float p",
		"outdata[i] = indata[i] * (uni[i] < p) / p",
		"dropoutKer"
	)


@memoize
def dropout2dKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata, const float *uni, float p, int mapsize",
		"outdata[i] = indata[i] * (uni[i / mapsize] < p) / p",
		"dropout2dKer"
	)


@memoize
def toVectorAddVectorKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata, float alpha",
		"outdata[i] += indata[i] * alpha",
		"toVectorAddVectorKer"
	)


@memoize
def adadeltaKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *param, const float *grad, float *msg, float *msdx, float rho, float epsilon",
		"""
		msg[i] += (1.0f - rho) * (grad[i] * grad[i] - msg[i]);
		float dx = sqrt((msdx[i] + epsilon) / (msg[i] + epsilon)) * grad[i];
		msdx[i] += (1.0f - rho) * (dx * dx - msdx[i]);
		param[i] += dx;
		""",
		"adadeltaKer"
	)


@memoize
def adagradKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *param, const float *grad, float *h, float learnRate, float epsilon",
		"h[i] += grad[i] * grad[i];"
		"param[i] += learnRate * grad[i] / (sqrt(h[i]) + epsilon)",
		"adagradKer"
	)


@memoize
def adamKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *param, const float *grad, float *mg, float *ms, float learnRate, float fix1, float fix2, float epsilon",
		"""
		mg[i] += fix1 * (grad[i] - mg[i]);
		ms[i] += fix2 * (grad[i] * grad[i] - ms[i]);
		param[i] += learnRate * mg[i] / (sqrt(ms[i]) + epsilon);
		""",
		"adamKer"
	)


@memoize
def classicMomSGDKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *param, const float *grad, float *mom, float learnRate, float momRate",
		"""
		mom[i] = momRate * mom[i] + learnRate * grad[i];
		param[i] += mom[i];
		""",
		"classicMomSGDKer"
	)


@memoize
def nesterovMomSGDKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *param, const float *grad, float *mom, float learnRate, float momRate",
		"""
		float m = mom[i];
		mom[i] = momRate * m + learnRate * grad[i];
		param[i] += momRate * momRate * m + (1.0f + momRate) * learnRate * grad[i];
		""",
		"nesterovMomSGDKer"
	)


@memoize
def rmspropKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *param, const float *grad, float *ms, float learnRate, float factor, float epsilon",
		"""
		ms[i] = factor * ms[i] + (1.0f - factor) * grad[i] * grad[i];
		param[i] += learnRate * grad[i] / (sqrt(ms[i]) + epsilon);
		""",
		"rmspropKer"
	)


@memoize
def rmspropGravesKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"""
		float *param, const float *grad, float *mg, float *ms, float *delta,
		float learnRate, float alpha, float momRate, float epsilon
		""",
		"""
		ms[i] = alpha * ms[i] + (1.0f - alpha) * grad[i] * grad[i];
		mg[i] = alpha * mg[i] + (1.0f - alpha) * grad[i];
		delta[i] = momRate * delta[i] + learnRate * grad[i] / sqrt(ms[i] - mg[i] * mg[i] + epsilon);
		param[i] += delta[i];
		""",
		"rmspropGravesKer"
	)


@memoize
def smorms3Ker(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *param, const float *grad, float *mem, float *g, float *g2, float learnRate, float epsilon",
		"""
		float memi = mem[i], r = 1.0f / (memi + 1.0f), gi = g[i], g2i = g2[i];
		gi = (1.0f - r) * gi + r * grad[i];
		g2i = (1.0f - r) * g2i + r * grad[i] * grad[i];
		float x = gi * gi / (g2i + epsilon);
		param[i] += grad[i] * min(learnRate, x) / (sqrt(g2i) + epsilon);
		mem[i] = 1.0f + memi * (1.0f - x);
		g[i] = gi;
		g2[i] = g2i;
		""",
		"smorms3Ker"
	)


@memoize
def addKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *a, float alpha, const float *b, float beta",
		"outdata[i] = alpha * a[i] + beta * b[i]",
		"addKer"
	)


@memoize
def mulKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *a, const float *b",
		"outdata[i] = a[i] * b[i]",
		"mulKer"
	)


@memoize
def linearKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		"float *outdata, const float *indata, float a, float b",
		"outdata[i] = a * indata[i] + b",
		"linearKer"
	)


rbmKer = ElementwiseKernel(
	"float *outdata, const float *indata, const float *uni",
	"float act = 1.0f / (1.0f + exp(-indata[i]));"
	"outdata[i] = (uni[i] < act)",
	"rbmKer"
)


absKer = ElementwiseKernel(
	"float *outdata, const float *indata",
	"outdata[i] = fabs(indata[i])",
	"absKer"
)


weightDecayKer = ElementwiseKernel(
	"float *grad, const float *param, float rate",
	"grad[i] -= rate * param[i]",
	"weightDecayKer"
)


l1penaltyKer = ElementwiseKernel(
	"float *outgrad, const float *ingrad, const float *data, float a",
	"outgrad[i] = ingrad[i] - a * ((0.0f <= data[i]) - (data[i] < 0.0f))",
	"l1penaltyKer"
)


l1gradKer = ElementwiseKernel(
	"float *grad, const float *pred, const float *target, float norm",
	"grad[i] = (pred[i] - target[i] > 0.0f ? -norm : norm)",
	"l1gradKer"
)


signKer = ElementwiseKernel(
	"float *outdata, const float *a, const float *b",
	"outdata[i] = a[i] * sign(b[i])",
	"signKer"
)
