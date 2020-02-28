import numpy as np

from PuzzleLib.Compiler.Codegen.Types import int32_t, uint32_t, float_t

from PuzzleLib.CPU.SourceModule import ElementwiseKernel
from PuzzleLib.CPU.Utils import memoize


@memoize
def sigmoidKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata")],
		"outdata[i] = 1.0f / (1.0f + expf(-indata[i]))",
		"sigmoidKer"
	)


@memoize
def sigmoidDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "ingrad"), (float_t.const.ptr, "outgrad"),
			(float_t.const.ptr, "outdata")
		],
		"ingrad[i] = outgrad[i] * outdata[i] * (1.0f - outdata[i])",
		"sigmoidDerKer"
	)


@memoize
def tanhKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata")],
		"outdata[i] = tanhf(indata[i])",
		"tanhKer"
	)


@memoize
def tanhDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "ingrad"), (float_t.const.ptr, "outgrad"), (float_t.const.ptr, "outdata")],
		"ingrad[i] = outgrad[i] * (1.0f - outdata[i] * outdata[i])",
		"tanhDerKer"
	)


@memoize
def reluKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata")],
		"outdata[i] = indata[i] * (indata[i] > 0.0f)",
		"reluKer"
	)


@memoize
def reluDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "ingrad"), (float_t.const.ptr, "outgrad"), (float_t.const.ptr, "outdata")],
		"ingrad[i] = outgrad[i] * (outdata[i] > 0.0f)",
		"reluDerKer"
	)


@memoize
def leakyReluKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata"), (float_t, "a")],
		"outdata[i] = indata[i] * ((indata[i] > 0.0f) + a * (indata[i] <= 0.0f))",
		"leakyReluKer"
	)


@memoize
def leakyReluDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "ingrad"), (float_t.const.ptr, "outgrad"), (float_t.const.ptr, "outdata"), (float_t, "a")],
		"ingrad[i] = outgrad[i] * ((outdata[i] > 0.0f) + a * (outdata[i] <= 0.0f))",
		"leakyReluDerKer"
	)


@memoize
def eluKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata"), (float_t, "a")],
		"outdata[i] = indata[i] * (indata[i] > 0.0f) + a * (expf(indata[i]) - 1.0f) * (indata[i] <= 0.0f)",
		"eluKer"
	)


@memoize
def eluDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "ingrad"), (float_t.const.ptr, "outgrad"), (float_t.const.ptr, "outdata"), (float_t, "a")],
		"ingrad[i] = outgrad[i] * ((outdata[i] > 0.0f) + (outdata[i] + a) * (outdata[i] <= 0.0f))",
		"eluDerKer"
	)


@memoize
def softPlusKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata")],
		"outdata[i] = logf(1.0f + expf(indata[i]))",
		"softPlusKer"
	)


@memoize
def softPlusDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "ingrad"), (float_t.const.ptr, "outgrad"), (float_t.const.ptr, "outdata")],
		"ingrad[i] = outgrad[i] * (1.0f - expf(-outdata[i]))",
		"softPlusDerKer"
	)


@memoize
def clipKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata"), (float_t, "a"), (float_t, "b")],
		"outdata[i] = indata[i] * (indata[i] > a && indata[i] < b) + a * (indata[i] <= a) + b * (indata[i] >= b)",
		"clipKer"
	)


@memoize
def clipDerKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "ingrad"), (float_t.const.ptr, "outgrad"), (float_t.const.ptr, "outdata"),
			(float_t, "a"), (float_t, "b")
		],
		"ingrad[i] = outgrad[i] * (outdata[i] > a && outdata[i] < b);",
		"clipDerKer"
	)


@memoize
def dropoutKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "outdata"), (float_t.const.ptr, "indata"), (uint32_t.const.ptr, "b"),
			(uint32_t, "v"), (float_t, "p")
		],
		"outdata[i] = indata[i] * (b[i] < v) / p",
		"dropoutKer"
	)


@memoize
def dropout2dKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "outdata"), (float_t.const.ptr, "indata"), (uint32_t.const.ptr, "b"), (uint32_t, "v"),
			(float_t, "p"), (int32_t, "mapsize")
		],
		"outdata[i] = indata[i] * (b[i / mapsize] < v) / p",
		"dropout2dKer"
	)


@memoize
def toVectorAddVectorKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata"), (float_t, "alpha")],
		"outdata[i] += indata[i] * alpha",
		"toVectorAddVectorKer"
	)


addVectorToVectorKer = ElementwiseKernel(
	[
		(float_t.ptr, "outdata"), (float_t.const.ptr, "x"), (float_t.const.ptr, "y"),
		(float_t, "alpha"), (float_t, "beta")
	],
	"outdata[i] = x[i] * alpha + y[i] * beta",
	"addVectorToVectorKer"
)


@memoize
def adadeltaKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "param"), (float_t.const.ptr, "grad"), (float_t.ptr, "msg"), (float_t.ptr, "msdx"),
			(float_t, "rho"), (float_t, "epsilon")
		],
		"""
		msg[i] += (1.0f - rho) * (grad[i] * grad[i] - msg[i]);
		float dx = sqrtf((msdx[i] + epsilon) / (msg[i] + epsilon)) * grad[i];
		msdx[i] += (1.0f - rho) * (dx * dx - msdx[i]);
		param[i] += dx;
		""",
		"adadeltaKer"
	)


@memoize
def adagradKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "param"), (float_t.const.ptr, "grad"), (float_t.ptr, "h"),
			(float_t, "learnRate"), (float_t, "epsilon")
		],
		"""
		h[i] += grad[i] * grad[i];
		param[i] += learnRate * grad[i] / (sqrtf(h[i]) + epsilon);
		""",
		"adagradKer"
	)


@memoize
def adamKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "param"), (float_t.const.ptr, "grad"), (float_t.ptr, "mg"), (float_t.ptr, "ms"),
			(float_t, "learnRate"), (float_t, "fix1"), (float_t, "fix2"), (float_t, "epsilon")
		],
		"""
		mg[i] += fix1 * (grad[i] - mg[i]);
		ms[i] += fix2 * (grad[i] * grad[i] - ms[i]);
		param[i] += learnRate * mg[i] / (sqrtf(ms[i]) + epsilon);
		""",
		"adamKer"
	)


@memoize
def classicMomSGDKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "param"), (float_t.const.ptr, "grad"), (float_t.ptr, "mom"),
			(float_t, "learnRate"), (float_t, "momRate")
		],
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
		[
			(float_t.ptr, "param"), (float_t.const.ptr, "grad"), (float_t.ptr, "mom"),
			(float_t, "learnRate"), (float_t, "momRate")
		],
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
		[
			(float_t.ptr, "param"), (float_t.const.ptr, "grad"), (float_t.ptr, "ms"),
			(float_t, "learnRate"), (float_t, "factor"), (float_t, "epsilon")
		],
		"""
		ms[i] = factor * ms[i] + (1.0f - factor) * grad[i] * grad[i];
		param[i] += learnRate * grad[i] / (sqrtf(ms[i]) + epsilon);
		""",
		"rmspropKer"
	)


@memoize
def rmspropGravesKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "param"), (float_t.const.ptr, "grad"), (float_t.ptr, "mg"), (float_t.ptr, "ms"),
			(float_t.ptr, "delta"), (float_t, "learnRate"), (float_t, "alpha"),
			(float_t, "momRate"), (float_t, "epsilon")
		],
		"""
		ms[i] = alpha * ms[i] + (1.0f - alpha) * grad[i] * grad[i];
		mg[i] = alpha * mg[i] + (1.0f - alpha) * grad[i];
		delta[i] = momRate * delta[i] + learnRate * grad[i] / sqrtf(ms[i] - mg[i] * mg[i] + epsilon);
		param[i] += delta[i];
		""",
		"rmspropGravesKer"
	)


@memoize
def smorms3Ker(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "param"), (float_t.const.ptr, "grad"), (float_t.ptr, "mem"),
			(float_t.ptr, "mg"), (float_t.ptr, "ms"), (float_t, "learnRate"), (float_t, "epsilon")
		],
		"""
		float r = 1.0f / (mem[i] + 1.0f);

		float mgi = (1.0f - r) * mg[i] + r * grad[i];
		float msi = (1.0f - r) * ms[i] + r * grad[i] * grad[i];
		float x = mgi * mgi / (msi + epsilon);
	
		mem[i] = 1.0f + mem[i] * (1.0f - x), mg[i] = mgi, ms[i] = msi;
		param[i] += grad[i] * fminf(learnRate, x) / (sqrtf(msi) + epsilon);
		""",
		"smorms3Ker"
	)


@memoize
def addKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[
			(float_t.ptr, "outdata"), (float_t.const.ptr, "a"), (float_t, "alpha"),
			(float_t.const.ptr, "b"), (float_t, "beta")
		],
		"outdata[i] = alpha * a[i] + beta * b[i]",
		"addKer"
	)


@memoize
def mulKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "a"), (float_t.const.ptr, "b")],
		"outdata[i] = a[i] * b[i]",
		"mulKer"
	)


@memoize
def linearKer(dtype):
	assert dtype == np.float32
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata"), (float_t, "a"), (float_t, "b")],
		"outdata[i] = a * indata[i] + b",
		"linearKer"
	)


rbmKer = ElementwiseKernel(
	[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata"), (float_t.const.ptr, "uni")],
	"float act = 1.0f / (1.0f + expf(-indata[i]));"
	"outdata[i] = (float)(uni[i] < act)",
	"rbmKer"
)

weightDecayKer = ElementwiseKernel(
	[(float_t.ptr, "grad"), (float_t.const.ptr, "param"), (float_t, "rate")],
	"grad[i] -= rate * param[i]",
	"weightDecayKer"
)


absKer = ElementwiseKernel(
	[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata")],
	"outdata[i] = fabsf(indata[i])",
	"absKer"
)

l1penaltyKer = ElementwiseKernel(
	[(float_t.ptr, "outgrad"), (float_t.const.ptr, "ingrad"), (float_t.ptr, "data"), (float_t, "a")],
	"outgrad[i] = ingrad[i] - a * ((0.0f <= data[i]) - (data[i] < 0.0f))",
	"l1penaltyKer"
)

l1gradKer = ElementwiseKernel(
	[(float_t.ptr, "grad"), (float_t.const.ptr, "pred"), (float_t.const.ptr, "target"), (float_t, "norm")],
	"grad[i] = (pred[i] > target[i] ? -norm : norm)",
	"l1gradKer"
)
