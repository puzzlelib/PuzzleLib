import numpy as np

from PuzzleLib.Compiler.Codegen.Types import ushort_t, int_t, uint_t, half_t, float_t

from PuzzleLib.Cuda.GPUArray import memoize
from PuzzleLib.Cuda.SourceModule import dtypeToCtype


def sigmoid(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def sigmoidKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "sigmoidKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata")]

		preambule = "__forceinline__ __device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }"
		operation = "outdata[i] = sigmoid((float)indata[i])"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 v = __half22float2(indata2[i]);
				outdata2[i] = __float22half2_rn(make_float2(sigmoid(v.x), sigmoid(v.y)));
				""",
				operation, name, preambule=preambule)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return sigmoidKer


def sigmoidDer(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def sigmoidDerKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "sigmoidDerKer"
		arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata")]

		preambule = "__forceinline__ __device__ float sigmoidDer(float g, float d) { return g * d * (1.0f - d); }"
		operation = "ingrad[i] = sigmoidDer((float)outgrad[i], (float)outdata[i])"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 gvec = __half22float2(outgrad2[i]), dvec = __half22float2(outdata2[i]);
				ingrad2[i] = __float22half2_rn(make_float2(sigmoidDer(gvec.x, dvec.x), sigmoidDer(gvec.y, dvec.y)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return sigmoidDerKer


def tanh(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def tanhKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "tanhKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata")]

		operation = "outdata[i] = tanhf((float)indata[i])"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 v = __half22float2(indata2[i]);
				outdata2[i] = __float22half2_rn(make_float2(tanhf(v.x), tanhf(v.y)));
				""",
				operation, name
			)

		else:
			return ElementwiseKernel(arguments, operation, name)

	return tanhKer


def tanhDer(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def tanhDerKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "tanhDerKer"
		arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata")]

		preambule = "__forceinline__ __device__ float tanhDer(float g, float d) { return g * (1.0f - d * d); }"
		operation = "ingrad[i] = tanhDer((float)outgrad[i], (float)outdata[i])"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 gvec = __half22float2(outgrad2[i]), dvec = __half22float2(outdata2[i]);
				ingrad2[i] = __float22half2_rn(make_float2(tanhDer(gvec.x, dvec.x), tanhDer(gvec.y, dvec.y)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return tanhDerKer


def relu(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def reluKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "reluKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata")]

		preambule = "__forceinline__ __device__ float relu(float x) { return x * (x > 0.0f); }"
		operation = "outdata[i] = relu((float)indata[i])"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 v = __half22float2(indata2[i]);
				outdata2[i] = __float22half2_rn(make_float2(relu(v.x), relu(v.y)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return reluKer


def reluDer(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def reluDerKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "reluDerKer"
		arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata")]

		preambule = "__forceinline__ __device__ float reluDer(float g, float d) { return g * (d > 0.0f); }"
		operation = "ingrad[i] = reluDer((float)outgrad[i], (float)outdata[i])"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 gvec = __half22float2(outgrad2[i]), dvec = __half22float2(outdata2[i]);
				ingrad2[i] = __float22half2_rn(make_float2(reluDer(gvec.x, dvec.x), reluDer(gvec.y, dvec.y)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return reluDerKer


def leakyRelu(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def leakyReluKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "leakyReluKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (float_t, "a")]

		preambule = """
		__forceinline__ __device__ float leakyRelu(float x, float a) { return x * ((x > 0.0f) + a * (x <= 0.0f)); }
		"""

		operation = "outdata[i] = leakyRelu((float)indata[i], a)"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 v = __half22float2(indata2[i]);
				outdata2[i] = __float22half2_rn(make_float2(leakyRelu(v.x, a), leakyRelu(v.y, a)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return leakyReluKer


def leakyReluDer(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def leakyReluDerKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "leakyReluDerKer"
		arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata"), (float_t, "a")]

		preambule = """
		__forceinline__ __device__
		float leakyReluDer(float g, float d, float a) { return g * ((d > 0.0f) + a * (d <= 0.0f)); }
		"""

		operation = "ingrad[i] = leakyReluDer((float)outgrad[i], (float)outdata[i], a)"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 gvec = __half22float2(outgrad2[i]), dvec = __half22float2(outdata2[i]);
				ingrad2[i] = __float22half2_rn(make_float2(
					leakyReluDer(gvec.x, dvec.x, a), leakyReluDer(gvec.y, dvec.y, a)
				));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return leakyReluDerKer


def elu(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def eluKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "eluKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (float_t, "a")]

		preambule = """
		__forceinline__ __device__
		float elu(float x, float a) { return x * (x > 0.0f) + a * (expf(x) - 1.0f) * (x <= 0.0f); }
		"""

		operation = "outdata[i] = elu((float)indata[i], a)"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 v = __half22float2(indata2[i]);
				outdata2[i] = __float22half2_rn(make_float2(elu(v.x, a), elu(v.y, a)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return eluKer


def eluDer(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def eluDerKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "eluDerKer"
		arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata"), (float_t, "a")]

		preambule = """
		__forceinline__ __device__
		float eluDer(float g, float d, float a) { return g * ((d > 0.0f) + (d + a) * (d <= 0.0f)); }
		"""

		operation = "ingrad[i] = eluDer((float)outgrad[i], (float)outdata[i], a)"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 gvec = __half22float2(outgrad2[i]), dvec = __half22float2(outdata2[i]);
				ingrad2[i] = __float22half2_rn(make_float2(eluDer(gvec.x, dvec.x, a), eluDer(gvec.y, dvec.y, a)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return eluDerKer


def softPlus(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def softPlusKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "softPlusKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata")]

		preambule = "__forceinline__ __device__ float softPlus(float x) { return logf(1.0f + expf(x)); }"
		operation = "outdata[i] = softPlus((float)indata[i])"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 v = __half22float2(indata2[i]);
				outdata2[i] = __float22half2_rn(make_float2(softPlus(v.x), softPlus(v.y)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return softPlusKer


def softPlusDer(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def softPlusDerKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "softPlusDerKer"
		arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata")]

		preambule = "__forceinline__ __device__ float softPlusDer(float g, float d) { return g * (1.0f - expf(-d)); }"
		operation = "ingrad[i] = softPlusDer((float)outgrad[i], (float)outdata[i])"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 gvec = __half22float2(outgrad2[i]), dvec = __half22float2(outdata2[i]);
				ingrad2[i] = __float22half2_rn(make_float2(softPlusDer(gvec.x, dvec.x), softPlusDer(gvec.y, dvec.y)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return softPlusDerKer


def clip(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def clipKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "clipKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (float_t, "a"), (float_t, "b")]

		preambule = """
		__forceinline__ __device__ float clip(float x, float a, float b) { return min(b, max(a, x)); }
		"""

		operation = "outdata[i] = clip((float)indata[i], a, b)"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 v = __half22float2(indata2[i]);
				outdata2[i] = __float22half2_rn(make_float2(clip(v.x, a, b), clip(v.y, a, b)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return clipKer


def clipDer(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def clipDerKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "clipDerKer"
		arguments = [
			(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata"),
			(float_t, "a"), (float_t, "b")
		]

		preambule = """
		__forceinline__ __device__ float clipDer(float g, float d, float a, float b) { return g * (d > a && d < b); }
		"""

		operation = "ingrad[i] = clipDer((float)outgrad[i], (float)outdata[i], a, b)"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 gvec = __half22float2(outgrad2[i]), dvec = __half22float2(outdata2[i]);
				ingrad2[i] = __float22half2_rn(make_float2(
					clipDer(gvec.x, dvec.x, a, b), clipDer(gvec.y, dvec.y, a, b)
				));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return clipDerKer


def gelu(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def geluKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "geluKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata")]

		preambule = """
		#define M_SQRT2 1.4142135623730951f
		__forceinline__ __device__ float gelu(float x) { return 0.5f * x * (1.0f + erff(x / M_SQRT2)); }
		"""

		operation = "outdata[i] = gelu((float)indata[i])"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 v = __half22float2(indata2[i]);
				outdata2[i] = __float22half2_rn(make_float2(gelu(v.x), gelu(v.y)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return geluKer


def geluDer(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def geluDerKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "geluDerKer"
		arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "indata")]

		preambule = """
		#define M_SQRT2 1.4142135623730951f
		#define M_SQRTPI 1.7724538509055159f
		__forceinline__ __device__ float geluDer(float g, float d)
		{
			return g * (0.5f * (1.0f + erff(d / M_SQRT2)) + d / M_SQRTPI * expf(-0.5f * d * d));
		}
		"""

		operation = "ingrad[i] = geluDer((float)outgrad[i], (float)indata[i])"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 gvec = __half22float2(outgrad2[i]), dvec = __half22float2(indata2[i]);
				ingrad2[i] = __float22half2_rn(make_float2(geluDer(gvec.x, dvec.x), geluDer(gvec.y, dvec.y)));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return geluDerKer


def dropout(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def dropoutKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		parttype = {
			np.float32: uint_t,
			np.float16: ushort_t
		}[dtype.type]

		name = "dropoutKer"
		arguments = [
			(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (parttype.const.ptr, "b"),
			(parttype, "v"), (float_t, "p")
		]

		preambule = """
		__forceinline__ __device__
		float dropout(float x, %s b, %s v, float p) { return x * (b < v) / p; }
		""" % (parttype, parttype)

		operation = "outdata[i] = dropout((float)indata[i], b[i], v, p)"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 dvec = __half22float2(indata2[i]);
				ushort2 bvec = b2[i];
	
				outdata2[i] = __float22half2_rn(make_float2(
					dropout(dvec.x, bvec.x, v, p), dropout(dvec.y, bvec.y, v, p)
				));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return dropoutKer


def dropout2d(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def dropout2dKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		parttype = {
			np.float32: uint_t,
			np.float16: ushort_t
		}[dtype.type]

		name = "dropout2dKer"
		arguments = [
			(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (parttype.const.ptr, "b"), (parttype, "v"),
			(float_t, "p"), (int_t, "mapsize")
		]

		preambule = """
		__forceinline__ __device__
		float dropout(float x, %s b, %s v, float p) { return x * (b < v) / p; }
		""" % (parttype, parttype)

		operation = "outdata[i] = dropout((float)indata[i], b[i / mapsize], v, p)"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 dvec = __half22float2(indata2[i]);
	
				outdata2[i] = __float22half2_rn(make_float2(
					dropout(dvec.x, b[2 * i / mapsize], v, p), dropout(dvec.y, b[(2 * i + 1) / mapsize], v, p)
				));
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return dropout2dKer


def toVectorAddVector(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def toVectorAddVectorKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "toVectorAddVectorKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (float_t, "alpha")]

		operation = "outdata[i] = (float)outdata[i] + (float)indata[i] * alpha"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 outv = __half22float2(outdata2[i]), inv = __half22float2(indata2[i]);
				outdata2[i] = __float22half2_rn(make_float2(outv.x + inv.x * alpha, outv.y + inv.y * alpha));
				""",
				operation, name
			)

		else:
			return ElementwiseKernel(arguments, operation, name)

	return toVectorAddVectorKer


def adadelta(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def adadeltaKer(dtype):
		assert dtype.type in {np.float32, np.float16}

		ctype = dtypeToCtype[dtype.type]
		name = "adadeltaKer"

		arguments = [
			(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "msg"), (ctype.ptr, "msdx"),
			(float_t, "rho"), (float_t, "epsilon")
		]

		preambule = """
		__forceinline__ __device__ void adadelta(float *param, float g, float *msg, float *msdx, float rho, float eps)
		{
			*msg += (1.0f - rho) * (g * g - *msg);
			float dx = sqrt((*msdx + eps) / (*msg + eps)) * g;

			*msdx += (1.0f - rho) * (dx * dx - *msdx);
			*param += dx;
		}
		"""

		operation = """
		float paramval = (float)param[i], msgval = (float)msg[i], msdxval = (float)msdx[i];
		adadelta(&paramval, (float)grad[i], &msgval, &msdxval, rho, epsilon);

		param[i] = paramval, msg[i] = msgval, msdx[i] = msdxval;
		"""

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 pvec = __half22float2(param2[i]), gvec = __half22float2(grad2[i]); 
				float2 msgvec = __half22float2(msg2[i]), msdxvec = __half22float2(msdx2[i]);

				adadelta(&pvec.x, gvec.x, &msgvec.x, &msdxvec.x, rho, epsilon);
				adadelta(&pvec.y, gvec.y, &msgvec.y, &msdxvec.y, rho, epsilon);

				param2[i] = __float22half2_rn(pvec);
				msg2[i] = __float22half2_rn(msgvec), msdx2[i] = __float22half2_rn(msdxvec);
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return adadeltaKer


def adagrad(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def adagradKer(dtype):
		assert dtype.type in {np.float32, np.float16}

		ctype = dtypeToCtype[dtype.type]
		name = "adagradKer"

		arguments = [
			(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "h"),
			(float_t, "learnRate"), (float_t, "epsilon")
		]

		preambule = """
		__forceinline__ __device__ void adagrad(float *param, float g, float *h, float learnRate, float epsilon)
		{
			*h += g * g;
			*param += learnRate * g / (sqrtf(*h) + epsilon);
		}
		"""

		operation = """
		float paramval = (float)param[i], hval = (float)h[i];
		adagrad(&paramval, (float)grad[i], &hval, learnRate, epsilon);

		param[i] = paramval, h[i] = hval;
		"""

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 pvec = __half22float2(param2[i]), gvec = __half22float2(grad2[i]), hvec = __half22float2(h2[i]);

				adagrad(&pvec.x, gvec.x, &hvec.x, learnRate, epsilon);
				adagrad(&pvec.y, gvec.y, &hvec.y, learnRate, epsilon);

				param2[i] = __float22half2_rn(pvec), h2[i] = __float22half2_rn(hvec);
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return adagradKer


def adam(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def adamKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "adamKer"
		arguments = [
			(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (float_t.ptr, "mg"), (float_t.ptr, "ms"),
			(float_t, "learnRate"), (float_t, "fix1"), (float_t, "fix2"), (float_t, "epsilon")
		]

		preambule = """
		__forceinline__ __device__ void adam(float *param, float g, float *mg, float *ms,
											 float learnRate, float fix1, float fix2, float epsilon)
		{
			*mg += fix1 * (g - *mg);
			*ms += fix2 * (g * g - *ms);

			*param += learnRate * *mg / (sqrtf(*ms) + epsilon);
		}
		"""

		operation = """
		float paramval = (float)param[i], mgval = (float)mg[i], msval = (float)ms[i];
		adam(&paramval, (float)grad[i], &mgval, &msval, learnRate, fix1, fix2, epsilon);

		param[i] = paramval, mg[i] = mgval, ms[i] = msval;
		"""

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 pvec = __half22float2(param2[i]), gvec = __half22float2(grad2[i]);

				adam(&pvec.x, gvec.x, &mg[2 * i], &ms[2 * i], learnRate, fix1, fix2, epsilon);
				adam(&pvec.y, gvec.y, &mg[2 * i + 1], &ms[2 * i + 1], learnRate, fix1, fix2, epsilon);

				param2[i] = __float22half2_rn(pvec);
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return adamKer


def classicMomSGD(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def classicMomSGDKer(dtype):
		assert dtype.type in {np.float32, np.float16}

		ctype = dtypeToCtype[dtype.type]
		name = "classicMomSGDKer"

		arguments = [
			(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "mom"),
			(float_t, "learnRate"), (float_t, "momRate")
		]

		preambule = """
		__forceinline__ __device__ void classicMomSGD(float *param, float g, float *mom, float learnRate, float momRate)
		{
			*mom = momRate * *mom + learnRate * g;
			*param += *mom;
		}
		"""

		operation = """
		float paramval = (float)param[i], momval = (float)mom[i];
		classicMomSGD(&paramval, (float)grad[i], &momval, learnRate, momRate);

		param[i] = paramval, mom[i] = momval;
		"""

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 pvec = __half22float2(param2[i]), gvec = __half22float2(grad2[i]);
				float2 mvec = __half22float2(mom2[i]);

				classicMomSGD(&pvec.x, gvec.x, &mvec.x, learnRate, momRate);
				classicMomSGD(&pvec.y, gvec.y, &mvec.y, learnRate, momRate);

				param2[i] = __float22half2_rn(pvec), mom2[i] = __float22half2_rn(mvec);
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return classicMomSGDKer


def nesterovMomSGD(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def nesterovMomSGDKer(dtype):
		assert dtype.type in {np.float32, np.float16}

		ctype = dtypeToCtype[dtype.type]
		name = "nesterovMomSGDKer"

		arguments = [
			(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "mom"),
			(float_t, "learnRate"), (float_t, "momRate")
		]

		preambule = """
		__forceinline__ __device__ void nesterovMomSGD(float *param, float g, float *m, float learnRate, float momRate)
		{
			*param += momRate * momRate * *m + (1.0f + momRate) * learnRate * g;
			*m = momRate * *m + learnRate * g;
		}
		"""

		operation = """
		float paramval = (float)param[i], momval = (float)mom[i];
		nesterovMomSGD(&paramval, (float)grad[i], &momval, learnRate, momRate);

		param[i] = paramval, mom[i] = momval;
		"""

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 pvec = __half22float2(param2[i]), gvec = __half22float2(grad2[i]);
				float2 mvec = __half22float2(mom2[i]);

				nesterovMomSGD(&pvec.x, gvec.x, &mvec.x, learnRate, momRate);
				nesterovMomSGD(&pvec.y, gvec.y, &mvec.y, learnRate, momRate);

				param2[i] = __float22half2_rn(pvec), mom2[i] = __float22half2_rn(mvec);
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return nesterovMomSGDKer


def rmsprop(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def rmspropKer(dtype):
		assert dtype.type in {np.float32, np.float16}

		ctype = dtypeToCtype[dtype.type]
		name = "rmspropKer"

		arguments = [
			(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "ms"),
			(float_t, "learnRate"), (float_t, "factor"), (float_t, "epsilon")
		]

		preambule = """
		__forceinline__ __device__ void rmsprop(float *param, float g, float *ms,
												float learnRate, float factor, float epsilon)
		{
			*ms = factor * *ms + (1.0f - factor) * g * g;
			*param += learnRate * g / (sqrtf(*ms) + epsilon);
		}
		"""

		operation = """
		float paramval = (float)param[i], msval = (float)ms[i];
		rmsprop(&paramval, (float)grad[i], &msval, learnRate, factor, epsilon);

		param[i] = paramval, ms[i] = msval;
		"""

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 pvec = __half22float2(param2[i]), gvec = __half22float2(grad2[i]);
				float2 msvec = __half22float2(ms2[i]);
	
				rmsprop(&pvec.x, gvec.x, &msvec.x, learnRate, factor, epsilon);
				rmsprop(&pvec.y, gvec.y, &msvec.y, learnRate, factor, epsilon);
	
				param2[i] = __float22half2_rn(pvec), ms2[i] = __float22half2_rn(msvec);
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return rmspropKer


def rmspropGraves(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def rmspropGravesKer(dtype):
		assert dtype.type in {np.float32, np.float16}

		ctype = dtypeToCtype[dtype.type]
		name = "rmspropGravesKer"

		arguments = [
			(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "mg"), (ctype.ptr, "ms"), (ctype.ptr, "delta"),
			(float_t, "learnRate"), (float_t, "alpha"), (float_t, "momRate"), (float_t, "epsilon")
		]

		preambule = """
		__forceinline__ __device__ void rmspropGraves(float *param, float g, float *mg, float *ms, float *delta,
													  float learnRate, float alpha, float momRate, float epsilon)
		{
			*mg = alpha * *mg + (1.0f - alpha) * g;
			*ms = alpha * *ms + (1.0f - alpha) * g * g;
			*delta = momRate * *delta + learnRate * g / sqrtf(*ms - *mg * *mg + epsilon);

			*param += *delta;
		}
		"""

		operation = """
		float paramval = (float)param[i], mgval = (float)mg[i], msval = (float)ms[i], deltaval = (float)delta[i];
		rmspropGraves(&paramval, (float)grad[i], &mgval, &msval, &deltaval, learnRate, alpha, momRate, epsilon);

		param[i] = paramval, mg[i] = mgval, ms[i] = msval, delta[i] = deltaval;
		"""

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 pvec = __half22float2(param2[i]), gvec = __half22float2(grad2[i]); 
				float2 mgvec = __half22float2(mg2[i]), msvec = __half22float2(ms2[i]), dvec = __half22float2(delta2[i]);

				rmspropGraves(&pvec.x, gvec.x, &mgvec.x, &msvec.x, &dvec.x, learnRate, alpha, momRate, epsilon);
				rmspropGraves(&pvec.y, gvec.y, &mgvec.y, &msvec.y, &dvec.y, learnRate, alpha, momRate, epsilon);

				param2[i] = __float22half2_rn(pvec), delta2[i] = __float22half2_rn(dvec);
				mg2[i] = __float22half2_rn(mgvec), ms2[i] = __float22half2_rn(msvec);
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return rmspropGravesKer


def smorms3(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def smorms3Ker(dtype):
		assert dtype.type in {np.float32, np.float16}

		ctype = dtypeToCtype[dtype.type]
		name = "smorms3Ker"

		arguments = [
			(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (float_t.ptr, "mem"),
			(float_t.ptr, "mg"), (float_t.ptr, "ms"), (float_t, "learnRate"), (float_t, "epsilon")
		]

		preambule = """
		__forceinline__ __device__ void smorms3(float *param, float g, float *mem, float *mg, float *ms,
												float learnRate, float epsilon)
		{
			float r = 1.0f / (*mem + 1.0f);
	
			*mg = (1.0f - r) * *mg + r * g;
			*ms = (1.0f - r) * *ms + r * g * g;
			float x = *mg * *mg / (*ms + epsilon);

			*mem = 1.0f + *mem * (1.0f - x);
			*param += g * min(learnRate, x) / (sqrtf(*ms) + epsilon);
		}
		"""

		operation = """
		float paramval = (float)param[i], memval = (float)mem[i], mgval = (float)mg[i], msval = (float)ms[i];
		smorms3(&paramval, (float)grad[i], &memval, &mgval, &msval, learnRate, epsilon);

		param[i] = paramval, mem[i] = memval, mg[i] = mgval, ms[i] = msval;
		"""

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 pvec = __half22float2(param2[i]), gvec = __half22float2(grad2[i]);

				smorms3(&pvec.x, gvec.x, &mem[2 * i], &mg[2 * i], &ms[2 * i], learnRate, epsilon);
				smorms3(&pvec.y, gvec.y, &mem[2 * i + 1], &mg[2 * i + 1], &ms[2 * i + 1], learnRate, epsilon);

				param2[i] = __float22half2_rn(pvec);
				""",
				operation, name, preambule=preambule
			)

		else:
			return ElementwiseKernel(arguments, operation, name, preambule=preambule)

	return smorms3Ker


def add(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def addKer(dtype):
		assert dtype.type in {np.float32, np.float16}

		ctype = dtypeToCtype[dtype.type]
		name = "addKer"

		arguments = [
			(ctype.ptr, "outdata"), (ctype.const.ptr, "x"), (float_t, "alpha"),
			(ctype.const.ptr, "y"), (float_t, "beta")
		]

		operation = "outdata[i] = (float)x[i] * alpha + (float)y[i] * beta"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 xv = __half22float2(x2[i]), yv = __half22float2(y2[i]);
				outdata2[i] = __float22half2_rn(make_float2(xv.x * alpha + yv.x * beta, xv.y * alpha + yv.y * beta));
				""",
				operation, name
			)

		else:
			return ElementwiseKernel(arguments, operation, name)

	return addKer


def mul(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def mulKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "mulKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "a"), (ctype.const.ptr, "b")]

		operation = "outdata[i] = (float)a[i] * (float)b[i]"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 av = __half22float2(a2[i]), bv = __half22float2(b2[i]);
				outdata2[i] = __float22half2_rn(make_float2(av.x * bv.x, av.y * bv.y));
				""",
				operation, name
			)

		else:
			return ElementwiseKernel(arguments, operation, name)

	return mulKer


def linear(ElementwiseKernel, ElementHalf2Kernel):
	@memoize
	def linearKer(dtype):
		assert dtype.type in {np.float32, np.float16}
		ctype = dtypeToCtype[dtype.type]

		name = "linearKer"
		arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (float_t, "a"), (float_t, "b")]

		operation = "outdata[i] = a * (float)indata[i] + b"

		if dtype == np.float16:
			return ElementHalf2Kernel(
				arguments,
				"""
				float2 v = __half22float2(indata2[i]);
				outdata2[i] = __float22half2_rn(make_float2(a * v.x + b, a * v.y + b));
				""",
				operation, name
			)

		else:
			return ElementwiseKernel(arguments, operation, name)

	return linearKer


def rbmKer(ElementwiseKernel):
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata"), (float_t.const.ptr, "uni")],
		"float p = 1.0f / (1.0f + expf(-indata[i]));"
		"outdata[i] = (uni[i] < p)",
		"rbmKer"
	)


def weightDecayKer(ElementwiseKernel):
	return ElementwiseKernel(
		[(float_t.ptr, "grad"), (float_t.const.ptr, "param"), (float_t, "rate")],
		"grad[i] -= rate * param[i]",
		"weightDecayKer"
	)


def absKer(ElementwiseKernel):
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata")],
		"outdata[i] = fabsf(indata[i])",
		"absKer"
	)


def l1penaltyKer(ElementwiseKernel):
	return ElementwiseKernel(
		[(float_t.ptr, "outgrad"), (float_t.const.ptr, "ingrad"), (float_t.ptr, "data"), (float_t, "a")],
		"outgrad[i] = ingrad[i] - a * ((0.0f <= data[i]) - (data[i] < 0.0f))",
		"l1penaltyKer"
	)


def l1gradKer(ElementwiseKernel):
	return ElementwiseKernel(
		[(float_t.ptr, "grad"), (float_t.const.ptr, "pred"), (float_t.const.ptr, "target"), (float_t, "norm")],
		"grad[i] = (pred[i] - target[i] > 0.0f ? -norm : norm)",
		"l1gradKer"
	)


def castFP16toFP32(ElementwiseKernel):
	return ElementwiseKernel(
		[(float_t.ptr, "outdata"), (half_t.const.ptr, "indata")],
		"outdata[i] = indata[i]",
		"castFP16toFP32", preambule="#include <cuda_fp16.h>"
	)


def castFP32toFP16(ElementwiseKernel):
	return ElementwiseKernel(
		[(half_t.ptr, "outdata"), (float_t.const.ptr, "indata")],
		"outdata[i] = indata[i]",
		"castFP32toFP16", preambule="#include <cuda_fp16.h>"
	)
