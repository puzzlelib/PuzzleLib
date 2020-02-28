import numpy as np

from PuzzleLib.Compiler.Codegen.Types import ushort_t, int_t, uint_t, half_t, float_t

from PuzzleLib.Cuda.GPUArray import memoizeOnCtx
from PuzzleLib.Cuda.SourceModule import ElementwiseKernel, ElementHalf2Kernel, dtypeToCtype


@memoizeOnCtx
def sigmoidKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "sigmoidKer"
	arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata")]

	operation = "outdata[i] = 1.0f / (1.0f + expf(-(float)indata[i]))"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 vec = __half22float2(indata2[i]);
			outdata2[i] = __float22half2_rn(make_float2(1.0f / (1.0f + expf(-vec.x)), 1.0f / (1.0f + expf(-vec.y))));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def sigmoidDerKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "sigmoidDerKer"
	arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata")]

	operation = "ingrad[i] = (float)outgrad[i] * (float)outdata[i] * (1.0f - (float)outdata[i])"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(outgrad2[i]), datavec = __half22float2(outdata2[i]);

			ingrad2[i] = __float22half2_rn(make_float2(
				gradvec.x * datavec.x * (1.0f - datavec.x),
				gradvec.y * datavec.y * (1.0f - datavec.y)
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
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
			float2 vec = __half22float2(indata2[i]);
			outdata2[i] = __float22half2_rn(make_float2(tanhf(vec.x), tanhf(vec.y)));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def tanhDerKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "tanhDerKer"
	arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata")]

	operation = "ingrad[i] = (float)outgrad[i] * (1.0f - (float)outdata[i] * (float)outdata[i])"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(outgrad2[i]), datavec = __half22float2(outdata2[i]);

			ingrad2[i] = __float22half2_rn(make_float2(
				gradvec.x * (1.0f - datavec.x * datavec.x),
				gradvec.y * (1.0f - datavec.y * datavec.y)
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def reluKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "reluKer"
	arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata")]

	operation = "outdata[i] = (float)indata[i] * ((float)indata[i] > 0.0f)"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 datavec = __half22float2(indata2[i]);

			outdata2[i] = __float22half2_rn(make_float2(
				datavec.x * (datavec.x > 0.0f),
				datavec.y * (datavec.y > 0.0f)
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def reluDerKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "reluDerKer"
	arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata")]

	operation = "ingrad[i] = (float)outgrad[i] * ((float)outdata[i] > 0.0f)"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(outgrad2[i]), datavec = __half22float2(outdata2[i]);

			ingrad2[i] = __float22half2_rn(make_float2(
				gradvec.x * (datavec.x > 0.0f),
				gradvec.y * (datavec.y > 0.0f)
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def leakyReluKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "leakyReluKer"
	arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (float_t, "a")]

	operation = "outdata[i] = (float)indata[i] * (((float)indata[i] > 0.0f) + a * ((float)indata[i] <= 0.0f))"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 datavec = __half22float2(indata2[i]);

			outdata2[i] = __float22half2_rn(make_float2(
				datavec.x * ((datavec.x > 0.0f) + a * (datavec.x <= 0.0f)),
				datavec.y * ((datavec.y > 0.0f) + a * (datavec.y <= 0.0f))
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def leakyReluDerKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "leakyReluDerKer"
	arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata"), (float_t, "a")]

	operation = "ingrad[i] = (float)outgrad[i] * (((float)outdata[i] > 0.0f) + a * ((float)outdata[i] <= 0.0f))"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(outgrad2[i]), datavec = __half22float2(outdata2[i]);

			ingrad2[i] = __float22half2_rn(make_float2(
				gradvec.x * ((datavec.x > 0.0f) + a * (datavec.x <= 0.0f)),
				gradvec.y * ((datavec.y > 0.0f) + a * (datavec.y <= 0.0f))
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def eluKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "eluKer"
	arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (float_t, "a")]

	operation = """
	float dataval = (float)indata[i];
	outdata[i] = dataval * (dataval > 0.0f) + a * (expf(dataval) - 1.0f) * (dataval <= 0.0f);
	"""

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 datavec = __half22float2(indata2[i]);

			outdata2[i] = __float22half2_rn(make_float2(
				datavec.x * (datavec.x > 0.0f) + a * (expf(datavec.x) - 1.0f) * (datavec.x <= 0.0f),
				datavec.y * (datavec.y > 0.0f) + a * (expf(datavec.y) - 1.0f) * (datavec.y <= 0.0f)
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def eluDerKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "eluDerKer"
	arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata"), (float_t, "a")]

	operation = """
	float dataval = outdata[i];
	ingrad[i] = (float)outgrad[i] * ((dataval > 0.0f) + (dataval + a) * (dataval <= 0.0f));
	"""

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(outgrad2[i]), datavec = __half22float2(outdata2[i]);

			ingrad2[i] = __float22half2_rn(make_float2(
				gradvec.x * ((datavec.x > 0.0f) + (datavec.x + a) * (datavec.x <= 0.0f)),
				gradvec.y * ((datavec.y > 0.0f) + (datavec.y + a) * (datavec.y <= 0.0f))
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def softPlusKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "softPlusKer"
	arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata")]

	operation = "outdata[i] = logf(1.0f + expf((float)indata[i]))"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 datavec = __half22float2(indata2[i]);
			outdata2[i] = __float22half2_rn(make_float2(logf(1.0f + expf(datavec.x)), logf(1.0f + expf(datavec.y))));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def softPlusDerKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "softPlusDerKer"
	arguments = [(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata")]

	operation = "ingrad[i] = (float)outgrad[i] * (1.0f - expf(-(float)outdata[i]))"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(outgrad2[i]), datavec = __half22float2(outdata2[i]);

			ingrad2[i] = __float22half2_rn(make_float2(
				gradvec.x * (1.0f - expf(-datavec.x)),
				gradvec.y * (1.0f - expf(-datavec.y))
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def clipKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "clipKer"
	arguments = [(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (float_t, "a"), (float_t, "b")]

	operation = "outdata[i] = min(b, max(a, indata[i]))"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 datavec = __half22float2(indata2[i]);
			outdata2[i] = __float22half2_rn(make_float2(min(b, max(a, datavec.x)), min(b, max(a, datavec.y))));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def clipDerKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "clipDerKer"
	arguments = [
		(ctype.ptr, "ingrad"), (ctype.const.ptr, "outgrad"), (ctype.const.ptr, "outdata"),
		(float_t, "a"), (float_t, "b")
	]

	operation = "ingrad[i] = (float)outgrad[i] * ((float)outdata[i] > a && (float)outdata[i] < b)"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(outgrad2[i]), datavec = __half22float2(outdata2[i]);

			ingrad2[i] = __float22half2_rn(make_float2(
				gradvec.x * (datavec.x > a && datavec.x < b),
				gradvec.y * (datavec.y > a && datavec.y < b)
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def dropoutKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	parttype = {
		np.float32: uint_t,
		np.float16: ushort_t
	}[dtype.type]

	name = "dropoutKer"
	arguments = [
		(ctype.ptr, "outdata"), (ctype.const.ptr, "indata"), (parttype.const.ptr, "b"), (parttype, "v"), (float_t, "p")
	]

	operation = "outdata[i] = (float)indata[i] * (b[i] < v) / p"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 datavec = __half22float2(indata2[i]);
			ushort2 bvec = b2[i];

			outdata2[i] = __float22half2_rn(make_float2(datavec.x * (bvec.x < v) / p, datavec.y * (bvec.y < v) / p));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
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

	operation = "outdata[i] = (float)indata[i] * (b[i / mapsize] < v) / p"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 datavec = __half22float2(indata2[i]);

			outdata2[i] = __float22half2_rn(make_float2(
				datavec.x * (b[2 * i / mapsize] < v) / p,
				datavec.y * (b[(2 * i + 1) / mapsize] < v) / p
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
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
			float2 outvec = __half22float2(outdata2[i]), invec = __half22float2(indata2[i]);
			outdata2[i] = __float22half2_rn(make_float2(outvec.x + invec.x * alpha, outvec.y + invec.y * alpha));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def adadeltaKer(dtype):
	assert dtype.type in {np.float32, np.float16}

	ctype = dtypeToCtype[dtype.type]
	name = "adadeltaKer"

	arguments = [
		(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "msg"), (ctype.ptr, "msdx"),
		(float_t, "rho"), (float_t, "epsilon")
	]

	operation = """
	float msgval = (float)msg[i] + (1.0f - rho) * ((float)grad[i] * (float)grad[i] - (float)msg[i]);
	msg[i] = msgval;

	float dx = sqrt(((float)msdx[i] + epsilon) / (msgval + epsilon)) * (float)grad[i];
	msdx[i] = (float)msdx[i] + (1.0f - rho) * (dx * dx - (float)msdx[i]);
	param[i] = (float)param[i] + dx;
	"""

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(grad2[i]), msgvec = __half22float2(msg2[i]);
			float2 paramvec = __half22float2(param2[i]), msdxvec = __half22float2(msdx2[i]);

			msgvec = make_float2(
				msgvec.x + (1.0f - rho) * (gradvec.x * gradvec.x - msgvec.x),
				msgvec.y + (1.0f - rho) * (gradvec.y * gradvec.y - msgvec.y)
			);
			msg2[i] = __float22half2_rn(msgvec);

			float dxx = sqrt((msdxvec.x + epsilon) / (msgvec.x + epsilon)) * gradvec.x;
			float dxy = sqrt((msdxvec.y + epsilon) / (msgvec.y + epsilon)) * gradvec.y;

			msdx2[i] = __float22half2_rn(make_float2(
				msdxvec.x + (1.0f - rho) * (dxx * dxx - msdxvec.x),
				msdxvec.y + (1.0f - rho) * (dxy * dxy - msdxvec.y)
			));

			param2[i] = __float22half2_rn(make_float2(paramvec.x + dxx, paramvec.y + dxy));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def adagradKer(dtype):
	assert dtype.type in {np.float32, np.float16}

	ctype = dtypeToCtype[dtype.type]
	name = "adagradKer"

	arguments = [
		(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "h"), (float_t, "learnRate"), (float_t, "epsilon")
	]

	operation = """
	float hval = (float)h[i] + (float)grad[i] * (float)grad[i];

	h[i] = hval;
	param[i] = (float)param[i] + learnRate * (float)grad[i] / (sqrtf(hval) + epsilon);
	"""

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(grad2[i]), hvec = __half22float2(h2[i]);
			float2 paramvec = __half22float2(param2[i]);

			hvec = make_float2(hvec.x + gradvec.x * gradvec.x, hvec.y + gradvec.y * gradvec.y);
			h2[i] = __float22half2_rn(hvec);

			param2[i] = __float22half2_rn(make_float2(
				paramvec.x + learnRate * gradvec.x / (sqrtf(hvec.x) + epsilon),
				paramvec.y + learnRate * gradvec.y / (sqrtf(hvec.y) + epsilon)
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def adamKer(dtype):
	assert dtype.type in {np.float32, np.float16}
	ctype = dtypeToCtype[dtype.type]

	name = "adamKer"
	arguments = [
		(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (float_t.ptr, "mg"), (float_t.ptr, "ms"),
		(float_t, "learnRate"), (float_t, "fix1"), (float_t, "fix2"), (float_t, "epsilon")
	]

	operation = """
	mg[i] = (float)mg[i] + fix1 * ((float)grad[i] - mg[i]);
	ms[i] = (float)ms[i] + fix2 * ((float)grad[i] * (float)grad[i] - ms[i]);

	param[i] = (float)param[i] + learnRate * mg[i] / (sqrtf(ms[i]) + epsilon);
	"""

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 paramvec = __half22float2(param2[i]), gradvec = __half22float2(grad2[i]);

			mg[2 * i] += fix1 * (gradvec.x - mg[2 * i]);
			mg[2 * i + 1] += fix1 * (gradvec.y - mg[2 * i + 1]);

			ms[2 * i] += fix2 * (gradvec.x * gradvec.x - ms[2 * i]);
			ms[2 * i + 1] += fix2 * (gradvec.y * gradvec.y - ms[2 * i + 1]);

			param2[i] = __float22half2_rn(make_float2(
				paramvec.x + learnRate * mg[2 * i] / (sqrtf(ms[2 * i]) + epsilon),
				paramvec.y + learnRate * mg[2 * i + 1] / (sqrtf(ms[2 * i + 1]) + epsilon)
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def classicMomSGDKer(dtype):
	assert dtype.type in {np.float32, np.float16}

	ctype = dtypeToCtype[dtype.type]
	name = "classicMomSGDKer"

	arguments = [
		(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "mom"),
		(float_t, "learnRate"), (float_t, "momRate")
	]

	operation = """
	float m = momRate * (float)mom[i] + learnRate * (float)grad[i];
	mom[i] = m;
	param[i] = (float)param[i] + m;
	"""

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(grad2[i]), momvec = __half22float2(mom2[i]);
			float2 paramvec = __half22float2(param2[i]);

			momvec = make_float2(
				momRate * momvec.x + learnRate * gradvec.x,
				momRate * momvec.y + learnRate * gradvec.y
			);

			mom2[i] = __float22half2_rn(momvec);
			param2[i] = __float22half2_rn(make_float2(paramvec.x + momvec.x, paramvec.y + momvec.y));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def nesterovMomSGDKer(dtype):
	assert dtype.type in {np.float32, np.float16}

	ctype = dtypeToCtype[dtype.type]
	name = "nesterovMomSGDKer"

	arguments = [
		(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "mom"),
		(float_t, "learnRate"), (float_t, "momRate")
	]

	operation = """
	float momval = mom[i], gradval = grad[i];
	mom[i] = momRate * momval + learnRate * gradval;
	param[i] = (float)param[i] + momRate * momRate * momval + (1.0f + momRate) * learnRate * gradval;
	"""

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 gradvec = __half22float2(grad2[i]), momvec = __half22float2(mom2[i]);
			float2 paramvec = __half22float2(param2[i]);

			mom2[i] = __float22half2_rn(make_float2(
				momRate * momvec.x + learnRate * gradvec.x,
				momRate * momvec.y + learnRate * gradvec.y
			));
			param2[i] = __float22half2_rn(make_float2(
				paramvec.x + momRate * momRate * momvec.x + (1.0f + momRate) * learnRate * gradvec.x,
				paramvec.y + momRate * momRate * momvec.y + (1.0f + momRate) * learnRate * gradvec.y
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def rmspropKer(dtype):
	assert dtype.type in {np.float32, np.float16}

	ctype = dtypeToCtype[dtype.type]
	name = "rmspropKer"

	arguments = [
		(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "ms"),
		(float_t, "learnRate"), (float_t, "factor"), (float_t, "epsilon")
	]

	operation = """
	float msval = factor * (float)ms[i] + (1.0f - factor) * (float)grad[i] * (float)grad[i];
	ms[i] = msval;
	param[i] = (float)param[i] + learnRate * (float)grad[i] / (sqrtf(msval) + epsilon);
	"""

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 msvec = __half22float2(ms2[i]), gradvec = __half22float2(grad2[i]);
			float2 paramvec = __half22float2(param2[i]);

			msvec = make_float2(
				factor * msvec.x + (1.0f - factor) * gradvec.x * gradvec.x,
				factor * msvec.y + (1.0f - factor) * gradvec.y * gradvec.y
			);
			ms2[i] = __float22half2_rn(msvec);

			param2[i] = __float22half2_rn(make_float2(
				paramvec.x + learnRate * gradvec.x / (sqrtf(msvec.x) + epsilon),
				paramvec.y + learnRate * gradvec.y / (sqrtf(msvec.y) + epsilon)
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def rmspropGravesKer(dtype):
	assert dtype.type in {np.float32, np.float16}

	ctype = dtypeToCtype[dtype.type]
	name = "rmspropGravesKer"

	arguments = [
		(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (ctype.ptr, "mg"), (ctype.ptr, "ms"),
		(ctype.ptr, "delta"), (float_t, "learnRate"), (float_t, "alpha"), (float_t, "momRate"), (float_t, "epsilon")
	]

	operation = """
	float mgv = alpha * (float)mg[i] + (1.0f - alpha) * (float)grad[i];
	float msv = alpha * (float)ms[i] + (1.0f - alpha) * (float)grad[i] * (float)grad[i];
	float deltav = momRate * (float)delta[i] + learnRate * (float)grad[i] / sqrtf(msv - mgv * mgv + epsilon);

	mg[i] = mgv, ms[i] = msv, delta[i] = deltav;
	param[i] = (float)param[i] + deltav;
	"""

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 mgvec = __half22float2(mg2[i]), msvec = __half22float2(ms2[i]), gradvec = __half22float2(grad2[i]);
			float2 deltavec = __half22float2(delta2[i]), paramvec = __half22float2(param2[i]); 

			mgvec = make_float2(
				alpha * mgvec.x + (1.0f - alpha) * gradvec.x,
				alpha * mgvec.y + (1.0f - alpha) * gradvec.y
			);
			msvec = make_float2(
				alpha * msvec.x + (1.0f - alpha) * gradvec.x * gradvec.x,
				alpha * msvec.y + (1.0f - alpha) * gradvec.y * gradvec.y
			);
			deltavec = make_float2(
				momRate * deltavec.x + learnRate * gradvec.x / sqrtf(msvec.x - mgvec.x * mgvec.x + epsilon),
				momRate * deltavec.y + learnRate * gradvec.y / sqrtf(msvec.y - mgvec.y * mgvec.y + epsilon)
			);

			mg2[i] = __float22half2_rn(mgvec), ms2[i] = __float22half2_rn(msvec);
			delta2[i] = __float22half2_rn(deltavec);

			param2[i] = __float22half2_rn(make_float2(paramvec.x + deltavec.x, paramvec.y + deltavec.y));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def smorms3Ker(dtype):
	assert dtype.type in {np.float32, np.float16}

	ctype = dtypeToCtype[dtype.type]
	name = "smorms3Ker"

	arguments = [
		(ctype.ptr, "param"), (ctype.const.ptr, "grad"), (float_t.ptr, "mem"),
		(float_t.ptr, "mg"), (float_t.ptr, "ms"), (float_t, "learnRate"), (float_t, "epsilon")
	]

	operation = """
	float r = 1.0f / (mem[i] + 1.0f);

	float mgi = (1.0f - r) * mg[i] + r * (float)grad[i];
	float msi = (1.0f - r) * ms[i] + r * (float)grad[i] * (float)grad[i];
	float x = mgi * mgi / (msi + epsilon);

	mem[i] = 1.0f + mem[i] * (1.0f - x), mg[i] = mgi, ms[i] = msi;
	param[i] = (float)param[i] + (float)grad[i] * min(learnRate, x) / (sqrtf(msi) + epsilon);
	"""

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 paramvec = __half22float2(param2[i]), gradvec = __half22float2(grad2[i]);
			float2 rvec = make_float2(1.0f / (mem[2 * i] + 1.0f), 1.0f / (mem[2 * i + 1] + 1.0f));

			float2 mgvec = make_float2(
				(1.0f - rvec.x) * mg[2 * i] + rvec.x * gradvec.x,
				(1.0f - rvec.y) * mg[2 * i + 1] + rvec.y * gradvec.y
			);
			float2 msvec = make_float2(
				(1.0f - rvec.x) * ms[2 * i] + rvec.x * gradvec.x * gradvec.x,
				(1.0f - rvec.y) * ms[2 * i + 1] + rvec.y * gradvec.y * gradvec.y
			);

			float2 xvec = make_float2(mgvec.x * mgvec.x / (msvec.x + epsilon), mgvec.y * mgvec.y / (msvec.y + epsilon));

			mem[2 * i] = 1.0f + mem[2 * i] * (1.0f - xvec.x);
			mem[2 * i + 1] = 1.0f + mem[2 * i + 1] * (1.0f - xvec.y);

			mg[2 * i] = mgvec.x, mg[2 * i + 1] = mgvec.y, ms[2 * i] = msvec.x, ms[2 * i + 1] = msvec.y;

			param2[i] = __float22half2_rn(make_float2(
				paramvec.x + gradvec.x * min(learnRate, xvec.x) / (sqrtf(msvec.x) + epsilon),
				paramvec.y + gradvec.y * min(learnRate, xvec.y) / (sqrtf(msvec.y) + epsilon)
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
def addKer(dtype):
	assert dtype.type in {np.float32, np.float16}

	ctype = dtypeToCtype[dtype.type]
	name = "addKer"

	arguments = [
		(ctype.ptr, "outdata"), (ctype.const.ptr, "x"), (float_t, "alpha"), (ctype.const.ptr, "y"), (float_t, "beta")
	]

	operation = "outdata[i] = (float)x[i] * alpha + (float)y[i] * beta"

	if dtype == np.float16:
		return ElementHalf2Kernel(
			arguments,
			"""
			float2 xvec = __half22float2(x2[i]), yvec = __half22float2(y2[i]);

			outdata2[i] = __float22half2_rn(make_float2(
				xvec.x * alpha + yvec.x * beta,
				xvec.y * alpha + yvec.y * beta
			));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
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
			float2 avec = __half22float2(a2[i]), bvec = __half22float2(b2[i]);
			outdata2[i] = __float22half2_rn(make_float2(avec.x * bvec.x, avec.y * bvec.y));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


@memoizeOnCtx
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
			float2 invec = __half22float2(indata2[i]);
			outdata2[i] = __float22half2_rn(make_float2(invec.x * a + b, invec.y * a + b));
			""",
			operation, name
		)

	else:
		return ElementwiseKernel(arguments, operation, name)


rbmKer = ElementwiseKernel(
	[(float_t.ptr, "outdata"), (float_t.const.ptr, "indata"), (float_t.const.ptr, "uni")],
	"float p = 1.0f / (1.0f + expf(-indata[i]));"
	"outdata[i] = (uni[i] < p)",
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
	"grad[i] = (pred[i] - target[i] > 0.0f ? -norm : norm)",
	"l1gradKer"
)


castFP16toFP32 = ElementwiseKernel(
	[(float_t.ptr, "outdata"), (half_t.const.ptr, "indata")],
	"outdata[i] = indata[i]",
	"castFP16toFP32", preambule="#include <cuda_fp16.h>"
)

castFP32toFP16 = ElementwiseKernel(
	[(half_t.ptr, "outdata"), (float_t.const.ptr, "indata")],
	"outdata[i] = indata[i]",
	"castFP32toFP16", preambule="#include <cuda_fp16.h>"
)
