import numpy as np

from PuzzleLib.Compiler.Codegen.Types import void_t, int32_t, float_t

from PuzzleLib.CPU.SourceModule import SourceModule
from PuzzleLib.CPU.CPUArray import CPUArray


src = """

inline static void map1d(int32_t b, int32_t c, int32_t maps, int32_t insize, int32_t outsize,
						 int32_t index, int32_t lpad, int32_t *inindex, int32_t *outindex)
{
	int32_t inoffset = (c + b * maps) * insize;
	int32_t outoffset = (c + b * maps) * outsize;

	int32_t instart = (lpad < 0) ? -lpad : 0;
	int32_t outstart = (lpad > 0) ? lpad : 0;

	int32_t x = abs(index - lpad) - abs(index - (insize + lpad - 1)) - index + 2 * lpad +
				insize - 1 - outstart + instart;

	*inindex = inoffset + x, *outindex = outoffset + index;
}

static void reflectpad1d(float * __restrict outdata, const float * __restrict indata,
						 int32_t batchsize, int32_t maps, int32_t insize, int32_t lpad, int32_t rpad)
{
	int outsize = insize + lpad + rpad;

	for (int32_t b = 0; b < batchsize; b++)
		for (int32_t c = 0; c < maps; c++)
			for (int32_t index = 0; index < outsize; index++)
			{
				int32_t inindex = 0, outindex = 0;
				map1d(b, c, maps, insize, outsize, index, lpad, &inindex, &outindex);

				outdata[outindex] = indata[inindex];
			}
}


inline static void map2d(int32_t b, int32_t c, int32_t maps, int32_t inh, int32_t inw, int32_t outh, int32_t outw,
						 int32_t index, int32_t upad, int32_t lpad, int32_t *inindex, int32_t *outindex)
{
	int32_t inoffset = (c + b * maps) * inh * inw;
	int32_t outoffset = (c + b * maps) * outh * outw;

	int32_t outx = index % outw, outy = index / outw;

	int32_t instartx = (lpad < 0) ? -lpad : 0, outstartx = (lpad > 0) ? lpad : 0;
	int32_t instarty = (upad < 0) ? -upad : 0, outstarty = (upad > 0) ? upad : 0;

	int32_t inx = abs(outx - lpad) - abs(outx - (inw + lpad - 1)) - outx + 2 * lpad + inw - 1 - outstartx + instartx;
	int32_t iny = abs(outy - upad) - abs(outy - (inh + upad - 1)) - outy + 2 * upad + inh - 1 - outstarty + instarty;

	*inindex = inoffset + iny * inw + inx;
	*outindex = outoffset + outy * outw + outx;
}

static void reflectpad2d(float * __restrict outdata, const float * __restrict indata,
						 int32_t batchsize, int32_t maps, int32_t inh, int32_t inw, int32_t upad, int32_t bpad,
						 int32_t lpad, int32_t rpad)
{
	int outh = inh + upad + bpad, outw = inw + lpad + rpad;

	for (int32_t b = 0; b < batchsize; b++)
		for (int32_t c = 0; c < maps; c++)
			for (int32_t index = 0; index < outh * outw; index++)
			{
				int32_t inindex = 0, outindex = 0;
				map2d(b, c, maps, inh, inw, outh, outw, index, upad, lpad, &inindex, &outindex);

				outdata[outindex] = indata[inindex];
			}
}

"""


mod = SourceModule(src, functions=[
	("reflectpad1d", void_t, [
		(float_t.ptr.restrict, "outdata"), (float_t.const.ptr.restrict, "indata"),
		(int32_t, "batchsize"), (int32_t, "maps"), (int32_t, "insize"), (int32_t, "lpad"), (int32_t, "rpad")
	]),
	("reflectpad2d", void_t, [
		(float_t.ptr.restrict, "outdata"), (float_t.const.ptr.restrict, "indata"),
		(int32_t, "batchsize"), (int32_t, "maps"), (int32_t, "inh"), (int32_t, "inw"),
		(int32_t, "upad"), (int32_t, "bpad"), (int32_t, "lpad"), (int32_t, "rpad")
	])
])


def reflectpad1d(data, pad):
	assert data.dtype == np.float32 and data.ndim == 3

	batchsize, maps, insize = data.shape
	lpad, rpad = pad

	assert insize >= max(lpad, rpad) + 1
	outdata = CPUArray.empty((batchsize, maps, insize + lpad + rpad), dtype=data.dtype)

	mod.reflectpad1d(outdata.data, data.data, batchsize, maps, insize, lpad, rpad)
	return outdata


def reflectpad2d(data, pad):
	assert data.dtype == np.float32 and data.ndim == 4

	batchsize, maps, inh, inw = data.shape
	upad, bpad, lpad, rpad = pad

	assert inh >= max(upad, bpad) + 1 and inw >= max(lpad, rpad) + 1
	outdata = CPUArray.empty((batchsize, maps, inh + upad + bpad, inw + lpad + rpad), dtype=data.dtype)

	mod.reflectpad2d(outdata.data, data.data, batchsize, maps, inh, inw, upad, bpad, lpad, rpad)
	return outdata


def unittest():
	reflectpad1dTest()
	reflectpad2dTest()


def reflectpad1dTest():
	batchsize, maps, insize = 4, 8, 48
	lpad, rpad = 2, 3

	data = CPUArray.toDevice(np.random.randn(batchsize, maps, insize).astype(np.float32))
	outdata = reflectpad1d(data, pad=(lpad, rpad))

	hostData, hostOutData = data.get(), outdata.get()

	assert np.allclose(hostOutData[:, :, lpad:insize + lpad], hostData)
	assert np.allclose(hostOutData[:, :, :lpad][:, :, ::-1], hostData[:, :, 1:lpad+1])
	assert np.allclose(hostOutData[:, :, insize + lpad:][:, :, ::-1], hostData[:, :, insize - 1 - rpad:insize - 1])


def reflectpad2dTest():
	batchsize, maps, inh, inw = 4, 8, 12, 15
	upad, bpad, lpad, rpad = 2, 3, 2, 3

	data = CPUArray.toDevice(np.random.randn(batchsize, maps, inh, inw).astype(np.float32))
	outdata = reflectpad2d(data, pad=(upad, bpad, lpad, rpad))

	hostData, hostOutData = data.get(), outdata.get()

	assert np.allclose(hostOutData[:, :, upad:inh + upad, lpad:inw + lpad], hostData)
	assert np.allclose(hostOutData[:, :, :upad, :lpad][:, :, ::-1, ::-1], hostData[:, :, 1:upad + 1, 1:lpad + 1])
	assert np.allclose(
		hostOutData[:, :, inh + upad:, inw + lpad:][:, :, ::-1, ::-1],
		hostData[:, :, inh - 1 - bpad:inh - 1, inw - 1 - rpad:inw - 1]
	)


if __name__ == "__main__":
	unittest()
