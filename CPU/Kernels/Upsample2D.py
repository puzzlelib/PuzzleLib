import numpy as np

from PuzzleLib.Compiler.Codegen.Types import void_t, int32_t, float_t

from PuzzleLib.CPU.SourceModule import SourceModule
from PuzzleLib.CPU.CPUArray import CPUArray


upsample2dNearestTmpl = """

static void upsample2dNearest(float * __restrict outdata, const float * __restrict indata,
							  int32_t batchsize, int32_t maps, int32_t inh, int32_t inw, int32_t hscale, int32_t wscale)
{
	int32_t outh = inh * hscale, outw = inw * wscale;

	for (int32_t z = 0; z < batchsize * maps; z++)
		for (int32_t y = 0; y < inh; y++)
			for (int32_t x = 0; x < inw; x++)
				for (int32_t i = 0; i < hscale; i++)
					for (int32_t j = 0; j < wscale; j++)
					{
						int32_t outidx = z * outh * outw + (y * hscale + i) * outw + (x * wscale + j);
						outdata[outidx] = indata[z * inh * inw + y * inw + x];
					}
}

"""


nearestMod = SourceModule(upsample2dNearestTmpl, functions=[
	("upsample2dNearest", void_t, [
		(float_t.ptr.restrict, "outdata"), (float_t.const.ptr.restrict, "indata"), (int32_t, "batchsize"),
		(int32_t, "maps"), (int32_t, "inh"), (int32_t, "inw"), (int32_t, "hscale"), (int32_t, "wscale")
	], True)
])


def upsample2d(data, scale, mode="nearest"):
	batchsize, maps, inh, inw = data.shape
	hscale, wscale = (scale, scale) if isinstance(scale, int) else scale

	outh, outw = hscale * inh, wscale * inw
	outdata = CPUArray.empty((batchsize, maps, outh, outw), dtype=data.dtype)

	if mode == "nearest":
		nearestMod.upsample2dNearest(outdata.data, data.data, batchsize, maps, inh, inw, hscale, wscale)

	else:
		raise ValueError("Unsupported upsampling mode")

	return outdata


def unittest():
	batchsize, maps, inh, inw = 3, 2, 16, 15
	scale = 2

	data = CPUArray.toDevice(np.random.uniform(low=-1.0, high=1.0, size=(batchsize, maps, inh, inw)).astype(np.float32))
	outdata = upsample2d(data, scale, mode="nearest")

	hostData = data.get()
	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for y in range(inh):
				for x in range(inw):
					hostOutData[b, c, y * scale:(y + 1) * scale, x * scale:(x + 1) * scale] = hostData[b, c, y, x]

	assert np.allclose(hostOutData, outdata.get())


if __name__ == "__main__":
	unittest()
