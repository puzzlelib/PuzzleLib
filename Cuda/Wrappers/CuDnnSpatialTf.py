import itertools, math

import numpy as np

from PuzzleLib.Cuda.Wrappers.CuDnn import context
from PuzzleLib.Cuda.GPUArray import GPUArray
from PuzzleLib.Cuda.Utils import dtypesSupported


def unittest():
	for dtype, atol in dtypesSupported():
		spatialTfTest(dtype, atol)


def spatialTfTest(dtype, atol):
	batchsize, maps, inh, inw = 1, 1, 4, 4
	outh, outw = int(1.0 * inh), int(1.0 * inw)

	hostData = np.random.randn(batchsize, maps, inh, inw).astype(dtype)
	hostTf = np.tile(np.array([[1.0, 0.1, -0.001], [0.0, 0.9, -0.001]], dtype=dtype), reps=(batchsize, 1, 1))

	data, transform = GPUArray.toGpu(hostData), GPUArray.toGpu(hostTf)
	outdata, grid = context.spatialTf(data, transform, outshape=(batchsize, maps, outh, outw), getGrid=True)

	hostGrid = np.empty((batchsize, outh, outw, 2), dtype=dtype)
	xstep, ystep = 2.0 / (outw - 1), 2.0 / (outh - 1)

	for b, y, x in itertools.product(range(batchsize), range(outh), range(outw)):
		hostGrid[b, y, x] = np.dot(hostTf[b], np.array([-1.0 + x * xstep, -1.0 + y * ystep, 1.0], dtype=np.float32))

	assert np.allclose(hostGrid, grid.get(), atol=atol)

	hostOutData = np.zeros(outdata.shape, dtype=dtype)
	xstep, ystep = 2.0 / (inw - 1), 2.0 / (inh - 1)

	for b, c, y, x in itertools.product(range(batchsize), range(maps), range(outh), range(outw)):
		dstx, dsty = hostGrid[b, y, x]
		ny, nx = (dsty + 1.0) / ystep, (dstx + 1.0) / xstep

		srcy, srcx = int(math.floor(ny)), int(math.floor(nx))
		dy, dx = ny - srcy, nx - srcx

		ul, ur, bl, br = 0.0, 0.0, 0.0, 0.0
		if 0 <= srcy < inh and 0 <= srcx < inw: ul = hostData[b, c, srcy, srcx] * (1 - dy) * (1 - dx)
		if 0 <= srcy + 1 < inh and 0 <= srcx < inw: bl = hostData[0, 0, srcy + 1, srcx] * dy * (1 - dx)
		if 0 <= srcy < inh and 0 <= srcx + 1 < inw: ur = hostData[0, 0, srcy, srcx + 1] * (1 - dy) * dx
		if 0 <= srcy + 1 < inh and 0 <= srcx + 1 < inw: br = hostData[0, 0, srcy + 1, srcx + 1] * dy * dx

		hostOutData[b, c, y, x] = ul + ur + bl + br

	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrad, dtransform, dgrid = context.spatialTfBackward(grad, data, grid, getDGrid=True)

	hostInGrad = np.zeros(data.shape, dtype=dtype)

	for b, c, y, x in itertools.product(range(batchsize), range(maps), range(outh), range(outw)):
		dstx, dsty = hostGrid[b, y, x]
		ny, nx = (dsty + 1.0) / ystep, (dstx + 1.0) / xstep

		srcy, srcx = int(math.floor(ny)), int(math.floor(nx))
		dy, dx = ny - srcy, nx - srcx

		val = hostGrad[b, c, y, x]
		if 0 <= srcy < inh and 0 <= srcx < inw: hostInGrad[b, c, srcy, srcx] += val * (1 - dy) * (1 - dx)
		if 0 <= srcy + 1 < inh and 0 <= srcx < inw: hostInGrad[b, c, srcy + 1, srcx] += val * dy * (1 - dx)
		if 0 <= srcy < inh and 0 <= srcx + 1 < inw: hostInGrad[b, c, srcy, srcx + 1] += val * (1 - dy) * dx
		if 0 <= srcy + 1 < inh and 0 <= srcx + 1 < inw: hostInGrad[b, c, srcy + 1, srcx + 1] += val * dy * dx

	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	hostDGrid = np.zeros(dgrid.shape, dtype=dtype)

	for b, c, y, x in itertools.product(range(batchsize), range(maps), range(outh), range(outw)):
		dstx, dsty = hostGrid[b, y, x]
		ny, nx = (dsty + 1.0) / ystep, (dstx + 1.0) / xstep

		srcy, srcx = int(math.floor(ny)), int(math.floor(nx))
		dy, dx = ny - srcy, nx - srcx

		valx, valy = 0, 0

		if 0 <= srcy < inh and 0 <= srcx < inw: valx -= hostData[b, c, srcy, srcx] / xstep * (1 - dy)
		if 0 <= srcy + 1 < inh and 0 <= srcx < inw: valx -= hostData[b, c, srcy + 1, srcx] / xstep * dy
		if 0 <= srcy < inh and 0 <= srcx + 1 < inw:  valx += hostData[b, c, srcy, srcx + 1] / xstep * (1-dy)
		if 0 <= srcy + 1 < inh and 0 <= srcx + 1 < inw: valx += hostData[b, c, srcy + 1, srcx + 1] / xstep * dy

		if 0 <= srcy < inh and 0 <= srcx < inw: valy -= hostData[b, c, srcy, srcx] / ystep * (1 - dx)
		if 0 <= srcy + 1 < inh and 0 <= srcx < inw: valy += hostData[b, c, srcy + 1, srcx] / ystep * (1-dx)
		if 0 <= srcy < inh and 0 <= srcx + 1 < inw: valy -= hostData[b, c, srcy, srcx + 1] / ystep * dx
		if 0 <= srcy + 1 < inh and 0 <= srcx + 1 < inw: valy += hostData[b, c, srcy + 1, srcx + 1] / ystep * dx

		hostDGrid[b, y, x] = hostGrad[b, c, y, x] * valx, hostGrad[b, c, y, x] * valy

	assert np.allclose(hostDGrid, dgrid.get(), atol=atol)

	hostDTransform = np.zeros(dtransform.shape, dtype=dtype)
	xstep, ystep = 2.0 / (outw - 1), 2.0 / (outh - 1)

	for b, y, x in itertools.product(range(batchsize), range(outh), range(outw)):
		hostDTransform[b] += np.outer(
			hostDGrid[b, y, x], np.array([-1.0 + x * xstep, -1.0 + y * ystep, 1], dtype=np.float32)
		)

	assert np.allclose(hostDTransform, dtransform.get(), atol=atol)


if __name__ == "__main__":
	unittest()
