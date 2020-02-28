import ctypes
from enum import Enum

import numpy as np

from PuzzleLib.Cuda.GPUArray import GPUArray
from PuzzleLib.Cuda.Utils import memoryPool as memPool

from PuzzleLib.Cuda.ThirdParty import libnpp


class InterpolationMode(Enum):
	nn = libnpp.NppiInterpolationMode["NPPI_INTER_NN"]
	linear = libnpp.NppiInterpolationMode["NPPI_INTER_LINEAR"]
	cubic = libnpp.NppiInterpolationMode["NPPI_INTER_CUBIC"]
	cubic2pbSpline = libnpp.NppiInterpolationMode["NPPI_INTER_CUBIC2P_BSPLINE"]
	cubic2pCatmullRom = libnpp.NppiInterpolationMode["NPPI_INTER_CUBIC2P_CATMULLROM"]
	cubic2pb05c03 = libnpp.NppiInterpolationMode["NPPI_INTER_CUBIC2P_B05C03"]
	super = libnpp.NppiInterpolationMode["NPPI_INTER_SUPER"]
	lanczos = libnpp.NppiInterpolationMode["NPPI_INTER_LANCZOS"]
	lanczos3Advanced = libnpp.NppiInterpolationMode["NPPI_INTER_LANCZOS3_ADVANCED"]
	smoothEdge = libnpp.NppiInterpolationMode["NPPI_SMOOTH_EDGE"]


class DataType(Enum):
	u8 = "8u"
	u16 = "16u"
	s16 = "16s"
	f32 = "32f"
	f64 = "64f"


class MemoryType(Enum):
	grayscale = "C1R"
	rgb = "C3R"
	rgba = "C4R"
	rgbPlanar = "P3R"
	rgbaPlanar = "P4R"


def isPlanarMemoryType(memoryType):
	return memoryType == MemoryType.rgbPlanar or memoryType == MemoryType.rgbaPlanar


def getDataRect(data, memoryType):
	if memoryType == MemoryType.grayscale or memoryType == MemoryType.rgb or memoryType == MemoryType.rgba:
		return 0, 0, data.shape[1], data.shape[0]

	elif isPlanarMemoryType(memoryType):
		return 0, 0, data.shape[2], data.shape[1]

	else:
		raise NotImplementedError(memoryType)


def getDataType(data):
	if data.dtype == np.uint8:
		return DataType.u8
	elif data.dtype == np.uint16:
		return DataType.u16
	elif data.dtype == np.int16:
		return DataType.s16
	elif data.dtype == np.float32:
		return DataType.f32
	elif data.dtype == np.float64:
		return DataType.f64

	else:
		raise NotImplementedError(data.dtype)


def getMemoryTypeLineSize(line, dtype, memoryType):
	if isPlanarMemoryType(memoryType) or memoryType == MemoryType.grayscale:
		return line * dtype.itemsize
	elif memoryType == MemoryType.rgb:
		return 3 * dtype.itemsize * line
	elif memoryType == MemoryType.rgba:
		return 4 * dtype.itemsize * line

	else:
		raise NotImplementedError(memoryType)


def getOutDataShape(data, outrect, memoryType):
	_, _, outw, outh = outrect

	if data.ndim == 2 and memoryType == MemoryType.grayscale:
		return outh, outw
	elif isPlanarMemoryType(memoryType):
		return data.shape[0], outh, outw
	elif memoryType == MemoryType.rgb or memoryType == MemoryType.rgba:
		return outh, outw, data.shape[2]

	else:
		raise NotImplementedError(memoryType)


def getOutDataRect(data, outshape, memoryType):
	if data.ndim == 2 and memoryType == MemoryType.grayscale:
		outh, outw = outshape
		return 0, 0, outw, outh
	elif isPlanarMemoryType(memoryType):
		_, outh, outw = outshape
		return 0, 0, outw, outh
	elif memoryType == MemoryType.rgb or memoryType == MemoryType.rgba:
		outh, outw, _ = outshape
		return 0, 0, outw, outh

	else:
		raise NotImplementedError(memoryType)


def getDataPointers(data, outdata, memoryType):
	dataPtr = data.ptr
	outdataPtr = outdata.ptr

	if isPlanarMemoryType(memoryType):
		dataPtr = [data.ptr + data.strides[0] * i for i in range(data.shape[0])]
		dataPtr = (ctypes.c_void_p * len(dataPtr))(*dataPtr)

		outdataPtr = [outdata.ptr + outdata.strides[0] * i for i in range(outdata.shape[0])]
		outdataPtr = (ctypes.c_void_p * len(outdataPtr))(*outdataPtr)

	return dataPtr, outdataPtr


def rescale(data, scale, memoryType, interpolation=InterpolationMode.nn, outdata=None, allocator=memPool):
	assert data.ndim == 2 and memoryType == MemoryType.grayscale or data.ndim == 3
	hscale, wscale = (scale, scale) if isinstance(scale, (int, float)) else scale

	inrect = getDataRect(data, memoryType)
	insize, inline = (inrect[2], inrect[3]), getMemoryTypeLineSize(inrect[2], data.dtype, memoryType)

	outrect = libnpp.nppiGetResizeRect(inrect, wscale, hscale, 0, 0, interpolation.value)
	outline = getMemoryTypeLineSize(outrect[2], data.dtype, memoryType)

	outshape = getOutDataShape(data, outrect, memoryType)

	if outdata is None:
		outdata = GPUArray.empty(outshape, dtype=data.dtype, allocator=allocator)
	else:
		assert outdata.shape == outshape

	dataPtr, outdataPtr = getDataPointers(data, outdata, memoryType)

	libnpp.nppiResizeSqrPixel(
		getDataType(data).value, memoryType.value, dataPtr, insize, inline, inrect,
		outdataPtr, outline, outrect, wscale, hscale, 0, 0, interpolation.value
	)

	return outdata


def resize(data, outshape, memoryType, interpolation=InterpolationMode.nn, outdata=None, allocator=memPool):
	inrect = getDataRect(data, memoryType)
	outrect = getOutDataRect(data, outshape, memoryType)

	hscale, wscale = outrect[3] / inrect[3], outrect[2] / inrect[2]
	return rescale(data, (hscale, wscale), memoryType, interpolation, outdata=outdata, allocator=allocator)


def warpAffine(data, coeffs, memoryType, outshape=None, interpolation=InterpolationMode.nn, cval=0, backward=False,
			   allocator=memPool):
	assert data.ndim == 2 and memoryType == MemoryType.grayscale or data.ndim == 3

	inrect = getDataRect(data, memoryType)
	insize, inline = (inrect[2], inrect[3]), getMemoryTypeLineSize(inrect[2], data.dtype, memoryType)

	if outshape is None:
		outshape = data.shape

	outrect = getOutDataRect(data, outshape, memoryType)
	outline = getMemoryTypeLineSize(outrect[2], data.dtype, memoryType)

	outdata = GPUArray.empty(outshape, dtype=data.dtype, allocator=allocator)
	outdata.fill(cval)

	dataPtr, outdataPtr = getDataPointers(data, outdata, memoryType)

	warpMethod = libnpp.nppiWarpAffine
	if backward:
		warpMethod = libnpp.nppiWarpAffineBack

	warpMethod(
		getDataType(data).value, memoryType.value, dataPtr, insize, inline, inrect, outdataPtr,
		outline, outrect, coeffs, interpolation.value
	)

	return outdata


def genAffineQuads(inpoints, outpoints, clip, inrect):
	inx0, iny0 = inpoints[0]
	inx1, iny1 = inpoints[1]
	inx2, iny2 = inpoints[2]

	outx0, outy0 = outpoints[0]
	outx1, outy1 = outpoints[1]
	outx2, outy2 = outpoints[2]

	srcQuad = inpoints[0] + inpoints[1] + inpoints[2]
	srcQuad.extend([inx2 + inx0 - inx1, iny2 + iny0 - iny1])

	dstQuad = outpoints[0] + outpoints[1] + outpoints[2]
	dstQuad.extend([outx2 + outx0 - outx1, outy2 + outy0 - outy1])

	if not clip:
		intransform = np.zeros((3, 3), dtype=np.float32)
		intransform[2, 2] = 1.0

		intransform[:2] = np.array(libnpp.nppiGetAffineTransform(inrect, srcQuad), dtype=np.float32).reshape(2, 3)

		outtransform = np.zeros((3, 3), dtype=np.float32)
		outtransform[2, 2] = 1.0

		outtransform[:2] = np.array(libnpp.nppiGetAffineTransform(inrect, dstQuad), dtype=np.float32).reshape(2, 3)

		transform = np.dot(outtransform, np.linalg.inv(intransform))[:2]

		inh, inw = inrect[2], inrect[3]
		srcQuad = [0, inw, 0.0, 0.0, inh, 0.0, inh, inw]

		dstQuad = []
		for i in range(len(srcQuad) >> 1):
			inpoint = srcQuad[2 * i: 2 * (i + 1)]
			dstQuad.extend(list(np.dot(transform, np.array(inpoint + [1.0]))))

	return srcQuad, dstQuad


def warpAffinePoints(data, inpoints, outpoints, memoryType, outshape=None, interpolation=InterpolationMode.nn, cval=0,
					 clip=True, allocator=memPool):
	assert data.ndim == 2 and memoryType == MemoryType.grayscale or data.ndim == 3

	inrect = getDataRect(data, memoryType)
	insize, inline = (inrect[2], inrect[3]), getMemoryTypeLineSize(inrect[2], data.dtype, memoryType)

	if outshape is None:
		outshape = data.shape

	outrect = getOutDataRect(data, outshape, memoryType)
	outline = getMemoryTypeLineSize(outrect[2], data.dtype, memoryType)

	outdata = GPUArray.empty(outshape, dtype=data.dtype, allocator=allocator)
	outdata.fill(cval)

	dataPtr, outdataPtr = getDataPointers(data, outdata, memoryType)
	srcQuad, dstQuad = genAffineQuads(inpoints, outpoints, clip, inrect)

	libnpp.nppiWarpAffineQuad(
		getDataType(data).value, memoryType.value, dataPtr, insize, inline, inrect, srcQuad,
		outdataPtr, outline, outrect, dstQuad, interpolation.value
	)

	return outdata


def unittest():
	resizeWidePixelTest()
	resizePlanarTest()
	warpAffineTest()
	warpAffinePointsTest()


def resizeWidePixelTest():
	inh, inw = 2, 4
	hscale, wscale = 2.5, 1.5

	data = GPUArray.toGpu(np.random.randn(inh, inw, 3).astype(np.float32))

	outdata = rescale(data, scale=(hscale, wscale), memoryType=MemoryType.rgb, interpolation=InterpolationMode.linear)
	outresdata = resize(data, outdata.shape, memoryType=MemoryType.rgb, interpolation=InterpolationMode.linear)

	hostData = data.get()
	hostOutData = np.empty((int(inh * hscale), int(inw * wscale), 3), dtype=data.dtype)

	def hostResizeWide(hostDat, hostOutDat):
		for y in range(hostOutDat.shape[0]):
			ny = max(0.0, (y + 0.5) / hscale - 0.5)
			iyp = int(ny)
			dy = ny - iyp
			iypp = min(iyp + 1, hostDat.shape[0] - 1)

			for x in range(hostOutDat.shape[1]):
				nx = max(0.0, (x + 0.5) / wscale - 0.5)
				ixp = int(nx)
				dx = nx - ixp
				ixpp = min(ixp + 1, hostDat.shape[1] - 1)

				hostOutDat[y, x, :] = (1.0 - dy) * (1.0 - dx) * hostDat[iyp, ixp, :] + \
									  dy * (1.0 - dx) * hostDat[iypp, ixp, :] + \
									  (1.0 - dy) * dx * hostDat[iyp, ixpp, :] + \
									  dy * dx * hostDat[iypp, ixpp, :]

	hostResizeWide(hostData, hostOutData)
	assert np.allclose(hostOutData, outdata.get())

	hscale, wscale = hostOutData.shape[0] / hostData.shape[0], hostOutData.shape[1] / hostData.shape[1]

	hostResizeWide(hostData, hostOutData)
	assert np.allclose(hostOutData, outresdata.get())


def resizePlanarTest():
	inh, inw = 2, 4
	hscale, wscale = 2.5, 1.5

	data = GPUArray.toGpu(np.random.randn(3, inh, inw).astype(np.float32))

	outdata = rescale(
		data, scale=(hscale, wscale), memoryType=MemoryType.rgbPlanar, interpolation=InterpolationMode.linear
	)
	outresdata = resize(
		data, outdata.shape, memoryType=MemoryType.rgbPlanar, interpolation=InterpolationMode.linear
	)

	hostData = data.get()
	hostOutData = np.empty((3, int(inh * hscale), int(inw * wscale)), dtype=data.dtype)

	def hostResizePlanar(hostDat, hostOutDat):
		for y in range(hostOutDat.shape[1]):
			ny = max(0.0, (y + 0.5) / hscale - 0.5)
			iyp = int(ny)
			dy = ny - iyp
			iypp = min(iyp + 1, hostDat.shape[1] - 1)

			for x in range(hostOutDat.shape[2]):
				nx = max(0.0, (x + 0.5) / wscale - 0.5)
				ixp = int(nx)
				dx = nx - ixp
				ixpp = min(ixp + 1, hostDat.shape[2] - 1)

				hostOutDat[:, y, x] = (1.0 - dy) * (1.0 - dx) * hostDat[:, iyp, ixp] + \
									  dy * (1.0 - dx) * hostDat[:, iypp, ixp] + \
									  (1.0 - dy) * dx * hostDat[:, iyp, ixpp] + \
									  dy * dx * hostDat[:, iypp, ixpp]

	hostResizePlanar(hostData, hostOutData)
	assert np.allclose(hostOutData, outdata.get())

	hscale, wscale = hostOutData.shape[1] / hostData.shape[1], hostOutData.shape[2] / hostData.shape[2]

	hostResizePlanar(hostData, hostOutData)
	assert np.allclose(hostOutData, outresdata.get())


def warpAffineTest():
	inh, inw = 4, 4
	outh, outw = 10, 10

	mat = np.array([
		[0.0, 2.0, 0.0],
		[5.0, 0.0, 0.0],
		[0.0, 0.0, 1.0]
	], dtype=np.float32)

	invMat = np.linalg.inv(mat)
	mat = list(mat[:2].ravel())

	data = GPUArray.toGpu(np.random.randn(inh, inw, 3).astype(np.float32))

	outdata = warpAffine(
		data, mat, memoryType=MemoryType.rgb, outshape=(outh, outw, 3), interpolation=InterpolationMode.nn
	)

	outbackdata = warpAffine(
		data, list(invMat[:2].ravel()), memoryType=MemoryType.rgb, outshape=(outh, outw, 3),
		interpolation=InterpolationMode.nn, backward=True
	)

	hostData = data.get()
	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	for y in range(hostOutData.shape[0]):
		for x in range(hostOutData.shape[1]):
			inx = int(invMat[0, 0] * x + invMat[0, 1] * y + invMat[0, 2] + 0.5)
			iny = int(invMat[1, 0] * x + invMat[1, 1] * y + invMat[1, 2] + 0.5)

			if 0 <= inx < hostData.shape[1] and 0 <= iny < hostData.shape[0]:
				hostOutData[y, x, :] = hostData[iny, inx, :]

	assert np.allclose(hostOutData, outdata.get())
	assert np.allclose(hostOutData, outbackdata.get())


def warpAffinePointsTest():
	inh, inw = 4, 4
	outh, outw = 4, 4

	inpoints = [[0, inw], [0, 0], [inh, 0]]
	outpoints = [[outw, 0], [0, 0], [0, outh]]

	data = GPUArray.toGpu(np.random.randn(inh, inw, 3).astype(np.float32))

	outdata = warpAffinePoints(
		data, inpoints, outpoints, memoryType=MemoryType.rgb, outshape=(outh, outw, 3),
		interpolation=InterpolationMode.nn
	)

	hostData = data.get()
	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	dx, dy = outpoints[1][0] - inpoints[1][0], outpoints[1][1] - inpoints[1][1]

	A = np.array([
		[inpoints[0][0] - inpoints[1][0], inpoints[2][0] - inpoints[1][0]],
		[inpoints[0][1] - inpoints[1][1], inpoints[2][1] - inpoints[1][1]]
	], dtype=np.float32)

	oa = np.array([outpoints[0][0] - outpoints[1][0], outpoints[0][1] - outpoints[1][1]])
	ob = np.array([outpoints[2][0] - outpoints[1][0], outpoints[2][1] - outpoints[1][1]])

	x1, y1 = np.linalg.solve(A, oa)
	x2, y2 = np.linalg.solve(A, ob)

	mat = np.array([
		[x1, x2, dx],
		[y1, y2, dy],
		[0.0, 0.0, 1.0]
	], dtype=np.float32)

	invMat = np.linalg.inv(mat)

	for y in range(hostOutData.shape[0]):
		for x in range(hostOutData.shape[1]):
			inx = int(invMat[0, 0] * x + invMat[0, 1] * y + invMat[0, 2] + 0.5)
			iny = int(invMat[1, 0] * x + invMat[1, 1] * y + invMat[1, 2] + 0.5)

			if 0 <= inx < hostData.shape[1] and 0 <= iny < hostData.shape[0]:
				hostOutData[y, x, :] = hostData[iny, inx, :]

	assert np.allclose(hostOutData, outdata.get())


if __name__ == "__main__":
	unittest()
