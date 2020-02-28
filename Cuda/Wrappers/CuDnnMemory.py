import itertools

import numpy as np

from PuzzleLib.Cuda.Utils import dtypesSupported
from PuzzleLib.Cuda.GPUArray import GPUArray
from PuzzleLib.Cuda.Wrappers.CuDnn import context


def unittest():
	for dtype, _ in dtypesSupported():
		transposeTest(dtype)
		moveAxisTest(dtype)
		swapAxesTest(dtype)
		depthConcatTest(dtype)


def transposeTest(dtype):
	shapes = [(10, ), (10, 3), (10, 3, 5, 4, 2)]

	for shape in shapes:
		for axes in itertools.permutations(range(len(shape))):
			hostData = np.random.randn(*shape).astype(dtype)

			data = GPUArray.toGpu(hostData)
			outdata = context.transpose(data, axes=axes)

			hostOutData = np.transpose(hostData, axes=axes)
			assert np.allclose(hostOutData, outdata.get())


def moveAxisTest(dtype):
	shapes = [(10, ), (10, 3), (10, 3, 5, 4, 2)]

	for shape in shapes:
		for src, dst in itertools.product(range(len(shape)), range(len(shape))):
			hostData = np.random.randn(*shape).astype(dtype)

			data = GPUArray.toGpu(hostData)
			outdata = context.moveaxis(data, src=src, dst=dst)

			hostOutData = np.moveaxis(hostData, source=src, destination=dst)
			assert np.allclose(hostOutData, outdata.get())


def swapAxesTest(dtype):
	shapes = [(10,), (10, 3), (10, 3, 5, 4, 2)]

	for shape in shapes:
		for axis1, axis2 in itertools.product(range(len(shape)), range(len(shape))):
			hostData = np.random.randn(*shape).astype(dtype)

			data = GPUArray.toGpu(hostData)
			outdata = context.swapaxes(data, axis1=axis1, axis2=axis2)

			hostOutData = np.swapaxes(hostData, axis1=axis1, axis2=axis2)
			assert np.allclose(hostOutData, outdata.get())


def depthConcatTest(dtype):
	hostData1 = np.random.randn(3, 4, 3, 3).astype(dtype)
	hostData2 = np.random.randn(3, 2, 6, 6).astype(dtype)
	hostData3 = np.random.randn(3, 5, 4, 4).astype(dtype)
	allHostData = [hostData1, hostData2, hostData3]

	allData = [GPUArray.toGpu(data) for data in allHostData]
	outdata = context.depthConcat(allData)

	depth, h, w = 0, 0, 0
	for data in allHostData:
		depth += data.shape[1]
		h, w = max(h, data.shape[2]), max(w, data.shape[3])

	hostOutData = np.zeros(shape=(allHostData[0].shape[0], depth, h, w), dtype=dtype)

	hostOutData[:, :4, 1:4, 1:4] = hostData1
	hostOutData[:, 4:6, :, :] = hostData2
	hostOutData[:, 6:, 1:5, 1:5] = hostData3

	assert np.allclose(hostOutData, outdata.get())

	hostGrad = np.random.randn(*hostOutData.shape).astype(dtype)

	grad = GPUArray.toGpu(hostGrad)
	ingrads = context.depthSplit(grad, allData)

	hostInGrads = [
		hostGrad[:, :4, 1:4, 1:4],
		hostGrad[:, 4:6, :, :],
		hostGrad[:, 6:, 1:5, 1:5]
	]

	assert all(np.allclose(hostInGrad, ingrads[i].get()) for i, hostInGrad in enumerate(hostInGrads))


if __name__ == "__main__":
	unittest()
