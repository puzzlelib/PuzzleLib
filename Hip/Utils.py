import numpy as np

from PuzzleLib.Hip import Driver as HipDriver
from PuzzleLib.Cuda.Utils import SharedArray, shareMemTest, randomTest


class HipSharedArray(SharedArray):
	GPUArray = HipDriver.GPUArray


def unittest():
	from PuzzleLib.Hip import Backend

	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx, initmode=2)

		for dtype, _ in bnd.dtypesSupported():
			shareMemTest(bnd, dtype)
			memCopyTest(bnd, dtype)

		randomTest(bnd)


def memCopyTest(bnd, dtype):
	hostSrc = np.random.randn(4, 4, 4, 4).astype(dtype)

	src = bnd.GPUArray.toGpu(hostSrc)
	assert np.allclose(hostSrc, src.copy().get())

	hostA = np.random.randn(7, 4, 4, 4).astype(dtype)
	a = bnd.GPUArray.toGpu(hostA)

	out = bnd.concatenate((src, a), axis=0)
	assert np.allclose(np.concatenate((hostSrc, hostA), axis=0), out.get())

	hostA = np.random.randn(4, 2, 4, 4).astype(dtype)
	hostB = np.random.randn(4, 1, 4, 4).astype(dtype)

	a, b = bnd.GPUArray.toGpu(hostA), bnd.GPUArray.toGpu(hostB)

	out = bnd.concatenate((src, a, b), axis=1)
	assert np.allclose(np.concatenate((hostSrc, hostA, hostB), axis=1), out.get())

	hostA = np.random.randn(4, 4, 5, 4).astype(dtype)

	out = bnd.concatenate((bnd.GPUArray.toGpu(hostA), src), axis=2)
	assert np.allclose(np.concatenate((hostA, hostSrc), axis=2), out.get())

	outs = bnd.split(src, (2, 2), axis=0)
	assert all(np.allclose(hostSrc[2 * i:2 * (i + 1)], out.get()) for i, out in enumerate(outs))

	outs = bnd.split(src, (2, 2), axis=1)
	assert all(np.allclose(hostSrc[:, 2 * i:2 * (i + 1), :, :], out.get()) for i, out in enumerate(outs))

	outs = bnd.split(src, (2, 2), axis=2)
	assert all(np.allclose(hostSrc[:, :, 2 * i:2 * (i + 1), :], out.get()) for i, out in enumerate(outs))

	assert np.allclose(np.tile(hostB, (1, 3, 1, 1)), bnd.tile(b, 3, axis=1).get())


if __name__ == "__main__":
	unittest()
