import numpy as np

from PuzzleLib.Cuda.GPUArray import extendGPUArray, arithmTest

from PuzzleLib.Hip import Driver as HipDriver
from PuzzleLib.Hip.SourceModule import HipEltwiseKernel, HipEltHalf2Kernel, HipReductionKernel


HipGPUArray = extendGPUArray(HipDriver, HipEltwiseKernel, HipEltHalf2Kernel, HipReductionKernel)


def unittest():
	from PuzzleLib.Hip import Backend

	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx)

		for dtype, _ in bnd.dtypesSupported():
			arithmTest(bnd, dtype)
			memoryTest(bnd, dtype)


def memoryTest(bnd, dtype):
	hostA = np.random.randn(10, 10).astype(dtype)
	a = bnd.GPUArray.toGpu(hostA)

	b = a[:, :6]
	hostB = hostA[:, :6]

	assert np.allclose(hostB.reshape((2, 5, 6)), b.reshape(2, 5, 6).get())
	assert np.allclose(hostB.reshape((5, 2, 3, 2)), b.reshape(5, 2, 3, 2).get())
	assert np.allclose(hostB.reshape((10, 1, 6)), b.reshape(10, 1, 6).get())

	hostA = np.random.randn(10, 10, 10).astype(dtype)
	a = bnd.GPUArray.toGpu(hostA)

	b = a[:, :, :6]
	assert np.allclose(hostA[:, :, :6], b.get())

	hostB = np.random.randn(*b.shape).astype(dtype)
	b.set(hostB)
	assert np.allclose(hostB, b.get())

	hostB = b.get()
	b = a[:, :6, :6]
	assert np.allclose(hostB[:, :6, :6], b.get())


if __name__ == "__main__":
	unittest()
