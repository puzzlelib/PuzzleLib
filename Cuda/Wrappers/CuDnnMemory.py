from PuzzleLib.Cuda.Kernels.Memory import transposeTest, moveAxisTest, swapAxesTest, depthConcatTest


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx, initmode=1)

		for dtype, _ in bnd.dtypesSupported():
			transposeTest(bnd, bnd.dnn, dtype)
			moveAxisTest(bnd, bnd.dnn, dtype)
			swapAxesTest(bnd, bnd.dnn, dtype)
			depthConcatTest(bnd, bnd.dnn, dtype)


if __name__ == "__main__":
	unittest()
