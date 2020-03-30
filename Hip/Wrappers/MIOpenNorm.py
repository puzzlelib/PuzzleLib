import itertools
import numpy as np

from PuzzleLib.Cuda.Wrappers.CuDnnNorm import batchNorm2dTest, batchNorm3dTest, instanceNorm2dTest


def unittest():
	from PuzzleLib.Hip import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx, initmode=2)

		float32 = bnd.dtypesSupported()[0]

		batchNorm2dTest(bnd, *float32, np.float32)
		batchNorm3dTest(bnd, *float32, np.float32)
		instanceNorm2dTest(bnd, *float32, np.float32)

		for dtype, atol in bnd.dtypesSupported():
			mapLRN2dTest(bnd, dtype, atol)


def mapLRN2dTest(bnd, dtype, atol):
	batchsize, maps, h, w = 2, 2, 9, 10
	N, alpha, beta, K = 5, 1.0, 0.5, 2.0

	lookBehind = int((N - 1) / 2)
	lookAhead = N - lookBehind

	hostData = np.random.randn(batchsize, maps, h, w).astype(dtype)

	data = bnd.GPUArray.toGpu(hostData)
	outdata, workspace = bnd.dnn.lrn(data, N=N, alpha=alpha, beta=beta, K=K, mode=bnd.LRNMode.map.value)

	norms = np.empty(hostData.shape, dtype=np.float32)

	for b, c, y, x in itertools.product(range(batchsize), range(maps), range(h), range(w)):
		slcy = slice(max(0, y - lookBehind), min(h, y + lookAhead))
		slcx = slice(max(0, x - lookBehind), min(w, x + lookAhead))

		slc = hostData[b, c, slcy, slcx].ravel()
		norms[b, c, y, x] = K + np.dot(slc, slc) * alpha / N**2

	hostOutData = (hostData / norms**beta).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.lrnBackward(
		grad, data, outdata, workspace, N=N, alpha=alpha, beta=beta, K=K, mode=bnd.LRNMode.map.value
	)

	hostInGrad = hostGrad / norms**beta
	k = 2.0 * alpha * beta / N**2

	for b, c, y, x in itertools.product(range(batchsize), range(maps), range(h), range(w)):
		slcy = slice(max(0, y - lookBehind), min(h, y + lookAhead))
		slcx = slice(max(0, x - lookBehind), min(w, x + lookAhead))

		slcdata, slcgrad = hostData[b, c, slcy, slcx].ravel(), hostGrad[b, c, slcy, slcx].ravel()
		slcnorms = norms[b, c, slcy, slcx].ravel()

		hostInGrad[b, c, y, x] -= k * hostData[b, c, y, x] * np.dot(slcgrad, slcdata / slcnorms**(beta + 1))

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)


if __name__ == "__main__":
	unittest()
