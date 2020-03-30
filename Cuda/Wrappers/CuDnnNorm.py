import itertools
import numpy as np


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx, initmode=2)

		for dtype, atol in bnd.dtypesSupported():
			batchNorm2dTest(bnd, dtype, atol, np.float32)
			batchNorm3dTest(bnd, dtype, atol, np.float32)
			instanceNorm2dTest(bnd, dtype, atol, np.float32)

			mapLRN2dTest(bnd, dtype, atol)
			crossMapLRN2dTest(bnd, dtype, atol)


def batchNorm2dTest(bnd, dtype, atol, calctype):
	batchsize, maps, h, w = 4, 5, 2, 3
	epsilon, norm = 1e-5, batchsize * h * w

	hostData = np.random.randn(batchsize, maps, h, w).astype(dtype)
	hostScale = np.random.randn(1, maps, 1, 1).astype(calctype)
	hostBias = np.random.randn(1, maps, 1, 1).astype(calctype)

	data = bnd.GPUArray.toGpu(hostData)
	scale, bias = bnd.GPUArray.toGpu(hostScale.ravel()), bnd.GPUArray.toGpu(hostBias.ravel())
	mean = bnd.GPUArray.zeros(scale.shape, dtype=calctype)
	var = bnd.GPUArray.toGpu(np.ones(scale.shape, dtype=calctype))

	outdata, savemean, saveinvvar = bnd.dnn.batchNormNd(data, mean, var, scale, bias, epsilon=epsilon, out=data)

	hostMean = np.sum(hostData, axis=(0, 2, 3), dtype=calctype, keepdims=True) / norm

	hostInvVar = np.sum((hostData - hostMean)**2, axis=(0, 2, 3), dtype=calctype, keepdims=True) / norm
	hostInvVar = 1.0 / np.sqrt(hostInvVar + epsilon)

	hostNormData = (hostData - hostMean) * hostInvVar
	hostOutData = (hostNormData * hostScale + hostBias).astype(dtype)

	assert np.allclose(hostMean.ravel(), mean.get(), atol=atol)
	assert np.allclose(hostInvVar.ravel(), saveinvvar.get(), atol=atol)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad, data = bnd.GPUArray.toGpu(hostGrad), bnd.GPUArray.toGpu(hostData)
	ingrad, scalegrad, bgrad = bnd.dnn.batchNormNdBackward(grad, data, scale, savemean, saveinvvar, epsilon=epsilon)

	hostScaleGrad = np.sum(hostGrad * hostNormData, axis=(0, 2, 3), dtype=calctype, keepdims=True)
	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3), dtype=calctype, keepdims=True)

	hostMeanGrad = -hostInvVar * hostBiasGrad * hostScale

	hostVarGrad = np.sum(hostGrad * (hostData - hostMean), axis=(0, 2, 3), dtype=calctype, keepdims=True)
	hostVarGrad = -0.5 * hostVarGrad * hostScale * hostInvVar**3

	hostInGrad = hostGrad * hostScale * hostInvVar + (2 * hostVarGrad * (hostData - hostMean) + hostMeanGrad) / norm
	hostInGrad = hostInGrad.astype(dtype)

	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)
	assert np.allclose(hostScaleGrad.ravel(), scalegrad.get(), atol=atol)
	assert np.allclose(hostBiasGrad.ravel(), bgrad.get(), atol=atol)

	hostMean = np.random.randn(*hostMean.shape).astype(np.float32)
	hostVar = 1.0 + np.random.randn(*hostInvVar.shape).astype(np.float32)**2

	mean, var = bnd.GPUArray.toGpu(hostMean.ravel()), bnd.GPUArray.toGpu(hostVar.ravel())
	outdata = bnd.dnn.batchNormNd(data, mean, var, scale, bias, test=True)

	hostOutData = ((hostData - hostMean) / np.sqrt(hostVar + epsilon) * hostScale + hostBias).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)


def batchNorm3dTest(bnd, dtype, atol, calctype):
	batchsize, maps, d, h, w = 2, 5, 2, 3, 2
	epsilon, norm = 1e-5, batchsize * d * h * w

	hostData = np.random.randn(batchsize, maps, d, h, w).astype(dtype)

	hostScale = np.random.randn(1, maps, 1, 1, 1).astype(calctype)
	hostBias = np.random.randn(1, maps, 1, 1, 1).astype(calctype)

	data = bnd.GPUArray.toGpu(hostData)
	scale, bias = bnd.GPUArray.toGpu(hostScale.ravel()), bnd.GPUArray.toGpu(hostBias.ravel())
	mean = bnd.GPUArray.zeros(scale.shape, dtype=calctype)
	var = bnd.GPUArray.toGpu(np.ones(scale.shape, dtype=calctype))

	outdata, savemean, saveinvvar = bnd.dnn.batchNormNd(data, mean, var, scale, bias, epsilon=epsilon, out=data)

	hostMean = np.sum(hostData, axis=(0, 2, 3, 4), dtype=calctype, keepdims=True) / norm

	hostInvVar = np.sum((hostData - hostMean) ** 2, axis=(0, 2, 3, 4), dtype=calctype, keepdims=True) / norm
	hostInvVar = 1.0 / np.sqrt(hostInvVar + epsilon)

	hostNormData = (hostData - hostMean) * hostInvVar
	hostOutData = (hostNormData * hostScale + hostBias).astype(dtype)

	assert np.allclose(hostMean.ravel(), mean.get(), atol=atol)
	assert np.allclose(hostInvVar.ravel(), saveinvvar.get(), atol=atol)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad, data = bnd.GPUArray.toGpu(hostGrad), bnd.GPUArray.toGpu(hostData)
	ingrad, scalegrad, biasgrad = bnd.dnn.batchNormNdBackward(grad, data, scale, savemean, saveinvvar, epsilon=epsilon)

	hostScaleGrad = np.sum(hostGrad * hostNormData, axis=(0, 2, 3, 4), dtype=calctype, keepdims=True)
	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3, 4), dtype=calctype, keepdims=True)

	hostMeanGrad = -hostInvVar * hostBiasGrad * hostScale

	hostVarGrad = np.sum(hostGrad * (hostData - hostMean), axis=(0, 2, 3, 4), dtype=calctype, keepdims=True)
	hostVarGrad = -0.5 * hostVarGrad * hostScale * hostInvVar**3

	hostInGrad = hostGrad * hostScale * hostInvVar + (2 * hostVarGrad * (hostData - hostMean) + hostMeanGrad) / norm
	hostInGrad = hostInGrad.astype(dtype)

	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)
	assert np.allclose(hostScaleGrad.ravel(), scalegrad.get(), atol=atol)
	assert np.allclose(hostBiasGrad.ravel(), biasgrad.get(), atol=atol)

	hostMean = np.random.randn(*hostMean.shape).astype(np.float32)
	hostVar = 1.0 + np.random.randn(*hostInvVar.shape).astype(np.float32)**2

	mean, var = bnd.GPUArray.toGpu(hostMean.ravel()), bnd.GPUArray.toGpu(hostVar.ravel())
	outdata = bnd.dnn.batchNormNd(data, mean, var, scale, bias, test=True)

	hostOutData = ((hostData - hostMean) / np.sqrt(hostVar + epsilon) * hostScale + hostBias).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)


def instanceNorm2dTest(bnd, dtype, atol, calctype):
	batchsize, maps, h, w = 3, 4, 5, 5
	epsilon, norm = 1e-5, h * w

	hostData = np.random.randn(batchsize, maps, h, w).astype(dtype)

	hostScale = np.random.randn(1, maps, 1, 1).astype(calctype)
	hostBias = np.random.randn(1, maps, 1, 1).astype(calctype)

	data = bnd.GPUArray.toGpu(hostData)
	scale, bias = bnd.GPUArray.toGpu(hostScale.ravel()), bnd.GPUArray.toGpu(hostBias.ravel())
	outdata, savemean, saveinvvar, extscale = bnd.instanceNorm2d(data, scale, bias, epsilon=epsilon)

	hostExtScale, hostExtBias = np.tile(hostScale, (batchsize, 1, 1, 1)), np.tile(hostBias, (batchsize, 1, 1, 1))

	hostMean = np.mean(hostData, axis=(2, 3), keepdims=True)
	hostInvVar = 1.0 / np.sqrt(np.var(hostData, axis=(2, 3), keepdims=True) + epsilon)

	hostNormData = (hostData - hostMean) * hostInvVar
	hostOutData = hostNormData * hostExtScale + hostExtBias

	assert np.allclose(hostMean.ravel(), savemean.get(), atol=atol)
	assert np.allclose(hostInvVar.ravel(), saveinvvar.get(), atol=atol)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad, scalegrad, bgrad = bnd.instanceNorm2dBackward(grad, data, extscale, savemean, saveinvvar, epsilon=epsilon)

	hostScaleGrad = np.sum(hostGrad * hostNormData, axis=(0, 2, 3), dtype=calctype, keepdims=True)
	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3), dtype=calctype, keepdims=True)

	hostScGrad = hostGrad * hostExtScale
	hostCorrs = np.empty(hostInvVar.shape, dtype=np.float32)

	for b, c in itertools.product(range(batchsize), range(maps)):
		hostCorrs[b, c] = np.dot(hostScGrad[b, c].ravel(), hostNormData[b, c].ravel()) / norm

	hostInGrad = (hostScGrad - np.mean(hostScGrad, axis=(2, 3), keepdims=True) - hostCorrs * hostNormData) * hostInvVar
	hostInGrad = hostInGrad.astype(dtype)

	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)
	assert np.allclose(hostScaleGrad.ravel(), scalegrad.get(), atol=atol)
	assert np.allclose(hostBiasGrad.ravel(), bgrad.get(), atol=atol)


def mapLRN2dTest(bnd, dtype, atol):
	batchsize, maps, h, w = 2, 2, 9, 10
	N, alpha, beta, K = 5, 1.0, 0.5, 2.0

	lookBehind = int((N - 1) / 2)
	lookAhead = N - lookBehind

	hostData = np.random.randn(batchsize, maps, h, w).astype(dtype)

	data = bnd.GPUArray.toGpu(hostData)
	outdata = bnd.dnn.mapLRN(data, N=N, alpha=alpha, beta=beta, K=K)

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
	ingrad = bnd.dnn.mapLRNBackward(data, grad, N=N, alpha=alpha, beta=beta, K=K)

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


def crossMapLRN2dTest(bnd, dtype, atol):
	batchsize, maps, h, w = 2, 10, 2, 3
	N, alpha, beta, K = 5, 1.0, 0.5, 2.0

	lookBehind = int((N - 1) / 2)
	lookAhead = N - lookBehind

	hostData = np.random.randn(batchsize, maps, h, w).astype(dtype)

	data = bnd.GPUArray.toGpu(hostData)
	outdata = bnd.dnn.crossMapLRN(data, N=N, alpha=alpha, beta=beta, K=K)

	norms = np.empty((batchsize, maps, h, w), dtype=np.float32)

	for b, c, y, x in itertools.product(range(batchsize), range(maps), range(h), range(w)):
		slc = hostData[b, max(0, c - lookBehind):min(maps, c + lookAhead), y, x].ravel()
		norms[b, c, y, x] = K + np.dot(slc, slc) * alpha / N

	hostOutData = (hostData / norms**beta).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.crossMapLRNBackward(data, outdata, grad, N=N, alpha=alpha, beta=beta, K=K)

	hostInGrad = hostGrad / norms**beta
	k = 2.0 * alpha * beta / N

	for b, c, y, x in itertools.product(range(batchsize), range(maps), range(h), range(w)):
		slcmaps = slice(max(0, c - lookBehind), min(maps, c + lookAhead))

		slcdata, slcgrad = hostData[b, slcmaps, y, x].ravel(), hostGrad[b, slcmaps, y, x].ravel()
		slcnorms = norms[b, slcmaps, y, x]

		hostInGrad[b, c, y, x] -= k * hostData[b, c, y, x] * np.dot(slcgrad, slcdata / slcnorms**(beta + 1))

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)


if __name__ == "__main__":
	unittest()
