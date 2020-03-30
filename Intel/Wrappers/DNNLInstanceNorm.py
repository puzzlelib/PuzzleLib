import numpy as np

from PuzzleLib.CPU.CPUArray import CPUArray
from PuzzleLib.Intel.Wrappers import DNNL


def instanceNorm2d(data, scale, bias, epsilon=1e-5):
	batchsize = data.shape[0]
	if batchsize > 1:
		extscale = CPUArray.toDevice(np.tile(scale.data, (batchsize, 1, 1)))
		extbias = CPUArray.toDevice(np.tile(bias.data, (batchsize, 1, 1)))

	else:
		extscale, extbias = scale, bias

	indata = data.reshape(1, batchsize * data.shape[1], data.shape[2], data.shape[3])
	mean = CPUArray.empty((1, indata.shape[1], 1, 1), dtype=np.float32)
	var = CPUArray.empty((1, indata.shape[1], 1, 1), dtype=np.float32)

	outdata, savemean, savevar, desc = DNNL.batchNormNd(indata, extscale, extbias, mean, var, epsilon, test=False)
	return outdata.reshape(data.shape), savemean, savevar, extscale, extbias, desc


def instanceNorm2dBackward(grad, data, extscale, extbias, savemean, savevar, epsilon, desc, affine=True):
	batchsize, maps = grad.shape[:2]

	outgrad = grad.reshape(1, batchsize * grad.shape[1], grad.shape[2], grad.shape[3])
	indata = data.reshape(1, batchsize * data.shape[1], data.shape[2], data.shape[3])

	ingrad, scalegrad, biasgrad = DNNL.batchNormNdBackward(
		indata, outgrad, extscale, extbias, savemean, savevar, desc, epsilon
	)

	if affine and batchsize > 1:
		scalegrad = np.sum(scalegrad.data.reshape(batchsize, -1), axis=0).reshape((1, maps, 1, 1))
		biasgrad = np.sum(biasgrad.data.reshape(batchsize, -1), axis=0).reshape((1, maps, 1, 1))

		scalegrad = CPUArray(scalegrad.shape, scalegrad.dtype, data=scalegrad, acquire=True)
		biasgrad = CPUArray(biasgrad.shape, biasgrad.dtype, data=biasgrad, acquire=True)

	return (ingrad.reshape(grad.shape), scalegrad, biasgrad) if affine else ingrad.reshape(grad.shape)


def unittest():
	batchsize, maps, h, w = 3, 4, 5, 5
	epsilon = 1e-5

	data = CPUArray.toDevice(np.random.randn(batchsize, maps, h, w).astype(np.float32))
	scale = CPUArray.toDevice(np.random.randn(1, maps, 1, 1).astype(np.float32))
	bias = CPUArray.toDevice(np.random.randn(1, maps, 1, 1).astype(np.float32))

	outdata, savemean, savevar, extscale, extbias, desc = instanceNorm2d(data, scale, bias, epsilon)

	hostData = data.get().reshape(data.shape[0] * data.shape[1], -1)
	hostScale, hostBias = scale.get().reshape(maps, 1), bias.get().reshape(maps, 1)
	hostExtScale, hostExtBias = np.tile(hostScale, (batchsize, 1)), np.tile(hostBias, (batchsize, 1))

	hostMean = np.mean(hostData, axis=1, keepdims=True)
	hostVar = np.var(hostData, axis=1)
	hostInvVar = 1.0 / np.sqrt(hostVar + epsilon)
	hostOutData = (hostData - hostMean) * hostInvVar[:, np.newaxis]
	hostOutScData = hostOutData * hostExtScale + hostExtBias

	assert np.allclose(hostOutScData.reshape(data.shape), outdata.get())
	assert np.allclose(hostMean.reshape(savemean.shape), savemean.get())
	assert np.allclose(hostVar.reshape(savevar.shape), savevar.get())

	grad = CPUArray.toDevice(np.random.randn(batchsize, maps, h, w).astype(np.float32))
	ingrad, scalegrad, bgrad = instanceNorm2dBackward(grad, data, extscale, extbias, savemean, savevar, epsilon, desc)

	hostGrad = grad.get().reshape(grad.shape[0] * grad.shape[1], -1)
	hostScGrad = hostGrad * hostExtScale
	hostCorrs = np.empty(hostInvVar.shape, dtype=np.float32)
	for i in range(hostCorrs.shape[0]):
		hostCorrs[i] = np.dot(hostScGrad[i], hostOutData[i]) / hostScGrad.shape[1]
	hostInGrad = hostScGrad - np.mean(hostScGrad, axis=1, keepdims=True) - hostCorrs[:, np.newaxis] * hostOutData
	hostInGrad *= hostInvVar[:, np.newaxis]

	hostScaleGrad = np.sum(np.sum(hostOutData * hostGrad, axis=1).reshape(batchsize, -1), axis=0)
	hostBiasGrad = np.sum(np.sum(hostGrad, axis=1).reshape(batchsize, -1), axis=0)

	assert np.allclose(hostInGrad.reshape(grad.shape), ingrad.get())
	assert np.allclose(hostScaleGrad.reshape((1, maps, 1, 1)), scalegrad.get())
	assert np.allclose(hostBiasGrad.reshape((1, maps, 1, 1)), bgrad.get())


if __name__ == "__main__":
	unittest()
