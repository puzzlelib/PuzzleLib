import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL import Utils
from PuzzleLib.OpenCL.Utils import queue, memoryPool as memPool
from PuzzleLib.OpenCL.Wrappers import CLBlas, MIOpen


def instanceNorm2d(data, scale, bias, epsilon=1e-5):
	batchsize = data.shape[0]
	if batchsize > 1:
		extscale = Utils.tile(scale, batchsize, axis=1)
		extbias = Utils.tile(bias, batchsize, axis=1)

	else:
		extscale = scale
		extbias = bias

	indata = data.reshape(1, batchsize * data.shape[1], data.shape[2], data.shape[3])
	mean = Driver.empty(queue, (1, indata.shape[1], 1, 1), dtype=np.float32, allocator=memPool)
	var = Driver.empty(queue, (1, indata.shape[1], 1, 1), dtype=np.float32, allocator=memPool)

	outdata, savemean, saveinvvar = MIOpen.batchNorm2d(indata, extscale, extbias, mean, var, epsilon, test=False)
	return outdata.reshape(data.shape), savemean, saveinvvar, extscale


def instanceNorm2dBackward(grad, data, extscale, savemean, saveinvvar, epsilon, affine=True):
	batchsize, maps = grad.shape[:2]

	outgrad = grad.reshape(1, batchsize * grad.shape[1], grad.shape[2], grad.shape[3])
	indata = data.reshape(1, batchsize * data.shape[1], data.shape[2], data.shape[3])

	ingrad, scalegrad, bgrad = MIOpen.batchNorm2dBackward(indata, outgrad, extscale, savemean, saveinvvar, epsilon)

	if affine and batchsize > 1:
		scalegrad = CLBlas.sumOnMatrix(scalegrad.reshape(batchsize, -1)).reshape(1, maps, 1, 1)
		bgrad = CLBlas.sumOnMatrix(bgrad.reshape(batchsize, -1)).reshape(1, maps, 1, 1)

	if affine:
		return ingrad.reshape(grad.shape), scalegrad, bgrad
	else:
		return ingrad.reshape(grad.shape)


def unittest():
	batchsize, maps, h, w = 3, 4, 5, 5
	epsilon = 1e-5

	data = Driver.to_device(queue, np.random.randn(batchsize, maps, h, w).astype(np.float32))
	scale = Driver.to_device(queue, np.random.randn(1, maps, 1, 1).astype(np.float32))
	bias = Driver.to_device(queue, np.random.randn(1, maps, 1, 1).astype(np.float32))

	outdata, savemean, saveinvvar, extscale = instanceNorm2d(data, scale, bias, epsilon)

	hostData = data.get().reshape(data.shape[0] * data.shape[1], -1)
	hostScale, hostBias = scale.get().reshape(maps, 1), bias.get().reshape(maps, 1)
	hostExtScale, hostExtBias = np.tile(hostScale, (batchsize, 1)), np.tile(hostBias, (batchsize, 1))

	hostMean = np.mean(hostData, axis=1, keepdims=True)
	hostInvVar = 1.0 / np.sqrt(np.var(hostData, axis=1) + epsilon)
	hostOutData = (hostData - hostMean) * hostInvVar[:, np.newaxis]
	hostOutScData = hostOutData * hostExtScale + hostExtBias

	assert np.allclose(hostOutScData.reshape(data.shape), outdata.get())
	assert np.allclose(hostMean.reshape(savemean.shape), savemean.get())
	assert np.allclose(hostInvVar.reshape(saveinvvar.shape), saveinvvar.get())

	grad = Driver.to_device(queue, np.random.randn(batchsize, maps, h, w).astype(np.float32))
	ingrad, scalegrad, bgrad = instanceNorm2dBackward(grad, data, extscale, savemean, saveinvvar, epsilon)

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
	assert np.allclose(hostScaleGrad.reshape(1, maps, 1, 1), scalegrad.get())
	assert np.allclose(hostBiasGrad.reshape(1, maps, 1, 1), bgrad.get())


if __name__ == "__main__":
	unittest()
