import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Modules.BatchNormND import BatchNormND


class BatchNorm3D(BatchNormND):
	def __init__(self, maps, epsilon=1e-5, initFactor=1.0, minFactor=0.1, sscale=0.01, affine=True, name=None,
				 empty=False, inplace=False):
		super().__init__(3, maps, epsilon, initFactor, minFactor, sscale, affine, name, empty, inplace)
		self.registerBlueprint(locals())


	def checkDataShape(self, shape):
		if len(shape) != 5:
			raise ModuleError("Data must be 5d tensor")

		_, maps, _, _, _ = shape
		if maps != self.maps:
			raise ModuleError("Data has %d maps (expected: %d)" % (maps, self.maps))


	def checkGradShape(self, shape):
		if len(shape) != 5:
			raise ModuleError("Grad must be 5d tensor")

		_, maps, _, _, _ = shape
		if maps != self.maps:
			raise ModuleError("Grad has %d maps (expected: %d)" % (maps, self.maps))


def unittest():
	batchsize, maps, d, h, w = 8, 5, 3, 4, 2
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, d, h, w).astype(np.float32))

	bn = BatchNorm3D(maps)
	bn(data)

	hostData, hostScale, hostBias = data.get(), bn.scale.get(), bn.bias.get()
	hostNormData, hostOutData = np.empty(hostData.shape, dtype=np.float32), np.empty(hostData.shape, dtype=np.float32)
	hostMean, hostInvVar = np.zeros(hostScale.shape, dtype=np.float32), np.zeros(hostScale.shape, dtype=np.float32)
	for c in range(maps):
		for b in range(batchsize):
			hostMean[0, c, 0, 0, 0] += np.sum(hostData[b, c])
		hostMean[0, c, 0, 0, 0] /= (batchsize * w * h * d)

		for b in range(batchsize):
			hostInvVar[0, c, 0, 0, 0] += np.sum((hostData[b, c] - hostMean[0, c, 0, 0, 0])**2)
		hostInvVar[0, c, 0, 0, 0] /= (batchsize * w * h * d)

		hostInvVar[0, c, 0, 0, 0] = 1.0 / np.sqrt(hostInvVar[0, c, 0, 0, 0] + bn.epsilon)
		hostNormData[:, c, :, :, :] = (hostData[:, c, :, :, :] - hostMean[0, c, 0, 0, 0]) * hostInvVar[0, c, 0, 0, 0]
		hostOutData[:, c, :, :, :] = hostNormData[:, c, :, :, :] * hostScale[0, c, 0, 0, 0] + hostBias[0, c, 0, 0, 0]

	assert np.allclose(hostMean, bn.mean.get())
	assert np.allclose(hostInvVar, bn.saveinvvar.get())
	assert np.allclose(hostOutData, bn.data.get())

	grad = gpuarray.to_gpu(np.random.randn(batchsize, maps, d, h, w).astype(np.float32))
	bn.backward(grad)

	hostGrad, hostInGrad = grad.get(), np.empty_like(hostData)
	hostScaleGrad, hostBiasGrad = np.empty_like(hostScale), np.empty_like(hostBias)
	hostMeanGrad, hostVarGrad = np.empty_like(hostMean), np.empty_like(hostInvVar)
	for c in range(maps):
		hostBiasGrad[0, c, 0, 0, 0] = np.sum(hostGrad[:, c, :, :, :])
		hostScaleGrad[0, c, 0, 0, 0] = np.sum(hostGrad[:, c, :, :, :] * hostNormData[:, c, :, :, :])

		hostMeanGrad[0, c, 0, 0, 0] = np.sum(hostGrad[:,c,:,:,:]) * hostScale[0,c,0,0,0] * -hostInvVar[0,c,0,0,0]
		hostVarGrad[0, c, 0, 0, 0] = np.sum(hostGrad[:,c,:,:,:] * (hostData[:,c,:,:,:] - hostMean[0,c,0,0,0])) * \
									 hostScale[0, c, 0, 0, 0] * -0.5 * hostInvVar[0, c, 0, 0, 0]**3

		hostInGrad[:, c, :, :, :] = hostGrad[:,c,:,:,:] * hostScale[0,c,0,0,0] * hostInvVar[0,c,0,0,0] + \
									hostVarGrad[0, c, 0, 0, 0] * 2.0 / (batchsize * w * h * d) * \
									(hostData[:, c, :, :, :] - hostMean[0, c, 0, 0, 0]) + \
									hostMeanGrad[0, c, 0, 0, 0] / (batchsize * w * h * d)

	assert np.allclose(hostInGrad, bn.grad.get())
	assert np.allclose(hostScaleGrad, bn.vars["scale"].grad.get())
	assert np.allclose(hostBiasGrad, bn.vars["bias"].grad.get())


if __name__ == "__main__":
	unittest()
