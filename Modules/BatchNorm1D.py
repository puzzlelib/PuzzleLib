import numpy as np

from PuzzleLib.Backend import gpuarray

from PuzzleLib.Modules.Module import ModuleError
from PuzzleLib.Modules.BatchNormND import BatchNormND


class BatchNorm1D(BatchNormND):
	def __init__(self, maps, epsilon=1e-5, initFactor=1.0, minFactor=0.1, sscale=0.01, affine=True, name=None,
				 empty=False, inplace=False):
		super().__init__(2, maps, epsilon, initFactor, minFactor, sscale, affine, name, empty, inplace)
		self.registerBlueprint(locals())


	def updateData(self, data):
		data = data.reshape(*data.shape[:2], 1, *data.shape[2:])
		super().updateData(data)
		self.data = self.data.reshape(*self.data.shape[:2], *self.data.shape[3:])


	def updateGrad(self, grad):
		grad = grad.reshape(*grad.shape[:2], 1, *grad.shape[2:])

		data = self.inData
		self.inData = data.reshape(*data.shape[:2], 1, *data.shape[2:])
		super().updateGrad(grad)
		self.inData = data

		self.grad = self.grad.reshape(*self.grad.shape[:2], *self.grad.shape[3:])


	def accGradParams(self, grad, scale=1.0, momentum=0.0):
		grad = grad.reshape(*grad.shape[:2], 1, *grad.shape[2:])

		data = self.inData
		self.inData = data.reshape(*data.shape[:2], 1, *data.shape[2:])
		super().accGradParams(grad, scale, momentum)
		self.inData = data


	def checkDataShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Data must be 3d tensor")

		_, maps, _ = shape
		if maps != self.maps:
			raise ModuleError("Data has %d maps (expected: %d)" % (maps, self.maps))


	def checkGradShape(self, shape):
		if len(shape) != 3:
			raise ModuleError("Grad must be 3d tensor")

		_, maps, _ = shape
		if maps != self.maps:
			raise ModuleError("Grad has %d maps (expected: %d)" % (maps, self.maps))


def unittest():
	batchsize, maps, size = 16, 5, 4
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, size).astype(np.float32))

	bn = BatchNorm1D(maps)
	bn(data)

	hostData, hostScale, hostBias = data.get(), bn.scale.get(), bn.bias.get()
	hostNormData, hostOutData = np.empty(hostData.shape, dtype=np.float32), np.empty(hostData.shape, dtype=np.float32)
	hostMean, hostInvVar = np.zeros(hostScale.shape, dtype=np.float32), np.zeros(hostScale.shape, dtype=np.float32)
	for c in range(maps):
		for b in range(batchsize):
			hostMean[0, c, 0] += np.sum(hostData[b, c])
		hostMean[0, c, 0] /= (batchsize * size)

		for b in range(batchsize):
			hostInvVar[0, c, 0] += np.sum((hostData[b, c] - hostMean[0, c, 0])**2)
		hostInvVar[0, c, 0] /= (batchsize * size)

		hostInvVar[0, c, 0] = 1.0 / np.sqrt(hostInvVar[0, c, 0] + bn.epsilon)
		hostNormData[:, c, :] = (hostData[:, c, :] - hostMean[0, c, 0]) * hostInvVar[0, c, 0]
		hostOutData[:, c, :] = hostNormData[:, c, :] * hostScale[0, c, 0] + hostBias[0, c, 0]

	assert np.allclose(hostMean, bn.mean.get())
	assert np.allclose(hostInvVar, bn.saveinvvar.get())
	assert np.allclose(hostOutData, bn.data.get())

	grad = gpuarray.to_gpu(np.random.randn(batchsize, maps, size).astype(np.float32))
	bn.backward(grad)

	hostGrad, hostInGrad = grad.get(), np.empty_like(hostData)
	hostScaleGrad, hostBiasGrad = np.empty_like(hostScale), np.empty_like(hostBias)
	hostMeanGrad, hostVarGrad = np.empty_like(hostMean), np.empty_like(hostInvVar)
	for c in range(maps):
		hostBiasGrad[0, c, 0] = np.sum(hostGrad[:, c, :])
		hostScaleGrad[0, c, 0] = np.sum(hostGrad[:, c, :] * hostNormData[:, c, :])

		hostMeanGrad[0, c, 0] = np.sum(hostGrad[:, c, :]) * hostScale[0, c, 0] * -hostInvVar[0, c, 0]
		hostVarGrad[0, c, 0] = np.sum(hostGrad[:, c, :] * (hostData[:, c, :] - hostMean[0, c, 0])) * \
							   hostScale[0, c, 0] * -0.5 * hostInvVar[0, c, 0]**3

		hostInGrad[:, c, :] = hostGrad[:, c, :] * hostScale[0, c, 0] * hostInvVar[0, c, 0] + \
							  hostVarGrad[0, c, 0] * 2.0 / (batchsize * size) * \
							  (hostData[:, c, :] - hostMean[0, c, 0]) + \
							  hostMeanGrad[0, c, 0] / (batchsize * size)

	assert np.allclose(hostInGrad, bn.grad.get())
	assert np.allclose(hostScaleGrad, bn.vars["scale"].grad.get())
	assert np.allclose(hostBiasGrad, bn.vars["bias"].grad.get())


if __name__ == "__main__":
	unittest()
