import numpy as np

from PuzzleLib.CPU.CPUArray import CPUArray
from PuzzleLib.Intel.Wrappers.DNNL import convNd, convNdBackwardData, convNdBackwardParams, \
	PoolMode, poolNd, poolNdBackward, batchNormNd, batchNormNdBackward


def unittest():
	conv3dTest()
	maxpool3dTest()
	batchNorm3dTest()


def conv3dTest():
	batchsize, inmaps, d, h, w = 1, 2, 3, 3, 3
	outmaps, fsize = 3, 2

	data = CPUArray.toDevice(np.random.randn(batchsize, inmaps, d, h, w).astype(np.float32))

	W = CPUArray.toDevice(np.random.randn(outmaps, inmaps, fsize, fsize, fsize).astype(np.float32))
	bias = CPUArray.toDevice(np.random.randn(1, outmaps, 1, 1, 1).astype(np.float32))

	outdata = convNd(data, W, bias)

	hostData = data.get()
	hostOutData = np.empty(outdata.shape, dtype=np.float32)

	hostW, hostBias = W.get(), bias.get()

	for c in range(outmaps):
		hostOutData[:, c, :, :, :] = hostBias[0, c, 0, 0, 0]

	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for z in range(outdata.shape[2]):
					for y in range(outdata.shape[3]):
						for x in range(outdata.shape[4]):
							for dz in range(fsize):
								for dy in range(fsize):
									for dx in range(fsize):
										hostOutData[b, oc, z, y, x] += hostData[b, ic, z + dz, y + dy, x + dx] * \
																	   hostW[oc, ic, dz, dy, dx]

	assert np.allclose(hostOutData, outdata.get())

	grad = CPUArray.toDevice(np.random.randn(*outdata.shape).astype(np.float32))
	ingrad = convNdBackwardData(grad, W)

	hostGrad, hostInGrad = grad.get(), np.zeros(data.shape, dtype=np.float32)

	for b in range(batchsize):
		for ic in range(inmaps):
			for oc in range(outmaps):
				for z in range(grad.shape[2]):
					for y in range(grad.shape[3]):
						for x in range(grad.shape[4]):
							for dz in range(fsize):
								for dy in range(fsize):
									for dx in range(fsize):
										hostInGrad[b, ic, z + dz, y + dy, x + dx] += hostW[oc, ic, dz, dy, dx] * \
																					 hostGrad[b, oc, z, y, x]

	assert np.allclose(hostInGrad, ingrad.get())

	wgrad, bgrad = convNdBackwardParams(data, grad, W, bias)

	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)
	for b in range(batchsize):
		for oc in range(outmaps):
			for ic in range(inmaps):
				for dz in range(fsize):
					for dy in range(fsize):
						for dx in range(fsize):
							for z in range(grad.shape[2]):
								for y in range(grad.shape[3]):
									for x in range(grad.shape[4]):
										hostWGrad[oc, ic, dz, dy, dx] += hostData[b, ic, z + dz, y + dy, x + dx] * \
																		 hostGrad[b, oc, z, y, x]

	assert np.allclose(hostWGrad, wgrad.get())

	hostBGrad = np.empty(bias.shape, dtype=np.float32)
	for oc in range(outmaps):
		hostBGrad[0, oc, 0, 0, 0] = np.sum(hostGrad[:, oc, :, :, :])

	assert np.allclose(hostBGrad, bgrad.get())


def maxpool3dTest():
	batchsize, maps, d, h, w = 1, 1, 6, 6, 6
	size, stride, pad = 3, 2, 1

	data = CPUArray.toDevice(np.random.randn(batchsize, maps, d, h, w).astype(np.float32))

	outdata, workspace, desc = poolNd(data, size=size, stride=stride, pad=pad, mode=PoolMode.max)

	hostData = np.full(shape=(batchsize, maps, d + 2 * pad, h + 2 * pad, w + 2 * pad),
					   fill_value=np.finfo(np.float32).min, dtype=np.float32)
	hostData[:, :, pad:-pad, pad:-pad, pad:-pad] = data.get()
	hostOutData = np.empty(outdata.shape)

	for b in range(batchsize):
		for c in range(maps):
			for z in range(hostOutData.shape[2]):
				for y in range(hostOutData.shape[3]):
					for x in range(hostOutData.shape[4]):
						hostOutData[b, c, z, y, x] = np.max(hostData[b, c, z * stride:z * stride + size,
															y * stride:y*stride + size, x * stride:x * stride + size])

	assert np.allclose(hostOutData, outdata.get())

	grad = CPUArray.toDevice(np.random.randn(*outdata.shape).astype(np.float32))
	ingrad = poolNdBackward(data, grad, workspace, desc, size=size, stride=stride, pad=pad, mode=PoolMode.max)

	hostGrad = grad.get()
	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b in range(batchsize):
		for c in range(maps):
			for z in range(hostOutData.shape[2]):
				for y in range(hostOutData.shape[3]):
					for x in range(hostOutData.shape[4]):
						for dz in range(size):
							for dy in range(size):
								for dx in range(size):
									if hostData[b,c,z*stride+dz,y*stride+dy,x*stride+dx] == hostOutData[b, c, z, y, x]:
										hostInGrad[b,c,z*stride + dz,y*stride + dy,x*stride + dx] += hostGrad[b,c,z,y,x]

	assert np.allclose(hostInGrad[:, :, pad:-pad, pad:-pad, pad:-pad], ingrad.get())


def batchNorm3dTest():
	batchsize, maps, d, h, w = 2, 5, 2, 3, 2
	data = CPUArray.toDevice(np.random.randn(batchsize, maps, d, h, w).astype(np.float32))
	hostData = data.get()

	scale = CPUArray.toDevice(np.random.randn(1, maps, 1, 1, 1).astype(np.float32))
	bias = CPUArray.toDevice(np.random.randn(*scale.shape).astype(np.float32))
	mean = CPUArray.zeros(scale.shape, dtype=np.float32)
	var = CPUArray.toDevice(np.ones(scale.shape, dtype=np.float32))

	outdata = CPUArray.copy(data)
	outdata, savemean, savevar, desc = batchNormNd(outdata, scale, bias, mean, var, out=outdata)

	hostScale, hostBias = scale.get(), bias.get()
	hostNormData = np.empty(data.shape, dtype=np.float32)
	hostOutData = np.empty(data.shape, dtype=np.float32)
	hostMean = np.zeros(scale.shape, dtype=np.float32)
	hostVar = np.zeros(scale.shape, dtype=np.float32)
	hostInvVar = np.zeros(scale.shape, dtype=np.float32)
	for c in range(maps):
		for b in range(batchsize):
			hostMean[0, c, 0, 0, 0] += np.sum(hostData[b, c])
		hostMean[0, c, 0, 0, 0] /= (batchsize * w * h * d)

		for b in range(batchsize):
			hostVar[0, c, 0, 0, 0] += np.sum((hostData[b, c] - hostMean[0, c, 0, 0, 0])**2)
		hostVar[0, c, 0, 0, 0] /= (batchsize * w * h * d)

		hostInvVar[0, c, 0, 0, 0] = 1.0 / np.sqrt(hostVar[0, c, 0, 0, 0] + 1e-5)
		hostNormData[:, c, :, :, :] = (hostData[:, c, :, :, :] - hostMean[0, c, 0, 0, 0]) * hostInvVar[0, c, 0, 0, 0]
		hostOutData[:, c, :, :, :] = hostNormData[:, c, :, :, :] * hostScale[0, c, 0, 0, 0] + hostBias[0, c, 0, 0, 0]

	assert np.allclose(hostMean, mean.get())
	assert np.allclose(hostVar, savevar.get())
	assert np.allclose(hostOutData, outdata.get())

	grad = CPUArray.toDevice(np.random.randn(batchsize, maps, d, h, w).astype(np.float32))

	ingrad, scalegrad, biasgrad = batchNormNdBackward(data, grad, scale, bias, savemean, savevar, desc)

	hostGrad = grad.get()
	hostInGrad, hostScaleGrad = np.empty(hostGrad.shape, dtype=np.float32), np.empty(hostScale.shape, dtype=np.float32)
	hostBiasGrad, hostMeanGrad = np.empty(hostBias.shape, dtype=np.float32), np.empty(hostMean.shape, dtype=np.float32)
	hostVarGrad = np.empty(hostInvVar.shape, dtype=np.float32)
	for c in range(maps):
		hostBiasGrad[0, c, 0, 0, 0] = np.sum(hostGrad[:, c, :, :, :])
		hostScaleGrad[0, c, 0, 0, 0] = np.sum(hostGrad[:, c, :, :, :] * hostNormData[:, c, :, :, :])

		hostMeanGrad[0, c, 0, 0, 0] = np.sum(hostGrad[:, c, :, :, :]) * hostScale[0,c,0,0,0] * -hostInvVar[0,c,0,0,0]
		hostVarGrad[0, c, 0, 0, 0] = np.sum(hostGrad[:, c, :, :, :] * (hostData[:,c,:,:,:] - hostMean[0,c,0,0,0])) * \
									 hostScale[0, c, 0, 0, 0] * -0.5 * hostInvVar[0, c, 0, 0, 0]**3

		hostInGrad[:, c, :, :, :] = hostGrad[:, c, :, :, :] * hostScale[0, c, 0, 0, 0] * hostInvVar[0, c, 0, 0, 0] + \
									hostVarGrad[0, c, 0, 0, 0] * 2 / (batchsize * w * h * d) * \
									(hostData[:, c, :, :, :] - hostMean[0, c, 0, 0, 0]) + \
									hostMeanGrad[0, c, 0, 0, 0] / (batchsize * w * h * d)
	assert np.allclose(hostInGrad, ingrad.get())
	assert np.allclose(hostScaleGrad, scalegrad.get())
	assert np.allclose(hostBiasGrad, biasgrad.get())

	batchNormNd(data, scale, bias, mean, var, test=True)


if __name__ == "__main__":
	unittest()
