import itertools
import numpy as np


def unittest():
	from PuzzleLib.Cuda import Backend
	backendTest(Backend)


def backendTest(Backend):
	for deviceIdx in range(Backend.getDeviceCount()):
		bnd = Backend.getBackend(deviceIdx, initmode=1)

		for dtype, atol in bnd.dtypesSupported():
			conv2dTest(bnd, dtype, atol)
			conv3dTest(bnd, dtype, atol)
			convGroupTest(bnd, dtype, atol)

			deconv2dTest(bnd, dtype, atol)
			deconv3dTest(bnd, dtype, atol)
			deconvGroupTest(bnd, dtype, atol)

			maxpool2dTest(bnd, dtype, atol)
			maxpool3dTest(bnd, dtype, atol)

			softmax2dTest(bnd, dtype, atol)


def conv2dTest(bnd, dtype, atol):
	batchsize, inmaps, h, w = 1, 2, 6, 6
	outmaps, fsize, stride = 4, 2, 2

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(dtype)
	hostW = np.random.randn(outmaps, inmaps, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = bnd.GPUArray.toGpu(hostData), bnd.GPUArray.toGpu(hostW), bnd.GPUArray.toGpu(hostBias)
	outdata = bnd.dnn.convNd(data, W, bias, stride=stride)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	for b, oc, ic, y, x, dy, dx in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(outdata.shape[2]), range(outdata.shape[3]),
		range(fsize), range(fsize)
	):
		hostOutData[b, oc, y, x] += hostData[b, ic, y * stride + dy, x * stride + dx] * hostW[oc, ic, dy, dx]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.convNdBackwardData(grad, W, data=data, stride=stride)

	hostInGrad = np.zeros(data.shape).astype(np.float32)

	for b, ic, oc, y, x, dy, dx in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(hostGrad.shape[2]), range(hostGrad.shape[3]),
		range(fsize), range(fsize)
	):
		hostInGrad[b, ic, y * stride + dy, x * stride + dx] += hostW[oc, ic, dy, dx] * hostGrad[b, oc, y, x]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = bnd.dnn.convNdBackwardParams(data, grad, W, stride=stride, withbias=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for b, oc, ic, dy, dx, y, x in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(fsize), range(fsize),
		range(hostGrad.shape[2]), range(hostGrad.shape[3])
	):
		hostWGrad[oc, ic, dy, dx] += hostData[b, ic, y * stride + dy, x * stride + dx] * hostGrad[b, oc, y, x]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def conv3dTest(bnd, dtype, atol):
	batchsize, inmaps, d, h, w = 1, 2, 4, 4, 4
	outmaps, fsize, s = 3, 2, 2

	hostData = np.random.randn(batchsize, inmaps, d, h, w).astype(dtype)
	hostW = np.random.randn(outmaps, inmaps, fsize, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = bnd.GPUArray.toGpu(hostData), bnd.GPUArray.toGpu(hostW), bnd.GPUArray.toGpu(hostBias)
	outdata = bnd.dnn.convNd(data, W, bias, stride=s)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	for b, oc, ic, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(outdata.shape[2]),
		range(outdata.shape[3]), range(outdata.shape[4]), range(fsize), range(fsize), range(fsize)
	):
		hostOutData[b, oc, z, y, x] += hostData[b, ic, z * s + dz, y * s + dy, x * s + dx] * hostW[oc, ic, dz, dy, dx]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.convNdBackwardData(grad, W, data=data, stride=s)

	hostInGrad = np.zeros(data.shape).astype(np.float32)

	for b, ic, oc, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(hostGrad.shape[2]),
		range(hostGrad.shape[3]), range(hostGrad.shape[4]), range(fsize), range(fsize), range(fsize)
	):
		hostInGrad[b, ic, z * s + dz, y * s + dy, x * s + dx] += hostW[oc, ic, dz, dy, dx] * hostGrad[b, oc, z, y, x]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = bnd.dnn.convNdBackwardParams(data, grad, W, stride=s, withbias=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for b, oc, ic, dz, dy, dx, z, y, x in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(fsize), range(fsize), range(fsize),
		range(hostGrad.shape[2]), range(hostGrad.shape[3]), range(hostGrad.shape[4])
	):
		hostWGrad[oc, ic, dz, dy, dx] += hostData[b, ic, z * s + dz, y * s + dy, x * s + dx] * hostGrad[b, oc, z, y, x]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3, 4), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def convGroupTest(bnd, dtype, atol):
	batchsize, inmaps, h, w = 5, 6, 3, 4
	outmaps, groups, fsize = 4, 2, 2

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(dtype)
	hostW = np.random.randn(outmaps, inmaps // groups, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = bnd.GPUArray.toGpu(hostData), bnd.GPUArray.toGpu(hostW), bnd.GPUArray.toGpu(hostBias)
	outdata = bnd.dnn.convNd(data, W, bias, groups=groups)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)
	ingroup, outgroup = inmaps // groups, outmaps // groups

	for g in range(groups):
		hostOutGroup = hostOutData[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostGroup = hostData[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, oc, ic, y, x, dy, dx in itertools.product(
			range(batchsize), range(outgroup), range(ingroup), range(outdata.shape[2]), range(outdata.shape[3]),
			range(fsize), range(fsize)
		):
			hostOutGroup[b, oc, y, x] += hostGroup[b, ic, y + dy, x + dx] * hostW[g * outgroup + oc, ic, dy, dx]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.convNdBackwardData(grad, W, groups=groups)

	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for g in range(groups):
		hostGroup = hostGrad[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostInGroup = hostInGrad[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, ic, oc, y, x, dy, dx in itertools.product(
			range(batchsize), range(ingroup), range(outgroup), range(hostGrad.shape[2]), range(hostGrad.shape[3]),
			range(fsize), range(fsize)
		):
			hostInGroup[b, ic, y + dy, x + dx] += hostW[g * outgroup + oc, ic, dy, dx] * hostGroup[b, oc, y, x]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = bnd.dnn.convNdBackwardParams(data, grad, W, groups=groups, withbias=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for g in range(groups):
		hostGrGroup = hostGrad[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostDtGroup = hostData[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, oc, ic, dy, dx, y, x in itertools.product(
			range(batchsize), range(outgroup), range(ingroup), range(fsize), range(fsize),
			range(hostGrad.shape[2]), range(hostGrad.shape[3])
		):
			hostWGrad[g * outgroup + oc, ic, dy, dx] += hostDtGroup[b, ic, y + dy, x + dx] * hostGrGroup[b, oc, y, x]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def deconv2dTest(bnd, dtype, atol):
	batchsize, inmaps, h, w = 1, 1, 2, 2
	outmaps, fsize, stride = 1, 3, 2

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(dtype)
	hostW = np.random.randn(inmaps, outmaps, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = bnd.GPUArray.toGpu(hostData), bnd.GPUArray.toGpu(hostW), bnd.GPUArray.toGpu(hostBias)
	outdata = bnd.dnn.convNdBackwardData(data, W, bias, stride=stride)

	hostOutData = np.zeros(outdata.shape).astype(np.float32)

	for b, oc, ic, y, x, dy, dx in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(h), range(w), range(fsize), range(fsize)
	):
		hostOutData[b, oc, y * stride + dy, x * stride + dx] += hostW[ic, oc, dy, dx] * hostData[b, ic, y, x]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.convNd(grad, W, stride=stride)

	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b, ic, oc, y, x, dy, dx in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(h), range(w), range(fsize), range(fsize)
	):
		hostInGrad[b, ic, y, x] += hostGrad[b, oc, y * stride + dy, x * stride + dx] * hostW[ic, oc, dy, dx]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = bnd.dnn.convNdBackwardParams(grad, data, W, stride=stride, withbias=True, deconv=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for b, ic, oc, dy, dx, y, x in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(fsize), range(fsize), range(h), range(w)
	):
		hostWGrad[ic, oc, dy, dx] += hostGrad[b, oc, y * stride + dy, x * stride + dx] * hostData[b, ic, y, x]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def deconv3dTest(bnd, dtype, atol):
	batchsize, inmaps, d, h, w = 1, 2, 2, 2, 2
	outmaps, fsize, s = 2, 2, 2

	hostData = np.random.randn(batchsize, inmaps, d, h, w).astype(dtype)
	hostW = np.random.randn(inmaps, outmaps, fsize, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = bnd.GPUArray.toGpu(hostData), bnd.GPUArray.toGpu(hostW), bnd.GPUArray.toGpu(hostBias)
	outdata = bnd.dnn.convNdBackwardData(data, W, bias, stride=s)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)

	for b, oc, ic, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(outmaps), range(inmaps), range(d), range(h), range(w),
		range(fsize), range(fsize), range(fsize)
	):
		hostOutData[b, oc, z * s + dz, y * s + dy, x * s + dx] += hostW[ic, oc, dz, dy, dx] * hostData[b, ic, z, y, x]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.convNd(grad, W, stride=s)

	hostInGrad = np.zeros(data.shape, dtype=np.float32)

	for b, ic, oc, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(d), range(h), range(w),
		range(fsize), range(fsize), range(fsize)
	):
		hostInGrad[b, ic, z, y, x] += hostGrad[b, oc, z * s + dz, y * s + dy, x * s + dx] * hostW[ic, oc, dz, dy, dx]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = bnd.dnn.convNdBackwardParams(grad, data, W, stride=s, withbias=True, deconv=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for b, ic, oc, dz, dy, dx, z, y, x in itertools.product(
		range(batchsize), range(inmaps), range(outmaps), range(fsize), range(fsize), range(fsize),
		range(d), range(h), range(w)
	):
		hostWGrad[ic, oc, dz, dy, dx] += hostGrad[b, oc, z * s + dz, y * s + dy, x * s + dx] * hostData[b, ic, z, y, x]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3, 4), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def deconvGroupTest(bnd, dtype, atol):
	batchsize, inmaps, h, w = 3, 4, 3, 4
	outmaps, groups, fsize = 4, 2, 2

	hostData = np.random.randn(batchsize, inmaps, h, w).astype(dtype)
	hostW = np.random.randn(inmaps, outmaps // groups, fsize, fsize).astype(dtype)
	hostBias = np.random.randn(outmaps).astype(dtype)

	data, W, bias = bnd.GPUArray.toGpu(hostData), bnd.GPUArray.toGpu(hostW), bnd.GPUArray.toGpu(hostBias)
	outdata = bnd.dnn.convNdBackwardData(data, W, bias, groups=groups)

	hostOutData = np.zeros(outdata.shape, dtype=np.float32)
	ingroup, outgroup = inmaps // groups, outmaps // groups

	for g in range(groups):
		hostOutGroup = hostOutData[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostGroup = hostData[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, oc, ic, y, x, dy, dx in itertools.product(
			range(batchsize), range(outgroup), range(ingroup), range(data.shape[2]), range(data.shape[3]),
			range(fsize), range(fsize)
		):
			hostOutGroup[b, oc, y + dy, x + dx] += hostW[g * ingroup + ic, oc, dy, dx] * hostGroup[b, ic, y, x]

	hostOutData = (hostOutData + hostBias[np.newaxis, :, np.newaxis, np.newaxis]).astype(dtype)
	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.convNd(grad, W, groups=groups)

	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for g in range(groups):
		hostGroup = hostGrad[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostInGroup = hostInGrad[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, ic, oc, y, x, dy, dx in itertools.product(
			range(batchsize), range(ingroup), range(outgroup), range(hostInGrad.shape[2]), range(hostInGrad.shape[3]),
			range(fsize), range(fsize)
		):
			hostInGroup[b, ic, y, x] += hostGroup[b, oc, y + dy, x + dx] * hostW[g * ingroup + ic, oc, dy, dx]

	hostInGrad = hostInGrad.astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)

	wgrad, bgrad = bnd.dnn.convNdBackwardParams(grad, data, W, groups=groups, withbias=True, deconv=True)
	hostWGrad = np.zeros(wgrad.shape, dtype=np.float32)

	for g in range(groups):
		hostGrGroup = hostGrad[:, g * outgroup:(g + 1) * outgroup, :, :]
		hostDtGroup = hostData[:, g * ingroup:(g + 1) * ingroup, :, :]

		for b, oc, ic, dy, dx, y, x in itertools.product(
			range(batchsize), range(outgroup), range(ingroup), range(fsize), range(fsize),
			range(hostData.shape[2]), range(hostData.shape[3])
		):
			hostWGrad[g * ingroup + ic, oc, dy, dx] += hostDtGroup[b, ic, y, x] * hostGrGroup[b, oc, y + dy, x + dx]

	hostWGrad = hostWGrad.astype(dtype)
	assert np.allclose(hostWGrad, wgrad.get(), atol=atol)

	hostBiasGrad = np.sum(hostGrad, axis=(0, 2, 3), dtype=np.float32).astype(dtype)
	assert np.allclose(hostBiasGrad, bgrad.get())


def maxpool2dTest(bnd, dtype, atol):
	batchsize, maps, h, w = 3, 2, 6, 6
	size, stride, pad = 3, 2, 1

	hostData = np.full(shape=(batchsize, maps, h + 2 * pad, w + 2 * pad), fill_value=np.finfo(dtype).min, dtype=dtype)
	hostData[:, :, pad:-pad, pad:-pad] = np.random.randn(batchsize, maps, h, w).astype(dtype)

	data = bnd.GPUArray.toGpu(hostData[:, :, pad:-pad, pad:-pad])
	outdata = bnd.dnn.poolNd(data, size=size, stride=stride, pad=pad, mode=bnd.PoolMode.max.value)

	hostOutData = np.empty(outdata.shape, dtype=dtype)

	for b, c, y, x in itertools.product(
		range(batchsize), range(maps), range(hostOutData.shape[2]), range(hostOutData.shape[3])
	):
		hostOutData[b, c, y, x] = np.max(hostData[b, c, y * stride:y * stride + size, x * stride:x * stride + size])

	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.poolNdBackward(grad, data, outdata, size=size, stride=stride, pad=pad, mode=bnd.PoolMode.max.value)

	hostInGrad = np.zeros(hostData.shape, dtype=dtype)

	for b, c, y, x, dy, dx in itertools.product(
		range(batchsize), range(maps), range(hostOutData.shape[2]), range(hostOutData.shape[3]),
		range(size), range(size)
	):
		if hostData[b, c, y * stride + dy, x * stride + dx] == hostOutData[b, c, y, x]:
			hostInGrad[b, c, y * stride + dy, x * stride + dx] += hostGrad[b, c, y, x]

	hostInGrad = hostInGrad[:, :, pad:-pad, pad:-pad].astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)


def maxpool3dTest(bnd, dtype, atol):
	batchsize, maps, d, h, w = 1, 1, 6, 6, 6
	size, s, pad = 3, 2, 1

	hostData = np.full(
		shape=(batchsize, maps, d + 2 * pad, h + 2 * pad, w + 2 * pad), fill_value=np.finfo(dtype).min, dtype=dtype
	)
	hostData[:, :, pad:-pad, pad:-pad, pad:-pad] = np.random.randn(batchsize, maps, d, h, w).astype(dtype)

	data = bnd.GPUArray.toGpu(np.ascontiguousarray(hostData[:, :, pad:-pad, pad:-pad, pad:-pad]))
	outdata = bnd.dnn.poolNd(data, size=size, stride=s, pad=pad, mode=bnd.PoolMode.max.value)

	hostOutData = np.empty(outdata.shape, dtype=dtype)

	for b, c, z, y, x in itertools.product(
		range(batchsize), range(maps),
		range(hostOutData.shape[2]), range(hostOutData.shape[3]), range(hostOutData.shape[4])
	):
		hostOutData[b, c, z, y, x] = np.max(hostData[b, c, z * s:z * s + size, y * s:y * s + size, x * s:x * s + size])

	assert np.allclose(hostOutData, outdata.get())

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.poolNdBackward(grad, data, outdata, size=size, stride=s, pad=pad, mode=bnd.PoolMode.max.value)

	hostInGrad = np.zeros(hostData.shape, dtype=np.float32)

	for b, c, z, y, x, dz, dy, dx in itertools.product(
		range(batchsize), range(maps),
		range(hostOutData.shape[2]), range(hostOutData.shape[3]), range(hostOutData.shape[4]),
		range(size), range(size), range(size)
	):
		if hostData[b, c, z * s + dz, y * s + dy, x * s + dx] == hostOutData[b, c, z, y, x]:
			hostInGrad[b, c, z * s + dz, y * s + dy, x * s + dx] += hostGrad[b, c, z, y, x]

	hostInGrad = hostInGrad[:, :, pad:-pad, pad:-pad, pad:-pad].astype(dtype)
	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)


def softmax2dTest(bnd, dtype, atol):
	batchsize, maps, h, w = 5, 8, 2, 3
	hostData = np.random.randn(batchsize, maps, h, w).astype(dtype)

	data = bnd.GPUArray.toGpu(hostData)
	outdata = bnd.dnn.softmaxNd(data)

	def hostSoftmax(tensor):
		e = np.exp(tensor - np.amax(tensor))
		return e / np.sum(e)

	hostOutData = np.empty(outdata.shape, dtype=dtype)

	for b, y, x in itertools.product(range(batchsize), range(h), range(w)):
		hostOutData[b, :, y, x] = hostSoftmax(hostData[b, :, y, x])

	assert np.allclose(hostOutData, outdata.get(), atol=atol)

	hostGrad = np.random.randn(*outdata.shape).astype(dtype)

	grad = bnd.GPUArray.toGpu(hostGrad)
	ingrad = bnd.dnn.softmaxNdBackward(grad, outdata)

	hostInGrad = np.empty(ingrad.shape, dtype=dtype)

	def hostSoftmaxBackward(d, gr):
		return d * (gr - np.dot(d, gr))

	for b, y, x in itertools.product(range(batchsize), range(h), range(w)):
		hostInGrad[b, :, y, x] = hostSoftmaxBackward(hostOutData[b, :, y, x], hostGrad[b, :, y, x])

	assert np.allclose(hostInGrad, ingrad.get(), atol=atol)


if __name__ == "__main__":
	unittest()
