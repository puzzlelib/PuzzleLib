import numpy as np

from PuzzleLib.Backend import gpuarray
from PuzzleLib.Backend.Dnn.Basic import PoolMode, poolNd, poolNdBackward

from PuzzleLib.Modules.Pool3D import Pool3D


class MaxPool3D(Pool3D):
	def __init__(self, size=2, stride=2, pad=0, name=None):
		super().__init__(size, stride, pad, name)
		self.registerBlueprint(locals())

		self.mode = PoolMode.max


	def updateData(self, data):
		self.data, self.workspace = poolNd(data, size=self.size, stride=self.stride, pad=self.pad, mode=self.mode,
										   test=not self.train)


	def updateGrad(self, grad):
		self.grad = poolNdBackward(self.inData, self.data, grad, self.workspace,
								   size=self.size, stride=self.stride, pad=self.pad, mode=self.mode)


def unittest():
	batchsize, maps, t, h, w = 1, 1, 6, 6, 6
	size, stride, pad = 3, 2, 1
	data = gpuarray.to_gpu(np.random.randn(batchsize, maps, t, h, w).astype(np.float32))

	maxpool3d = MaxPool3D(size=size, stride=stride, pad=pad)
	maxpool3d(data)

	hostData = np.full(shape=(batchsize, maps, t + 2 * pad, h + 2 * pad, w + 2 * pad),
					   fill_value=np.finfo(np.float32).min, dtype=np.float32)
	hostData[:, :, pad:-pad, pad:-pad, pad:-pad] = data.get()
	hostOutData = np.empty(maxpool3d.data.shape)

	for b in range(batchsize):
		for c in range(maps):
			for z in range(hostOutData.shape[2]):
				for y in range(hostOutData.shape[3]):
					for x in range(hostOutData.shape[4]):
						hostOutData[b, c, z, y, x] = np.max(hostData[b, c, z * stride:z * stride + size,
															y * stride:y * stride + size, x * stride:x * stride + size])

	assert np.allclose(hostOutData, maxpool3d.data.get())

	grad = gpuarray.to_gpu(np.random.randn(*maxpool3d.data.shape).astype(np.float32))
	maxpool3d.backward(grad)

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
									if hostData[b,c,z*stride+dz,y*stride + dy,x*stride + dx] == hostOutData[b,c,z,y,x]:
										hostInGrad[b,c,z*stride + dz,y*stride + dy,x*stride + dx] += hostGrad[b,c,z,y,x]

	assert np.allclose(hostInGrad[:, :, pad:-pad, pad:-pad, pad:-pad], maxpool3d.grad.get())


if __name__ == "__main__":
	unittest()
