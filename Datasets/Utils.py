import random

import numpy as np

from PuzzleLib.Handlers.Calculator import Calculator
from PuzzleLib import Statistics


def validate(net, valData, valLabels, dim=0, batchsize=128, log=False):
	if dim == 0:
		dim = getDim(valLabels)

	confMat = np.zeros(shape=(dim, dim))
	predictions = Calculator(net, batchsize=batchsize).calcFromHost(valData)

	for i in range(predictions.shape[0]):
		confMat[valLabels[i], np.argmax(predictions[i])] += 1

	if log:
		print("Confusion matrix:\n" + str(confMat))

	precision, _ = Statistics.precision(confMat, log=log)
	recall, _ = Statistics.recall(confMat, log=log)
	accuracy = Statistics.accuracy(confMat, log=log)

	return precision, recall, accuracy


def splitData(data, labels=None, dim=0, validation=0.1, permutation=True, uniformVal=True):
	if len(data) == 0:
		return None

	if permutation:
		data, labels = permutateData(data, labels)

	if labels is None:
		splitter = int(validation * len(data))
		return data[splitter:], data[:splitter]

	if dim < 1:
		dim = getDim(labels)

	counter = [0] * dim
	coe = [0] * dim
	for label in labels:
		coe[label] += 1

	if uniformVal:
		size = int(validation * min(coe))
		for i in range(len(coe)):
			coe[i] = size
		valSize = dim * size
	else:
		for i in range(len(coe)):
			coe[i] = int(coe[i] * validation)
		valSize = sum(coe)

	trainSize = len(data) - valSize

	valLabels = np.empty((valSize, ), labels.dtype) if isinstance(labels, np.ndarray) else [labels[0]] * valSize
	valData = np.empty((valSize, ) + data.shape[1:], data.dtype) if isinstance(data, np.ndarray) else \
		[data[0]] * valSize

	trainLabels = np.empty((trainSize, ), labels.dtype) if isinstance(labels, np.ndarray) else [labels[0]] * trainSize
	trainData = np.empty((trainSize, ) + data.shape[1:], data.dtype) if isinstance(data, np.ndarray) else \
		[data[0]] * trainSize

	valIdx, trainIdx = 0, 0
	for i in range(len(data)):
		if counter[labels[i]] < coe[labels[i]]:
			valData[valIdx] = data[i]
			valLabels[valIdx] = labels[i]

			valIdx += 1
			counter[labels[i]] += 1
		else:
			trainData[trainIdx] = data[i]
			trainLabels[trainIdx] = labels[i]

			trainIdx += 1

	return trainData, valData, trainLabels, valLabels


def replicateData(data, labels, dim=0, permutation=True):
	checkShape(data, labels)

	if dim < 1:
		dim = getDim(labels)

	coe = [0] * dim
	for label in labels:
		coe[label] += 1

	top = max(coe)
	for i in range(dim):
		if coe[i] > 0:
			coe[i] = top / coe[i]

	cur = [0] * dim
	res = [0] * dim

	length = dim * top

	newData = np.empty((length, ) + data.shape[1:], data.dtype) if isinstance(data, np.ndarray) else [data[0]] * length
	newLabels = np.empty((length, ), labels.dtype) if isinstance(labels, np.ndarray) else [labels[0]] * length

	idx = 0
	for i in range(len(data)):
		cur[labels[i]] += coe[labels[i]]

		while res[labels[i]] < cur[labels[i]] - 0.1:
			newData[idx] = data[i]
			newLabels[idx] = labels[i]
			idx += 1
			res[labels[i]] += 1

	if permutation:
		newData, newLabels = permutateData(newData, newLabels)

	return newData, newLabels


def permutateData(data, labels=None, constantMemory=False):
	perm = np.random.permutation(len(data))

	if not constantMemory:
		if labels is not None:
			tmp = labels.copy()
			for i in range(checkShape(data, labels)):
				labels[i] = tmp[perm[i]]

		tmp = data.copy()
		for i in range(len(data)):
			data[i] = tmp[perm[i]]

	else:
		while True:
			idx = 0
			flag = False

			for idx in range(len(perm)):
				if perm[idx] >= 0:
					flag = True
					break

			if not flag:
				break

			jdx = idx
			while True:
				if perm[jdx] != idx:
					data[jdx], data[perm[jdx]] = data[perm[jdx]], data[jdx]

					if labels is not None:
						labels[jdx], labels[perm[jdx]] = labels[perm[jdx]], labels[jdx]

					oldjdx = jdx
					jdx = perm[jdx]
					perm[oldjdx] = -1  # in place

				else:
					perm[jdx] = -1  # in place
					break

	return data, labels


def checkShape(data, labels):
	assert len(data) == len(labels)
	return len(data)


def getDim(labels, log=False):
	assert len(labels) > 0
	assert (isinstance(labels[0], np.int32) or isinstance(labels[0], int))

	dim = np.max(labels) + 1

	if log:
		coe = [0] * dim

		for label in labels:
			coe[label] += 1

		print("Labels count:")
		for i in range(dim):
			print("%d: %d (~%d%%)" % (i, coe[i], 100 * coe[i] // labels.shape[0]))

	return int(dim)


def merge(data):
	res = []

	for i, item in enumerate(data):
		text = []

		for j, sentence in enumerate(item):
			text += data[i][j]

		res.append(text)

	return res


def merge2D(data):
	mesh = [0] * len(data)
	res = []
	cnt = 0

	for i, item in enumerate(data):
		res += item
		mesh[i] = {"x1": cnt, "x2": cnt+len(item)}
		cnt += len(item)

	return res, mesh


def split2D(data, mesh):
	res = []

	for idx in mesh:
		res.append(data[(idx["x1"]):(idx["x2"])])

	return res


def resizeDataToSize(data, dataSize):
	newData = [''] * (dataSize - len(data))
	newData = data + newData

	return newData


def unittest():
	mergeTest()
	numpyInterfaceTest()
	pyInterfaceTest()


def mergeTest():
	data = [["sfas", "sdfasfasdf", "gdfgd"], ["dfg"], ["yry", "rtyher"]]
	a, mesh = merge2D(data)
	b = split2D(a, mesh)

	assert data == b


def numpyInterfaceTest():
	data = np.random.randn(10000).astype(np.float32)
	labels = np.random.randint(0, 10, size=(10000, ), dtype=np.int32)

	assert checkShape(data, labels) == len(data)
	assert getDim(labels) == 10

	tData, vData, tLabels, vLabels = splitData(data, labels, validation=0.1, permutation=True)

	assert isinstance(tData, np.ndarray)
	assert isinstance(tLabels, np.ndarray)
	assert checkShape(tData, tLabels) == len(tLabels)
	assert checkShape(vData, vLabels) == len(vLabels)

	interfaceTest(tData, tLabels, np.ndarray)


def pyInterfaceTest():
	data = [random.random() for _ in range(10000)]
	labels = [random.randint(0, 9) for _ in range(10000)]

	assert checkShape(data, labels) == len(data)
	assert getDim(labels) == 10

	tData, vData, tLabels, vLabels = splitData(data, labels, validation=0.5, permutation=True, uniformVal=False)

	assert isinstance(tData, list)
	assert isinstance(tLabels, list)
	assert checkShape(tData, tLabels) == len(tLabels)
	assert checkShape(vData, vLabels) == len(vLabels)

	interfaceTest(tData, tLabels, list)


def interfaceTest(data, labels, typ):
	data, labels = replicateData(data, labels, permutation=True)

	assert isinstance(data, typ)
	assert isinstance(labels, typ)
	assert checkShape(data, labels) == len(labels)

	tData, tLabels = permutateData(data, labels)

	assert isinstance(tData, typ)
	assert isinstance(tLabels, typ)
	assert checkShape(tData, tLabels) == len(tLabels)

	tData, tLabels = permutateData(tData, tLabels, constantMemory=True)

	assert isinstance(tData, typ)
	assert isinstance(tLabels, typ)
	assert checkShape(tData, tLabels) == len(tLabels)

	res = [0] * 10

	for i in range(len(tLabels)):
		res[tLabels[i]] += 1

	for r in res:
		assert r == res[0]


if __name__ == "__main__":
	unittest()
