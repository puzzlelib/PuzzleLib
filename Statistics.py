import numpy as np


def confusion(labels, predictions, dim=0, log=True):
	if dim <= 0:
		for label in labels:
			if int(label) >= dim:
				dim = int(label)+1

		for label in predictions:
			if int(label) >= dim:
				dim = int(label) + 1

	cm = [[0] * dim for _ in range(dim)]

	for i in range(min(len(labels), len(predictions))):
		cm[int(labels[i])][int(predictions[i])] += 1

	if log:
		print("Confusion Matrix:")

		for mst in cm:
			print(str(mst))

	return cm


def precision(cm, log=True, verbose=True):
	dim = len(cm)
	prs, pr = [], 0

	for i in range(dim):
		summary = 0

		for j in range(dim):
			summary += cm[j][i]

		if summary == 0:
			tpr = 1.0
		else:
			tpr = cm[i][i] / summary

		pr += tpr
		prs.append(tpr)

		if log and verbose:
			print("Precision on class %s: %s" % (i, tpr))

	pr /= dim

	if log:
		print("Precision mean: %s" % pr)

	return pr, prs


def recall(cm, log=True, verbose=True):
	dim = len(cm)
	rcs, rc = [], 0

	for i in range(dim):
		summary = 0

		for j in range(dim):
			summary += cm[i][j]

		if summary == 0:
			trc = 1.0
		else:
			trc = cm[i][i] / summary

		rc += trc
		rcs.append(trc)

		if log and verbose:
			print("Recall on class %d: %f" % (i, trc))

	rc /= dim

	if log:
		print("Recall mean: %s" % rc)

	return rc, rcs


def accuracy(cm, log=True):
	dim = len(cm)
	acc, summary = 0, 0

	for i in range(dim):
		for j in range(dim):
			summary += cm[i][j]

		acc += cm[i][i]

	acc /= summary

	if log:
		print("Accuracy: %s" % acc)

	return acc


def fullstats(labels, predictions, dim=0, printing=True, verbose=True):
	cm = confusion(labels, predictions, dim, printing)
	pr, prs = precision(cm, printing, verbose)
	rc, rcs = recall(cm, printing, verbose)

	return cm, pr, rc, prs, rcs


def unittest():
	cm = np.ones(shape=(1000, 1000), dtype=np.float32)

	assert np.isclose(precision(cm, verbose=False)[0], 0.001)
	assert np.isclose(recall(cm, verbose=False)[0], 0.001)
	assert np.isclose(accuracy(cm), 0.001)

	import random
	labels = [random.randint(0, 5) for _ in range(10000)]
	results = [random.randint(0, 5) for _ in range(10000)]

	cm, pr, rc, prs, rcs = fullstats(labels, results, verbose=False)

	labels = np.array(labels)
	results = np.array(results)

	npcm, nppr, nprc, npprs, nprcs = fullstats(labels, results, verbose=False)

	assert np.allclose(np.array(cm), npcm)
	assert np.allclose(np.array(pr), nppr)
	assert np.allclose(np.array(rc), nprc)
	assert np.allclose(np.array(prs), npprs)
	assert np.allclose(np.array(rcs), nprcs)


if __name__ == "__main__":
	unittest()
