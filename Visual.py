import math, os, io

import numpy as np
from PIL import Image


class VisualError(Exception):
	pass


def loadImage(filename, shape=None, normalize=True, mapsToFront=True):
	img = Image.open(filename)
	return imageToArray(img, shape, normalize, mapsToFront)


def loadImageFromBytes(bytebuffer, shape=None, normalize=True, mapsToFront=True):
	img = Image.open(io.BytesIO(bytebuffer))
	return imageToArray(img, shape, normalize, mapsToFront)


def imageToArray(img, shape=None, normalize=True, mapsToFront=True):
	img = img.resize(shape, Image.ANTIALIAS) if shape is not None else img
	img = np.array(img, dtype=np.float32 if normalize else np.uint8)

	if img.ndim == 3 and img.shape[-1] == 4:
		img = img[:, :, :3]

	if normalize:
		if img.max() > 0.0:
			img *= 2.0 / img.max()

		img -= 1.0

	if mapsToFront:
		if img.ndim == 2:
			img = img.reshape((1, 1, *img.shape))

		else:
			img = np.rollaxis(img, 2)
			img = np.ascontiguousarray(img, dtype=img.dtype).reshape(1, *img.shape)

	elif img.ndim == 2:
		img = img.reshape(*img.shape, 1)

	return img


def showImage(img, filename, rollMaps=True):
	if img.ndim == 4:
		if img.shape[0] != 1:
			raise VisualError("Image tensor must be exactly one image")
		else:
			img = img[0]

	normImg = img

	if img.dtype == np.float32:
		normImg = np.copy(img)
		normalizeImageInplace(normImg)

		if normImg.ndim == 3:
			if normImg.shape[0] == 1:
				normImg = normImg.reshape(*normImg.shape[1:])

			elif rollMaps:
				normImg = np.rollaxis(normImg, 0, 3)

		normImg = imageToInt(normImg)

	Image.fromarray(normImg).save(filename)


def showImageBatch(batch, filebase, ext="png", rollMaps=True):
	if batch.ndim != 4:
		raise VisualError("Imagebatch tensor must be 4d")

	for i in range(batch.shape[0]):
		img = batch[i]
		img = img[0] if img.shape[0] == 1 else img

		showImage(img, "%s-%d.%s" % (filebase, i + 1, ext.replace(".", "")), rollMaps)


def showImageBatchInFolder(batch, foldername, basename, ext="png", rollMaps=True):
	if not os.path.isdir(foldername):
		os.mkdir(foldername)

	showImageBatch(batch, os.path.join(foldername, basename), ext, rollMaps)


def showImageBasedFilters(filters, filename, offset=4, normalize=True, cols=16):
	outmaps, inmaps, fh, fw = filters.shape
	if inmaps != 3:
		raise VisualError("Filter tensor must have 3 inmaps")

	rows = int(math.ceil(outmaps / cols))

	width = cols * fw + (cols + 1) * offset
	height = rows * fh + (rows + 1) * offset

	image = np.zeros((height, width, 3), dtype=np.uint8)

	hstep = offset + fh
	wstep = offset + fw

	for r in range(rows):
		for c in range(cols):
			if r * cols + c >= outmaps:
				break

			f = np.copy(filters[r * cols + c])

			if normalize:
				normalizeImageInplace(f)

			f = np.rollaxis(imageToInt(f), 0, 3)
			image[offset + r * hstep:offset + r * hstep + fh, offset + c * wstep:offset + c * wstep + fw] = f

	Image.fromarray(image).save(filename)


def showFilters(filters, filename, offset=4, normalize=True):
	outmaps, inmaps, fh, fw = filters.shape

	if fh == fw == 1:
		print("Aborting showing 1x1 filters in file %s ..." % filename)
		return

	width = inmaps * fw + (inmaps + 1) * offset
	height = outmaps * fh + (outmaps + 1) * offset

	image = np.zeros((height, width), dtype=np.uint8)

	hstep = offset + fh
	wstep = offset + fw

	for i in range(outmaps):
		for j in range(inmaps):
			f = np.copy(filters[i, j])

			if normalize:
				normalizeImageInplace(f)

			f = imageToInt(f)
			image[offset + i * hstep:offset + i * hstep + fh, offset + j * wstep:offset + j * wstep + fw] = f

	Image.fromarray(image).save(filename)


def showChanneledFilters(filters, filename, offset=4, normalize=True):
	outmaps, inmaps, ch, fh, fw = filters.shape

	width = inmaps * fw + (inmaps + 1) * offset
	height = outmaps * fh + (outmaps + 1) * offset

	image = np.zeros((height, width, ch), dtype=np.uint8)

	hstep = offset + fh
	wstep = offset + fw

	for i in range(outmaps):
		for j in range(inmaps):
			f = np.copy(filters[i, j])

			if normalize:
				normalizeImageInplace(f)

			f = np.moveaxis(imageToInt(f), 0, 2)
			image[offset + i * hstep:offset + i * hstep + fh, offset + j * wstep:offset + j * wstep + fw, :] = f

	Image.fromarray(image).save(filename)


def normalizeImageInplace(img):
	img -= img.min()
	if img.max() > 0.0:
		img /= img.max()


def imageToInt(img):
	return (img * 255.0).astype(np.uint8)


def whiten(batch, epsilon=1e-2, PCA=False):
	shape = batch.shape
	batch = batch.reshape(batch.shape[0], -1)

	mean = np.mean(batch, axis=0)
	batch -= mean[np.newaxis, :]

	sigma = np.dot(batch.T, batch) / batch.shape[0]
	U, S, V = np.linalg.svd(sigma.astype(np.float32))

	zca = np.dot(U, np.diag(1.0 / np.sqrt(S + epsilon)))
	zca = np.dot(zca, V) if not PCA else zca

	return np.dot(batch, zca).reshape(shape)


def unittest():
	filters = np.random.randn(16, 16, 16, 16).astype(np.float32)
	showFilters(filters, "./TestData/testFilters.png")

	filters = np.random.normal(size=(32, 3, 32, 32)).astype(np.float32)
	showImageBasedFilters(filters, "./TestData/testColorFilters.png")

	filters = np.random.normal(size=(16, 24, 3, 16, 16)).astype(np.float32)
	showChanneledFilters(filters, "./TestData/testChanneledFilters.png")

	img = np.random.normal(size=(3, 32, 32)).astype(np.float32)
	showImage(img, "./TestData/testImage.png")

	batch = np.random.normal(size=(4, 1, 16, 16)).astype(np.float32)
	showImageBatch(batch, "./TestData/testBatch", ".png")


if __name__ == "__main__":
	unittest()
