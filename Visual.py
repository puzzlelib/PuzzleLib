import io, os

import numpy as np
from PIL import Image


class VisualError(Exception):
	pass


def loadImage(filename, shape=None, normalize=True, mapsToFront=True, contiguous=True):
	img = Image.open(filename)
	return imageToArray(img, shape, normalize, mapsToFront, contiguous)


def loadImageFromBytes(bytebuffer, shape=None, normalize=True, mapsToFront=True, contiguous=True):
	img = Image.open(io.BytesIO(bytebuffer))
	return imageToArray(img, shape, normalize, mapsToFront, contiguous)


def imageToArray(img, shape=None, normalize=True, mapsToFront=True, contiguous=True):
	img = np.array(img.resize(shape, Image.ANTIALIAS) if shape is not None else img, dtype=np.uint8)

	if img.ndim == 3 and img.shape[-1] == 4:
		img = img[:, :, :3]

	if mapsToFront:
		img = img[np.newaxis, np.newaxis, ...] if img.ndim == 2 else np.rollaxis(img, 2)[np.newaxis, ...]

	elif img.ndim == 2:
		img = img[..., np.newaxis]

	if normalize:
		img = img.astype(np.float32)

		if img.max() > 0.0:
			img *= 2.0 / img.max()

		img -= 1.0

	return np.ascontiguousarray(img) if contiguous else img


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

		if rollMaps and normImg.ndim == 3 and normImg.shape[0] > 1:
			normImg = np.rollaxis(normImg, 0, 3)

		normImg = imageToInt(normImg)

	Image.fromarray(normImg.squeeze()).save(filename)


def showImageBatch(batch, filebase, ext="png", rollMaps=True):
	if batch.ndim != 4:
		raise VisualError("Imagebatch tensor must be 4d tensor")

	ext = ext.replace(".", "")

	for i in range(batch.shape[0]):
		showImage(batch[i], "%s-%d.%s" % (filebase, i + 1, ext), rollMaps)


def showImageBatchInFolder(batch, foldername, basename, ext="png", rollMaps=True):
	if not os.path.isdir(foldername):
		os.mkdir(foldername)

	showImageBatch(batch, os.path.join(foldername, basename), ext, rollMaps)


def showFilters(filters, filename, offset=4, normalize=True):
	outmaps, inmaps, fh, fw = filters.shape
	showImageBasedFilters(
		filters.reshape(outmaps * inmaps, 1, fh, fw), filename, cols=inmaps, offset=offset, normalize=normalize
	)


def showImageBasedFilters(filters, filename, cols=16, offset=4, normalize=True):
	outmaps, inmaps, fh, fw = filters.shape

	if fh == fw == 1:
		print("Aborting showing 1x1 filters in file %s ..." % filename)
		return

	rows = (outmaps + cols - 1) // cols

	height = rows * fh + (rows + 1) * offset
	width = cols * fw + (cols + 1) * offset

	image = np.zeros((height, width, inmaps), dtype=np.uint8)
	hstep, wstep = offset + fh, offset + fw

	for index in range(outmaps):
		r, c = index // cols, index % cols
		f = filters[index]

		if normalize:
			f = np.copy(f)
			normalizeImageInplace(f)

		f = np.moveaxis(imageToInt(f), 0, 2)
		image[offset + r * hstep:offset + r * hstep + fh, offset + c * wstep:offset + c * wstep + fw] = f

	Image.fromarray(image.squeeze()).save(filename)


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

	img = np.random.normal(size=(3, 32, 32)).astype(np.float32)
	showImage(img, "./TestData/testImage.png")

	batch = np.random.normal(size=(4, 1, 16, 16)).astype(np.float32)
	showImageBatch(batch, "./TestData/testBatch")


if __name__ == "__main__":
	unittest()
