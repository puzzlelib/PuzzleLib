import numpy as np

from PuzzleLib.OpenCL.Driver import Driver

from PuzzleLib.OpenCL.Utils import queue, memoryPool as memPool, globalFills
from PuzzleLib.OpenCL.ThirdParty import libclblast


def outerBatch(xs, ys, outs=None, alpha=1.0, beta=0.0):
	assert xs.ndim == 2 and ys.ndim == 2

	assert xs.dtype == np.float32 and ys.dtype == np.float32
	assert xs.shape[0] == ys.shape[0]

	if outs is None:
		outs = Driver.empty(queue, xs.shape + (ys.shape[1], ), dtype=np.float32, allocator=memPool)

	xoffs = list(range(xs.item_offset, xs.size, xs.shape[1]))
	yoffs = list(range(ys.item_offset, ys.size, ys.shape[1]))
	outoffs = list(range(outs.item_offset, outs.size, outs.strides[0] // outs.dtype.itemsize))

	count = xs.shape[0]
	m, n = outs.shape[1:]

	alphas = [alpha] * count
	betas = [beta] * count

	libclblast.clblastSgemmBatched(queue.int_ptr, 'c', 'n', 't', n, m, 1, alphas, ys.int_ptr, yoffs, n,
								   xs.int_ptr, xoffs, m, betas, outs.int_ptr, outoffs, n)

	return outs


def dotBatch(xs, ys, outs=None, alpha=1.0, beta=0.0):
	assert xs.ndim == 2 and ys.ndim == 2

	assert xs.dtype == np.float32 and ys.dtype == np.float32
	assert xs.shape == ys.shape

	if outs is None:
		outs = Driver.empty(queue, (xs.shape[0], ), dtype=np.float32, allocator=memPool)

	xoffs = list(range(xs.item_offset, xs.size, xs.shape[1]))
	yoffs = list(range(ys.item_offset, ys.size, ys.shape[1]))
	outoffs = list(range(outs.item_offset, outs.size, 1))

	count = xs.shape[0]
	size = xs.shape[1]

	alphas = [alpha] * count
	betas = [beta] * count

	libclblast.clblastSgemmBatched(queue.int_ptr, 'c', 't', 'n', 1, 1, size, alphas, xs.int_ptr, xoffs, size,
								   ys.int_ptr, yoffs, size, betas, outs.int_ptr, outoffs, 1)

	return outs


def sumOnTensorGroup(tensor, out=None, formatT="bgp", cols=True, alpha=1.0, beta=0.0):
	assert tensor.ndim == 3
	assert tensor.dtype == np.float32

	if formatT == "bgp":
		if out is None:
			out = Driver.empty(queue, tensor.shape[1:] if cols else tensor.shape[:2],
							   dtype=np.float32, allocator=memPool)
		else:
			if cols: assert out.shape == tensor.shape[1:]
			else: assert out.shape == tensor.shape[:2]

		if cols:
			ones = globalFills((1, tensor.shape[0]), dtype=tensor.dtype)
			mulTensorOnVecGroup(tensor, ones, out=out, formatT=formatT, transpT=True, alpha=alpha, beta=beta)

		else:
			ones = globalFills((1, tensor.shape[2]), dtype=tensor.dtype)
			mulTensorOnVecGroup(tensor, ones, out=out, formatT=formatT, alpha=alpha, beta=beta)

	elif formatT == "gbp":
		if out is None:
			out = Driver.empty(queue, (tensor.shape[0], tensor.shape[2]) if cols else tensor.shape[:2],
							   dtype=np.float32, allocator=memPool)
		else:
			if cols: assert out.shape == (tensor.shape[0], tensor.shape[2])
			else: assert out.shape == tensor.shape[:2]

		if cols:
			ones = globalFills((1, tensor.shape[1]), dtype=tensor.dtype)
			mulTensorOnVecGroup(tensor, ones, out=out, formatT=formatT, transpT=True, alpha=alpha, beta=beta)

		else:
			ones = globalFills((1, tensor.shape[2]), dtype=tensor.dtype)
			mulTensorOnVecGroup(tensor, ones, out=out, formatT=formatT, alpha=alpha, beta=beta)

	else:
		raise ValueError("Unsupported tensor format")

	return out


def mulTensorOnVecGroup(tensor, vecs, out=None, formatT="bgp", transpT=False, alpha=1.0, beta=0.0):
	assert tensor.ndim == 3 and vecs.ndim == 2
	assert tensor.dtype == np.float32 and vecs.dtype == np.float32

	if formatT == "bgp":
		assert tensor.shape[1] == vecs.shape[0] or vecs.shape[0] == 1
		if transpT:
			assert tensor.shape[0] == vecs.shape[1]
			shape = (tensor.shape[1], tensor.shape[2])
		else:
			assert tensor.shape[2] == vecs.shape[1]
			shape = (tensor.shape[1], tensor.shape[0])

	elif formatT == "gbp":
		assert tensor.shape[0] == vecs.shape[0] or vecs.shape[0] == 1
		if transpT:
			assert tensor.shape[1] == vecs.shape[1]
			shape = (tensor.shape[0], tensor.shape[2])
		else:
			assert tensor.shape[2] == vecs.shape[1]
			shape = (tensor.shape[0], tensor.shape[1])

	else:
		raise ValueError("Unsupported tensor format")

	if out is None:
		out = Driver.empty(queue, shape, dtype=np.float32, allocator=memPool)
	else:
		assert shape == out.shape

	if formatT == "bgp":
		tensoroffs = list(range(tensor.item_offset, tensor.strides[0] // tensor.dtype.itemsize,
								tensor.strides[1] // tensor.dtype.itemsize))
	else:
		tensoroffs = list(range(tensor.item_offset, tensor.size, tensor.strides[0] // tensor.dtype.itemsize))

	count = len(tensoroffs)
	alphas, betas = [alpha] * count, [beta] * count

	if vecs.shape[0] == 1:
		vecoffs = [vecs.item_offset] * count
	else:
		vecoffs = list(range(vecs.item_offset, vecs.size, vecs.strides[0] // vecs.dtype.itemsize))

	outoffs = list(range(out.item_offset, out.size, out.strides[0] // out.dtype.itemsize))

	if formatT == "bgp":
		k, m = tensor.shape[0], tensor.shape[2]

		if transpT:
			libclblast.clblastSgemmBatched(queue.int_ptr, 'c', 'n', 't', 1, m, k, alphas, vecs.int_ptr,
										   vecoffs, 1, tensor.int_ptr, tensoroffs, m * count, betas,
										   out.int_ptr, outoffs, 1)
		else:
			libclblast.clblastSgemmBatched(queue.int_ptr, 'c', 'n', 'n', 1, k, m, alphas, vecs.int_ptr,
										   vecoffs, 1, tensor.int_ptr, tensoroffs, m * count, betas,
										   out.int_ptr, outoffs, 1)

	else:
		k, m = tensor.shape[1:]

		if transpT:
			libclblast.clblastSgemmBatched(queue.int_ptr, 'c', 'n', 't', 1, m, k, alphas, vecs.int_ptr,
										   vecoffs, 1, tensor.int_ptr, tensoroffs, m, betas,
										   out.int_ptr, outoffs, 1)
		else:
			libclblast.clblastSgemmBatched(queue.int_ptr, 'c', 'n', 'n', 1, k, m, alphas, vecs.int_ptr,
										   vecoffs, 1, tensor.int_ptr, tensoroffs, m, betas,
										   out.int_ptr, outoffs, 1)

	return out


def mulTensorBatch(A, B, formatA="bgp", formatB="bgp", out=None, formatOut="bgp", transpA=False, transpB=False,
				   alpha=1.0, beta=0.0):
	shape, count = inferTensorShapes(A, formatA, transpA, B, formatB, transpB, formatOut)

	if out is None:
		out = Driver.empty(queue, shape, dtype=np.float32, allocator=memPool)
	else:
		assert out.shape == shape

	Aoffs, Boffs, outoffs = acquireTensorPtrs(A, formatA, B, formatB, out, formatOut, count)

	alphas, betas = [alpha] * count, [beta] * count

	q = queue.int_ptr
	Ap, Bp, outp = A.int_ptr, B.int_ptr, out.int_ptr

	if formatA == "gbp":
		lda = 1
	else:
		lda = A.shape[1]

	if formatB == "gbp":
		ldb = 1
	else:
		ldb = B.shape[1]

	if formatOut == "bgp":
		ldc = count
	elif formatOut == "gbp":
		ldc = 1
	else:
		raise ValueError("Unsupported out tensor format")

	if formatA == "bgp" and formatB == "bgp":
		if transpA:
			k, m, n = B.shape[0], B.shape[2], A.shape[2]
			libclblast.clblastSgemmBatched(q, 'c', 'n', 't', m, n, k, alphas, Bp, Boffs, m * ldb, Ap, Aoffs, n * lda,
										   betas, outp, outoffs, m * ldc)
		elif transpB:
			m, k, n = B.shape[0], B.shape[2], A.shape[0]
			libclblast.clblastSgemmBatched(q, 'c', 't', 'n', m, n, k, alphas, Bp, Boffs, k * ldb, Ap, Aoffs, k * lda,
										   betas, outp, outoffs, m * ldc)
		else:
			k, m, n = B.shape[0], B.shape[2], A.shape[0]
			libclblast.clblastSgemmBatched(q, 'c', 'n', 'n', m, n, k, alphas, Bp, Boffs, m * ldb, Ap, Aoffs, k * lda,
										   betas, outp, outoffs, m * ldc)

	elif formatA == "bgp" and formatB == "gbp":
		if transpA:
			k, m, n = B.shape[1], B.shape[2], A.shape[2]
			libclblast.clblastSgemmBatched(q, 'c', 'n', 't', m, n, k, alphas, Bp, Boffs, m, Ap, Aoffs, n * lda, betas,
										   outp, outoffs, m * ldc)
		elif transpB:
			m, k, n = B.shape[1], B.shape[2], A.shape[0]
			libclblast.clblastSgemmBatched(q, 'c', 't', 'n', m, n, k, alphas, Bp, Boffs, k, Ap, Aoffs, k * lda, betas,
										   outp, outoffs, m * ldc)
		else:
			k, m, n = B.shape[1], B.shape[2], A.shape[0]
			libclblast.clblastSgemmBatched(q, 'c', 'n', 'n', m, n, k, alphas, Bp, Boffs, m, Ap, Aoffs, k * lda, betas,
										   outp, outoffs, m * ldc)

	elif formatA == "gbp" and formatB == "bgp":
		if transpA:
			k, m, n = B.shape[0], B.shape[2], A.shape[2]
			libclblast.clblastSgemmBatched(q, 'c', 'n', 't', m, n, k, alphas, Bp, Boffs, m * ldb, Ap, Aoffs, n, betas,
										   outp, outoffs, m * ldc)
		elif transpB:
			m, k, n = B.shape[0], B.shape[2], A.shape[1]
			libclblast.clblastSgemmBatched(q, 'c', 't', 'n', m, n, k, alphas, Bp, Boffs, k * ldb, Ap, Aoffs, k, betas,
										   outp, outoffs, m * ldc)
		else:
			k, m, n = B.shape[0], B.shape[2], A.shape[1]
			libclblast.clblastSgemmBatched(q, 'c', 'n', 'n', m, n, k, alphas, Bp, Boffs, m * ldb, Ap, Aoffs, k, betas,
										   outp, outoffs, m * ldc)

	elif formatA == "gbp" and formatB == "gbp":
		if transpA:
			k, m, n = B.shape[1], B.shape[2], A.shape[2]
			libclblast.clblastSgemmBatched(q, 'c', 'n', 't', m, n, k, alphas, Bp, Boffs, m, Ap, Aoffs, n, betas,
										   outp, outoffs, m * ldc)
		elif transpB:
			m, k, n = B.shape[1], B.shape[2], A.shape[1]
			libclblast.clblastSgemmBatched(q, 'c', 't', 'n', m, n, k, alphas, Bp, Boffs, k, Ap, Aoffs, k, betas,
										   outp, outoffs, m * ldc)
		else:
			k, m, n = B.shape[1], B.shape[2], A.shape[1]
			libclblast.clblastSgemmBatched(q, 'c', 'n', 'n', m, n, k, alphas, Bp, Boffs, m, Ap, Aoffs, k, betas,
										   outp, outoffs, m * ldc)

	return out


def inferTensorShapes(A, formatA, transpA, B, formatB, transpB, formatOut):
	assert not (transpA and transpB)

	assert A.ndim == 3 and B.ndim == 3
	assert A.dtype == B.dtype and B.dtype == np.float32

	if formatA == "bgp" and formatB == "bgp":
		assert A.shape[1] == B.shape[1] or A.shape[1] == 1 or B.shape[1] == 1
		count = max(A.shape[1], B.shape[1])

		if transpA:
			assert A.shape[0] == B.shape[0]
			if formatOut == "bgp":
				shape = (A.shape[2], count, B.shape[2])
			elif formatOut == "gbp":
				shape = (count, A.shape[2], B.shape[2])
			else:
				raise ValueError("Unsupported out tensor format")

		elif transpB:
			assert A.shape[2] == B.shape[2]
			if formatOut == "bgp":
				shape = (A.shape[0], count, B.shape[0])
			elif formatOut == "gbp":
				shape = (count, A.shape[0], B.shape[0])
			else:
				raise ValueError("Unsupported out tensor format")
		else:
			assert A.shape[2] == B.shape[0]
			if formatOut == "bgp":
				shape = (A.shape[0], count, B.shape[2])
			elif formatOut == "gbp":
				shape = (count, A.shape[0], B.shape[2])
			else:
				raise ValueError("Unsupported out tensor format")

	elif formatA == "bgp" and formatB == "gbp":
		assert A.shape[1] == B.shape[0] or A.shape[1] == 1 or B.shape[0] == 1
		count = max(A.shape[1], B.shape[0])

		if transpA:
			assert A.shape[0] == B.shape[1]
			if formatOut == "bgp":
				shape = (A.shape[2], count, B.shape[2])
			elif formatOut == "gbp":
				shape = (count, A.shape[2], B.shape[2])
			else:
				raise ValueError("Unsupported out tensor format")
		elif transpB:
			assert A.shape[2] == B.shape[2]
			if formatOut == "bgp":
				shape = (A.shape[0], count, B.shape[1])
			elif formatOut == "gbp":
				shape = (count, A.shape[0], B.shape[1])
			else:
				raise ValueError("Unsupported out tensor format")
		else:
			assert A.shape[2] == B.shape[1]
			if formatOut == "bgp":
				shape = (A.shape[0], count, B.shape[2])
			elif formatOut == "gbp":
				shape = (count, A.shape[0], B.shape[2])
			else:
				raise ValueError("Unsupported out tensor format")

	elif formatA == "gbp" and formatB == "bgp":
		assert A.shape[0] == B.shape[1] or A.shape[0] == 1 or B.shape[1] == 1
		count = max(A.shape[0], B.shape[1])

		if transpA:
			assert A.shape[1] == B.shape[0]
			if formatOut == "bgp":
				shape = (A.shape[2], count, B.shape[2])
			elif formatOut == "gbp":
				shape = (count, A.shape[2], B.shape[2])
			else:
				raise ValueError("Unsupported out tensor format")
		elif transpB:
			assert A.shape[2] == B.shape[2]
			if formatOut == "bgp":
				shape = (A.shape[1], count, B.shape[0])
			elif formatOut == "gbp":
				shape = (count, A.shape[1], B.shape[0])
			else:
				raise ValueError("Unsupported out tensor format")
		else:
			assert A.shape[2] == B.shape[0]
			if formatOut == "bgp":
				shape = (A.shape[1], count, B.shape[2])
			elif formatOut == "gbp":
				shape = (count, A.shape[1], B.shape[2])
			else:
				raise ValueError("Unsupported out tensor format")

	elif formatA == "gbp" and formatB == "gbp":
		assert A.shape[0] == B.shape[0] or A.shape[0] == 1 or B.shape[0] == 1
		count = max(A.shape[0], B.shape[0])

		if transpA:
			assert A.shape[1] == B.shape[1]
			if formatOut == "bgp":
				shape = (A.shape[2], count, B.shape[2])
			elif formatOut == "gbp":
				shape = (count, A.shape[2], B.shape[2])
			else:
				raise ValueError("Unsupported out tensor format")
		elif transpB:
			assert A.shape[2] == B.shape[2]
			if formatOut == "bgp":
				shape = (A.shape[1], count, B.shape[1])
			elif formatOut == "gbp":
				shape = (count, A.shape[1], B.shape[1])
			else:
				raise ValueError("Unsupported out tensor format")
		else:
			assert A.shape[2] == B.shape[1]
			if formatOut == "bgp":
				shape = (A.shape[1], count, B.shape[2])
			elif formatOut == "gbp":
				shape = (count, A.shape[1], B.shape[2])
			else:
				raise ValueError("Unsupported out tensor format")

	else:
		raise ValueError("Unsupported operand tensors formats")

	return shape, count


def acquireTensorPtrs(A, formatA, B, formatB, out, formatOut, count):
	itemsize = A.dtype.itemsize

	if formatA == "bgp" and formatB == "bgp":
		if A.shape[1] > 1 or (A.shape[1] == 1 and B.shape[1] == 1):
			Aoffs = list(range(A.item_offset, A.strides[0] // itemsize, A.strides[1] // itemsize))
		else:
			Aoffs = [A.item_offset] * count

		if B.shape[1] > 1 or (A.shape[1] == 1 and B.shape[1] == 1):
			Boffs = list(range(B.item_offset, B.strides[0] // itemsize, B.strides[1] // itemsize))
		else:
			Boffs = [B.item_offset] * count

	elif formatA == "bgp" and formatB == "gbp":
		if A.shape[1] > 1 or (A.shape[1] == 1 and B.shape[0] == 1):
			Aoffs = list(range(A.item_offset, A.strides[0] // itemsize, A.strides[1] // itemsize))
		else:
			Aoffs = [A.item_offset] * count

		if B.shape[0] > 1 or (A.shape[1] == 1 and B.shape[0] == 1):
			Boffs = list(range(B.item_offset, B.size, B.strides[0] // itemsize))
		else:
			Boffs = [B.item_offset] * count

	elif formatA == "gbp" and formatB == "bgp":
		if A.shape[0] > 1 or (A.shape[0] == 1 and B.shape[1] == 1):
			Aoffs = list(range(A.item_offset, A.size, A.strides[0] // itemsize))
		else:
			Aoffs = [A.item_offset] * count

		if B.shape[1] > 1 or (A.shape[0] == 1 and B.shape[1] == 1):
			Boffs = list(range(B.item_offset, B.strides[0] // itemsize, B.strides[1] // itemsize))
		else:
			Boffs = [B.item_offset] * count

	elif formatA == "gbp" and formatB == "gbp":
		if A.shape[0] > 1 or (A.shape[0] == 1 and B.shape[0] == 1):
			Aoffs = list(range(A.item_offset, A.size, A.strides[0] // itemsize))
		else:
			Aoffs = [A.item_offset] * count

		if B.shape[0] > 1 or (A.shape[0] == 1 and B.shape[0] == 1):
			Boffs = list(range(B.item_offset, B.size, B.strides[0] // itemsize))
		else:
			Boffs = [B.item_offset] * count

	else:
		raise ValueError("Unsupported operand tensors formats")

	if formatOut == "bgp":
		outoffs = list(range(out.item_offset, out.strides[0] // itemsize, out.strides[1] // itemsize))
	elif formatOut == "gbp":
		outoffs = list(range(out.item_offset, out.size, out.strides[0] // itemsize))

	else:
		raise ValueError("Unsupported out tensor format")

	return Aoffs, Boffs, outoffs


def unittest():
	vecBgpTest()
	vecGbpTest()
	bgpBgpTest()
	bgpGbpTest()
	gbpBgpTest()
	gbpGbpTest()


def vecBgpTest():
	groups = 5

	tensor = Driver.to_device(queue, np.random.randn(7, groups, 4).astype(np.float32))
	x = Driver.to_device(queue, np.random.randn(groups, tensor.shape[2]).astype(np.float32))
	y = Driver.to_device(queue, np.random.randn(groups, tensor.shape[0]).astype(np.float32))

	out = mulTensorOnVecGroup(tensor, x)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[i] = np.dot(tensor.get()[:, i, :], x.get()[i])

	assert np.allclose(hostOut, out.get())

	out = mulTensorOnVecGroup(tensor, y, transpT=True)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[i] = np.dot(tensor.get()[:, i, :].T, y.get()[i])

	assert np.allclose(hostOut, out.get())


def vecGbpTest():
	groups = 5

	tensor = Driver.to_device(queue, np.random.randn(groups, 6, 4).astype(np.float32))
	x = Driver.to_device(queue, np.random.randn(groups, tensor.shape[2]).astype(np.float32))
	y = Driver.to_device(queue, np.random.randn(groups, tensor.shape[1]).astype(np.float32))
	z = Driver.to_device(queue, np.random.randn(groups, tensor.shape[2]).astype(np.float32))

	out = mulTensorOnVecGroup(tensor, x, formatT="gbp")

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[i] = np.dot(tensor.get()[i], x.get()[i])

	assert np.allclose(hostOut, out.get())

	out = mulTensorOnVecGroup(tensor, y, formatT="gbp", transpT=True)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[i] = np.dot(tensor.get()[i].T, y.get()[i])

	assert np.allclose(hostOut, out.get())

	outs = outerBatch(y, x)

	hostOuts = np.empty(outs.shape, dtype=np.float32)
	for i in range(groups):
		hostOuts[i] = np.outer(y.get()[i], x.get()[i])

	assert np.allclose(hostOuts, outs.get())

	outs = dotBatch(x, z)

	hostOuts = np.empty(outs.shape, dtype=np.float32)
	for i in range(groups):
		hostOuts[i] = np.dot(x.get()[i], z.get()[i])

	assert np.allclose(hostOuts, outs.get())


def bgpBgpTest():
	groups = 3

	A = Driver.to_device(queue, np.random.randn(4, groups, 7).astype(np.float32))
	B = Driver.to_device(queue, np.random.randn(A.shape[2], groups, 5).astype(np.float32))
	C = Driver.to_device(queue, np.random.randn(A.shape[0], groups, B.shape[2]).astype(np.float32))

	out = mulTensorBatch(A, B, formatA="bgp", formatB="bgp")

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(A.get()[:, i, :], B.get()[:, i, :])

	assert np.allclose(hostOut, out.get())

	out = mulTensorBatch(A, C, formatA="bgp", formatB="bgp", transpA=True)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(A.get()[:, i, :].T, C.get()[:, i, :])

	assert np.allclose(hostOut, out.get())

	out = mulTensorBatch(B, C, formatA="bgp", formatB="bgp", transpB=True)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(B.get()[:, i, :], C.get()[:, i, :].T)

	assert np.allclose(hostOut, out.get())


def bgpGbpTest():
	groups = 3

	A = Driver.to_device(queue, np.random.randn(4, groups, 7).astype(np.float32))
	B = Driver.to_device(queue, np.random.randn(groups, A.shape[2], 5).astype(np.float32))
	C = Driver.to_device(queue, np.random.randn(groups, A.shape[0], 8).astype(np.float32))
	D = Driver.to_device(queue, np.random.randn(groups, 6, A.shape[2]).astype(np.float32))

	out = mulTensorBatch(A, B, formatA="bgp", formatB="gbp")

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(A.get()[:, i, :], B.get()[i])

	assert np.allclose(hostOut, out.get())

	out = mulTensorBatch(A, C, formatA="bgp", formatB="gbp", transpA=True)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(A.get()[:, i, :].T, C.get()[i])

	assert np.allclose(hostOut, out.get())

	out = mulTensorBatch(A, D, formatA="bgp", formatB="gbp", transpB=True)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(A.get()[:, i, :], D.get()[i].T)

	assert np.allclose(hostOut, out.get())


def gbpBgpTest():
	groups = 3

	A = Driver.to_device(queue, np.random.randn(groups, 4, 7).astype(np.float32))
	B = Driver.to_device(queue, np.random.randn(A.shape[2], groups, 5).astype(np.float32))
	C = Driver.to_device(queue, np.random.randn(A.shape[1], groups, 8).astype(np.float32))
	D = Driver.to_device(queue, np.random.randn(6, groups, A.shape[2]).astype(np.float32))

	out = mulTensorBatch(A, B, formatA="gbp", formatB="bgp")

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(A.get()[i], B.get()[:, i, :])

	assert np.allclose(hostOut, out.get())

	out = mulTensorBatch(A, C, formatA="gbp", formatB="bgp", transpA=True)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(A.get()[i].T, C.get()[:, i, :])

	assert np.allclose(hostOut, out.get())

	out = mulTensorBatch(A, D, formatA="gbp", formatB="bgp", transpB=True)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[:, i, :] = np.dot(A.get()[i], D.get()[:, i, :].T)

	assert np.allclose(hostOut, out.get())


def gbpGbpTest():
	groups = 3

	A = Driver.to_device(queue, np.random.randn(groups, 4, 3).astype(np.float32))
	B = Driver.to_device(queue, np.random.randn(groups, A.shape[2], 4).astype(np.float32))
	C = Driver.to_device(queue, np.random.randn(groups, A.shape[1], 6).astype(np.float32))
	D = Driver.to_device(queue, np.random.randn(groups, 8, C.shape[2]).astype(np.float32))

	out = mulTensorBatch(A, B, formatA="gbp", formatB="gbp", formatOut="gbp")

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[i] = np.dot(A.get()[i], B.get()[i])

	assert np.allclose(hostOut, out.get())

	out = mulTensorBatch(C, A, formatA="gbp", formatB="gbp", formatOut="gbp", transpA=True)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[i] = np.dot(C.get()[i].T, A.get()[i])

	assert np.allclose(hostOut, out.get())

	out = mulTensorBatch(C, D, formatA="gbp", formatB="gbp", formatOut="gbp", transpB=True)

	hostOut = np.empty(out.shape, dtype=np.float32)
	for i in range(groups):
		hostOut[i] = np.dot(C.get()[i], D.get()[i].T)

	assert np.allclose(hostOut, out.get())


if __name__ == "__main__":
	unittest()
