from PuzzleLib import Config


mulTensorOnVecGroup = None
sumOnTensorGroup = None
mulTensorBatch = None


def autoinit():
	if not Config.shouldInit():
		return

	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.hip:
		initHip()
	elif Config.isCPUBased(Config.backend):
		initCPU()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda import Backend
	initGPU(Backend)


def initHip():
	from PuzzleLib.Hip import Backend
	initGPU(Backend)


def initGPU(Backend):
	backend = Backend.getBackend(Config.deviceIdx, initmode=2)
	memoryPool, blas, matmod = backend.memoryPool, backend.blas, backend.matmod

	formats = {
		"gbp": backend.GroupFormat.gbp.value,
		"bgp": backend.GroupFormat.bgp.value
	}

	def wrapMulTensorOnVecGroup(tensor, vecs, out=None, formatT="bgp", transpT=False, alpha=1.0, beta=0.0):
		assert tensor.ndim == 3 and formatT == "gbp"
		axis = 0 if transpT else 1

		return matmod.matvec(tensor, vecs, axis, out, alpha, beta, memoryPool)

	def wrapSumOnTensorGroup(tensor, out=None, formatT="bgp", cols=True, alpha=1.0, beta=0.0):
		assert tensor.ndim == 3
		axis = (1 if formatT == "gbp" else 0) if cols else 2

		return matmod.matsum(tensor, axis, out, alpha, beta, memoryPool)

	def wrapMulTensorBatch(A, B, formatA="bgp", formatB="bgp", out=None, formatOut="bgp", transpA=False, transpB=False,
						   alpha=1.0, beta=0.0):
		formatA, formatB, formatOut = formats[formatA], formats[formatB], formats[formatOut]
		return blas.gemmBatched(A, B, formatA, formatB, formatOut, transpA, transpB, alpha, beta, out, memoryPool)

	global mulTensorOnVecGroup, sumOnTensorGroup, mulTensorBatch
	mulTensorOnVecGroup = wrapMulTensorOnVecGroup
	sumOnTensorGroup = wrapSumOnTensorGroup
	mulTensorBatch = wrapMulTensorBatch


def initCPU():
	pass


autoinit()
