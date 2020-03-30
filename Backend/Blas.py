from PuzzleLib import Config


toVectorAddVector = None
addVectorToVector = None
dot = None
vectorL1Norm = None

mulMatrixOnMatrix = None
sumOnMatrix = None


def autoinit():
	if not Config.shouldInit():
		return

	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.hip:
		initHip()
	elif Config.backend == Config.Backend.cpu:
		initCPU()
	elif Config.backend == Config.Backend.intel:
		initIntel()
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
	GPUArray, memoryPool, blas, matmod = backend.GPUArray, backend.memoryPool, backend.blas, backend.matmod

	def wrapToVectorAddVector(y, x, alpha=1.0):
		backend.toVectorAddVectorKer(y.dtype)(y, x, alpha)
		return y

	def wrapAddVectorToVector(x, y, out=None, alpha=1.0, beta=1.0):
		if out is None:
			out = GPUArray.empty(x.shape, dtype=x.dtype, allocator=memoryPool)
		else:
			assert out.shape == x.shape

		backend.addKer(out.dtype)(out, x, alpha, y, beta)
		return out

	def wrapGemm(A, B, out=None, transpA=False, transpB=False, alpha=1.0, beta=0.0):
		return blas.gemm(A, B, out, transpA, transpB, alpha, beta, memoryPool)

	def wrapSumOnMatrix(A, out=None, cols=True, alpha=1.0, beta=0.0):
		assert A.ndim == 2
		return matmod.matsum(A, 0 if cols else 1, out, alpha, beta, memoryPool)

	global toVectorAddVector, addVectorToVector, dot, vectorL1Norm
	toVectorAddVector = wrapToVectorAddVector
	addVectorToVector = wrapAddVectorToVector
	dot = blas.dot
	vectorL1Norm = blas.l1norm

	global mulMatrixOnMatrix, sumOnMatrix
	mulMatrixOnMatrix = wrapGemm
	sumOnMatrix = wrapSumOnMatrix


def initCPU():
	from PuzzleLib.CPU.Wrappers import NumpyBlas

	global toVectorAddVector, addVectorToVector, dot, vectorL1Norm
	toVectorAddVector = NumpyBlas.toVectorAddVector
	addVectorToVector = NumpyBlas.addVectorToVector
	dot = NumpyBlas.dot
	vectorL1Norm = NumpyBlas.vectorL1Norm

	global mulMatrixOnMatrix, sumOnMatrix
	mulMatrixOnMatrix = NumpyBlas.mulMatrixOnMatrix
	sumOnMatrix = NumpyBlas.sumOnMatrix


def initIntel():
	initCPU()

	from PuzzleLib.Intel.Wrappers import DNNLBlas

	global mulMatrixOnMatrix
	mulMatrixOnMatrix = DNNLBlas.mulMatrixOnMatrix


autoinit()
