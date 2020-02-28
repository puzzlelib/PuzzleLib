from PuzzleLib import Config


toVectorAddVector = None
addVectorToVector = None
dot = None
vectorL1Norm = None

mulMatrixOnMatrix = None
sumOnMatrix = None


def autoinit():
	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.opencl:
		initOpenCL()
	elif Config.backend == Config.Backend.cpu:
		initCPU()
	elif Config.backend == Config.Backend.intel:
		initIntel()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda.GPUArray import GPUArray
	from PuzzleLib.Cuda.Utils import memoryPool

	from PuzzleLib.Cuda.Wrappers.CuBlas import context
	from PuzzleLib.Cuda.Kernels.MatVec import matsum
	from PuzzleLib.Cuda.Kernels.ElementWise import toVectorAddVectorKer, addKer

	def wrapToVectorAddVector(y, x, alpha=1.0):
		toVectorAddVectorKer(y.dtype)(y, x, alpha)
		return y

	def wrapAddVectorToVector(x, y, out=None, alpha=1.0, beta=1.0, allocator=memoryPool):
		if out is None:
			out = GPUArray.empty(x.shape, dtype=x.dtype, allocator=allocator)
		else:
			assert out.shape == x.shape

		addKer(out.dtype)(out, x, alpha, y, beta)
		return out

	def wrapGemm(A, B, out=None, transpA=False, transpB=False, alpha=1.0, beta=0.0, allocator=memoryPool):
		return context.gemm(A, B, out, transpA, transpB, alpha, beta, allocator)

	def wrapSumOnMatrix(A, out=None, cols=True, alpha=1.0, beta=0.0, allocator=memoryPool):
		assert A.ndim == 2
		return matsum(A, 0 if cols else 1, out, alpha, beta, allocator)

	global toVectorAddVector, addVectorToVector, dot, vectorL1Norm
	toVectorAddVector = wrapToVectorAddVector
	addVectorToVector = wrapAddVectorToVector
	dot = context.dot
	vectorL1Norm = context.l1norm

	global mulMatrixOnMatrix, sumOnMatrix
	mulMatrixOnMatrix = wrapGemm
	sumOnMatrix = wrapSumOnMatrix


def initOpenCL():
	from PuzzleLib.OpenCL.Wrappers import CLBlas

	global toVectorAddVector, addVectorToVector, dot, vectorL1Norm
	toVectorAddVector = CLBlas.toVectorAddVector
	addVectorToVector = CLBlas.addVectorToVector
	dot = CLBlas.dot
	vectorL1Norm = CLBlas.vectorL1Norm

	global mulMatrixOnMatrix, sumOnMatrix
	mulMatrixOnMatrix = CLBlas.mulMatrixOnMatrix
	sumOnMatrix = CLBlas.sumOnMatrix


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
