from PuzzleLib import Config


mulTensorOnVecGroup = None
sumOnTensorGroup = None
mulTensorBatch = None


def autoinit():
	if Config.backend == Config.Backend.cuda:
		initCuda()
	elif Config.backend == Config.Backend.opencl:
		initOpenCL()
	elif Config.isCPUBased(Config.backend):
		initCPU()
	else:
		raise Config.ConfigError(Config.backend)


def initCuda():
	from PuzzleLib.Cuda.Utils import memoryPool

	from PuzzleLib.Cuda.Wrappers.CuBlas import context, CuBlas
	from PuzzleLib.Cuda.Kernels.MatVec import matsum, matvec

	def wrapMulTensorOnVecGroup(tensor, vecs, out=None, formatT="bgp", transpT=False,
								alpha=1.0, beta=0.0, allocator=memoryPool):
		assert tensor.ndim == 3 and formatT == "gbp"
		axis = 0 if transpT else 1

		return matvec(tensor, vecs, axis, out, alpha, beta, allocator)

	def wrapSumOnTensorGroup(tensor, out=None, formatT="bgp", cols=True, alpha=1.0, beta=0.0, allocator=memoryPool):
		assert tensor.ndim == 3
		axis = (1 if formatT == "gbp" else 0) if cols else 2

		return matsum(tensor, axis, out, alpha, beta, allocator)

	def wrapMulTensorBatch(A, B, formatA="bgp", formatB="bgp", out=None, formatOut="bgp", transpA=False, transpB=False,
						   alpha=1.0, beta=0.0, allocator=memoryPool):
		formats = {
			"gbp": CuBlas.GROUPFORMAT_GBP,
			"bgp": CuBlas.GROUPFORMAT_BGP
		}

		formatA, formatB, formatOut = formats[formatA], formats[formatB], formats[formatOut]
		return context.gemmBatched(A, B, formatA, formatB, formatOut, transpA, transpB, alpha, beta, out, allocator)

	global mulTensorOnVecGroup, sumOnTensorGroup, mulTensorBatch
	mulTensorOnVecGroup = wrapMulTensorOnVecGroup
	sumOnTensorGroup = wrapSumOnTensorGroup
	mulTensorBatch = wrapMulTensorBatch


def initOpenCL():
	from PuzzleLib.OpenCL.Wrappers import CLBlasGroup

	global mulTensorOnVecGroup, sumOnTensorGroup, mulTensorBatch
	mulTensorOnVecGroup = CLBlasGroup.mulTensorOnVecGroup
	sumOnTensorGroup = CLBlasGroup.sumOnTensorGroup
	mulTensorBatch = CLBlasGroup.mulTensorBatch


def initCPU():
	pass


autoinit()
