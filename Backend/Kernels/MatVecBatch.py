from PuzzleLib import Config


addVecToMatBatch = None
argmaxBatch = None


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
	from PuzzleLib.Cuda.Kernels import MatVec

	global addVecToMatBatch, argmaxBatch
	addVecToMatBatch = MatVec.addVecToMat
	argmaxBatch = MatVec.argmax


def initOpenCL():
	from PuzzleLib.OpenCL.Kernels import MatVecBatch

	global addVecToMatBatch, argmaxBatch
	addVecToMatBatch = MatVecBatch.addVecToMatBatch
	argmaxBatch = MatVecBatch.argmaxBatch


def initCPU():
	import numpy as np
	from PuzzleLib.CPU.CPUArray import CPUArray

	def wrapArgmax(mats, axis):
		out = np.empty(mats.shape[:axis] + mats.shape[axis + 1:], dtype=np.int32)
		np.argmax(mats.get(copy=False), axis, out=out)

		return CPUArray(out.shape, out.dtype, data=out, acquire=True)

	global argmaxBatch
	argmaxBatch = wrapArgmax


autoinit()
