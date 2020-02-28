from PuzzleLib import Config


addVecToMat = None
argmax = None


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

	global addVecToMat, argmax
	addVecToMat = MatVec.addVecToMat
	argmax = MatVec.argmax


def initOpenCL():
	from PuzzleLib.OpenCL.Kernels import MatVec

	global addVecToMat, argmax
	addVecToMat = MatVec.addVecToMat
	argmax = MatVec.argmax


def initCPU():
	import numpy as np
	from PuzzleLib.CPU.CPUArray import CPUArray

	def wrapAddVecToMat(v, m, axis, out):
		if axis == 0:
			v = v[:, np.newaxis]
		elif axis == 1:
			v = v[np.newaxis, :]

		np.add(m.get(copy=False), v.get(copy=False), out=out.get(copy=False))

	def wrapArgmax(mats, axis):
		out = np.empty(mats.shape[:axis] + mats.shape[axis + 1:], dtype=np.int32)
		np.argmax(mats.get(copy=False), axis, out=out)

		return CPUArray(out.shape, out.dtype, data=out, acquire=True)

	global addVecToMat, argmax
	addVecToMat = wrapAddVecToMat
	argmax = wrapArgmax


autoinit()
