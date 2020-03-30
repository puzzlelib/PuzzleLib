from PuzzleLib import Config


addVecToMatBatch = None
argmaxBatch = None


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
	memoryPool, matmod = backend.memoryPool, backend.matmod

	def wrapAddVecToMatBatch(vec, mat, axis, out):
		return matmod.addVecToMat(vec, mat, axis, out, memoryPool)

	def wrapArgmaxBatch(tensor, axis):
		return matmod.argmax(tensor, axis, memoryPool)

	global addVecToMatBatch, argmaxBatch
	addVecToMatBatch = wrapAddVecToMatBatch
	argmaxBatch = wrapArgmaxBatch


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
